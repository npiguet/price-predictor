"""A padded, device-resident batch of gen-2 RL examples, the two-headed forward
and the RL loss: policy gradient, critic MSE, entropy bonus and KL anchor
(data-model §4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from draft.application.imitation_batches import BatchSource
from draft.application.rl_examples import RLExample
from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_state import TYPE_PACK
from price_predictor.infrastructure.torch_training import (
    kl_divergence,
    masked_log_softmax,
    policy_entropy,
)

if TYPE_CHECKING:
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig


@dataclass
class _Batch:
    card_emb: torch.Tensor
    type_idx: torch.Tensor
    packs_ago: torch.Tensor
    pick_ago: torch.Tensor
    card_mask: torch.Tensor
    pack_mask: torch.Tensor
    pack_number: torch.Tensor
    pick_number: torch.Tensor
    action_token: torch.Tensor
    learner_active: torch.Tensor
    critic_active: torch.Tensor
    reward_std: torch.Tensor
    advantage: torch.Tensor


@dataclass(frozen=True)
class PolicyPair:
    """The model being trained and the frozen KL reference it is anchored to."""

    model: DraftAgentModel
    reference: DraftAgentModel


@dataclass(frozen=True)
class LossSettings:
    """What weighs the RL loss's terms: the run's config (rollout temperature,
    value weight) and the entropy and KL coefficients at this point of the run."""

    config: TrainDraftAgentRLConfig
    entropy_coef: float
    kl_coef: float


def _pad_tokens(batch: list[RLExample], table: np.ndarray) -> tuple[np.ndarray, ...]:
    """The batch's token arrays, right-padded to its longest example: card
    embeddings, type, packs-ago, pick-ago, the card mask and the PACK mask."""
    b = len(batch)
    max_n = max(ex.n_tokens for ex in batch)
    dim = table.shape[1]
    card_emb = np.zeros((b, max_n, dim), dtype=np.float32)
    type_idx = np.zeros((b, max_n), dtype=np.int64)
    packs_ago = np.zeros((b, max_n), dtype=np.int64)
    pick_ago = np.zeros((b, max_n), dtype=np.int64)
    card_mask = np.zeros((b, max_n), dtype=bool)
    pack_mask = np.zeros((b, max_n), dtype=bool)
    for i, ex in enumerate(batch):
        n = ex.n_tokens
        card_emb[i, :n] = table[ex.card_idx]
        type_idx[i, :n] = ex.type_idx
        packs_ago[i, :n] = ex.packs_ago
        pick_ago[i, :n] = ex.pick_ago
        card_mask[i, :n] = True
        pack_mask[i, :n] = ex.type_idx == TYPE_PACK
    return card_emb, type_idx, packs_ago, pick_ago, card_mask, pack_mask


def _column(
    batch: list[RLExample], field: str, device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One per-example field of the batch as a ``(B,)`` tensor."""
    return torch.tensor([getattr(ex, field) for ex in batch], device=device, dtype=dtype)


def _collate(batch: list[RLExample], source: BatchSource) -> _Batch:
    device = source.device
    card_emb, type_idx, packs_ago, pick_ago, card_mask, pack_mask = _pad_tokens(
        batch, source.table,
    )
    return _Batch(
        card_emb=torch.from_numpy(card_emb).to(device),
        type_idx=torch.from_numpy(type_idx).to(device),
        packs_ago=torch.from_numpy(packs_ago).to(device),
        pick_ago=torch.from_numpy(pick_ago).to(device),
        card_mask=torch.from_numpy(card_mask).to(device),
        pack_mask=torch.from_numpy(pack_mask).to(device),
        pack_number=_column(batch, "pack_number", device),
        pick_number=_column(batch, "pick_number", device),
        action_token=_column(batch, "action_token", device),
        learner_active=_column(batch, "learner_active", device, torch.bool),
        critic_active=_column(batch, "critic_active", device, torch.bool),
        reward_std=torch.tensor(
            [(ex.reward - source.mean) / source.std for ex in batch],
            device=device, dtype=torch.float32,
        ),
        advantage=_column(batch, "advantage", device, torch.float32),
    )


@dataclass
class _LossOut:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    kl: torch.Tensor


def _forward_logits_critic(
    model: DraftAgentModel, batch: _Batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )


def _compute_loss(models: PolicyPair, batch: _Batch, settings: LossSettings) -> _LossOut:
    """The RL loss (data-model §4). Distributions are π_T = softmax(logits / T)."""
    config = settings.config
    temp = float(config.rollout_temperature or 1.0)
    policy_logits, critic_pred = _forward_logits_critic(models.model, batch)
    device = policy_logits.device
    scaled = policy_logits / temp

    policy = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    kl = torch.zeros((), device=device)
    learner = batch.learner_active
    if bool(learner.any()):
        logp = masked_log_softmax(scaled, batch.pack_mask)  # (B, N)
        idx = batch.action_token.clamp(min=0).unsqueeze(1)
        logp_action = logp.gather(1, idx).squeeze(1)  # (B,)
        adv = batch.advantage[learner]
        policy = -(adv * logp_action[learner]).mean()
        entropy = policy_entropy(scaled, batch.pack_mask)[learner].mean()
        with torch.no_grad():
            ref_logits, _ = _forward_logits_critic(models.reference, batch)
        kl = kl_divergence(scaled, ref_logits / temp, batch.pack_mask)[learner].mean()

    value = torch.zeros((), device=device)
    if bool(batch.critic_active.any()):
        active = batch.critic_active
        value = F.mse_loss(critic_pred[active], batch.reward_std[active])

    total = (
        policy + config.value_weight * value
        - settings.entropy_coef * entropy + settings.kl_coef * kl
    )
    return _LossOut(total, policy, value, entropy, kl)
