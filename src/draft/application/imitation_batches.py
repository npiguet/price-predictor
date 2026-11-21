"""A padded, device-resident batch of gen-1 examples and its joint loss:
``imitation_weight·CE + critic_weight·MSE`` (FR-032, FR-033).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from draft.application.imitation_examples import DraftExample
from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_state import TYPE_PACK
from price_predictor.infrastructure.torch_training import masked_log_softmax


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
    target_token: torch.Tensor      # (B,) absolute index, -1 if inactive
    imitation_active: torch.Tensor  # (B,) bool
    critic_active: torch.Tensor     # (B,) bool
    critic_target: torch.Tensor     # (B,) standardized


@dataclass(frozen=True)
class BatchSource:
    """What every batch of one run is built from: the shared embedding table, the
    critic targets' standardization (``mean``, ``std``) and the device."""

    table: np.ndarray
    mean: float
    std: float
    device: torch.device


def _pad_tokens(batch: list[DraftExample], table: np.ndarray) -> tuple[np.ndarray, ...]:
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
        card_emb[i, :n] = table[ex.card_idx]  # materialize embeddings per batch
        type_idx[i, :n] = ex.type_idx
        packs_ago[i, :n] = ex.packs_ago
        pick_ago[i, :n] = ex.pick_ago
        card_mask[i, :n] = True
        pack_mask[i, :n] = ex.type_idx == TYPE_PACK
    return card_emb, type_idx, packs_ago, pick_ago, card_mask, pack_mask


def _column(
    batch: list[DraftExample], field: str, device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """One per-example field of the batch as a ``(B,)`` tensor."""
    return torch.tensor([getattr(ex, field) for ex in batch], device=device, dtype=dtype)


def _collate(batch: list[DraftExample], source: BatchSource) -> _Batch:
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
        target_token=_column(batch, "target_token", device),
        imitation_active=_column(batch, "imitation_active", device, torch.bool),
        critic_active=_column(batch, "critic_active", device, torch.bool),
        critic_target=torch.tensor(
            [(ex.critic_target - source.mean) / source.std for ex in batch],
            device=device, dtype=torch.float32,
        ),
    )


@dataclass
class _LossOut:
    total: torch.Tensor
    imitation: torch.Tensor
    critic: torch.Tensor


def _compute_loss(
    model: DraftAgentModel, batch: _Batch, imitation_weight: float, critic_weight: float,
) -> tuple[_LossOut, torch.Tensor, torch.Tensor]:
    """Return (losses, policy_logits, critic_pred)."""
    policy_logits, critic_pred = model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )
    device = policy_logits.device

    imitation = torch.zeros((), device=device)
    if bool(batch.imitation_active.any()):
        logp = masked_log_softmax(policy_logits, batch.pack_mask)  # (B, N)
        idx = batch.target_token.clamp(min=0).unsqueeze(1)
        nll = -logp.gather(1, idx).squeeze(1)  # (B,)
        active = batch.imitation_active
        imitation = nll[active].mean()

    critic = torch.zeros((), device=device)
    if bool(batch.critic_active.any()):
        active = batch.critic_active
        critic = F.mse_loss(critic_pred[active], batch.critic_target[active])

    total = imitation_weight * imitation + critic_weight * critic
    return _LossOut(total, imitation, critic), policy_logits, critic_pred
