"""Validation of the gen-1 agent over the held-out split: the joint loss, top-k
imitation accuracy and per-pack critic MSE (FR-037).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from draft.application.imitation_batches import BatchSource, _collate, _compute_loss
from draft.application.imitation_examples import DraftExample
from draft.domain.draft_agent_model import DraftAgentModel
from draft.domain.draft_state import TYPE_PACK

if TYPE_CHECKING:
    from draft.application.train_draft_agent import TrainDraftAgentConfig


def imitation_topk_accuracy(
    predictions: list[tuple[np.ndarray, int]], k: int,
) -> float:
    """Top-k accuracy over (pack-action logits, target index) pairs (FR-037)."""
    if not predictions:
        return float("nan")
    hits = 0
    for logits, target in predictions:
        order = np.argsort(-logits)[:k]
        if target in order:
            hits += 1
    return hits / len(predictions)


def per_pack_critic_mse(
    preds: list[float], targets: list[float], pack_numbers: list[int],
) -> dict[int, float]:
    """MSE between standardized critic preds and targets, sliced by pack number."""
    by_pack: dict[int, list[tuple[float, float]]] = {}
    for p, t, pk in zip(preds, targets, pack_numbers):
        by_pack.setdefault(pk, []).append((p, t))
    out: dict[int, float] = {}
    for pk, pairs in by_pack.items():
        diffs = np.asarray([p - t for p, t in pairs], dtype=np.float64)
        out[pk] = float((diffs ** 2).mean())
    return out


@dataclass
class _ValReport:
    loss: float
    imitation: float
    critic: float
    top1: float
    top3: float
    per_pack_mse: dict[int, float]


def _validate(
    model: DraftAgentModel,
    val: list[DraftExample],
    source: BatchSource,
    config: TrainDraftAgentConfig,
) -> _ValReport:
    model.eval()
    if not val:
        return _ValReport(0.0, 0.0, 0.0, float("nan"), float("nan"), {})
    imitation_preds: list[tuple[np.ndarray, int]] = []
    critic_preds: list[float] = []
    critic_targets: list[float] = []
    critic_packs: list[int] = []
    tot_imit = 0.0
    tot_crit = 0.0
    n_imit = 0
    n_crit = 0
    mean, std = source.mean, source.std
    with torch.no_grad():
        for start in range(0, len(val), config.batch_size):
            chunk = val[start:start + config.batch_size]
            batch = _collate(chunk, source)
            losses, logits, critic_pred = _compute_loss(
                model, batch, config.imitation_weight, config.critic_weight,
            )
            for i, ex in enumerate(chunk):
                if ex.imitation_active:
                    pack_idx = np.flatnonzero(ex.type_idx == TYPE_PACK)
                    row = logits[i, :ex.n_tokens].cpu().numpy()
                    pack_logits = row[pack_idx]
                    target_in_pack = int(np.flatnonzero(pack_idx == ex.target_token)[0])
                    imitation_preds.append((pack_logits, target_in_pack))
                if ex.critic_active:
                    critic_preds.append(float(critic_pred[i].cpu()))
                    critic_targets.append((ex.critic_target - mean) / std)
                    critic_packs.append(ex.pack_number)
            if bool(batch.imitation_active.any()):
                tot_imit += float(losses.imitation) * int(batch.imitation_active.sum())
                n_imit += int(batch.imitation_active.sum())
            if bool(batch.critic_active.any()):
                tot_crit += float(losses.critic) * int(batch.critic_active.sum())
                n_crit += int(batch.critic_active.sum())
    imit = tot_imit / n_imit if n_imit else 0.0
    crit = tot_crit / n_crit if n_crit else 0.0
    return _ValReport(
        loss=config.imitation_weight * imit + config.critic_weight * crit,
        imitation=imit,
        critic=crit,
        top1=imitation_topk_accuracy(imitation_preds, 1),
        top3=imitation_topk_accuracy(imitation_preds, 3),
        per_pack_mse=per_pack_critic_mse(critic_preds, critic_targets, critic_packs),
    )
