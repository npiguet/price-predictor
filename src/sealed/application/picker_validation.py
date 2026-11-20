"""The ``train-picker`` validation: the deterministic chosen deck of every
validation pool, its mean reward, the cross-scorer audit and the decks'
distributional summaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sealed.application.picker_pools import PreparedPool, _collate
from sealed.application.picker_scoring import _score_chosen_decks
from sealed.domain.card_embedding_layout import (
    COLOR_FLAGS,
    FEATURE_COUNT,
    MANA_VALUE,
    POWER,
    TOUGHNESS,
)
from sealed.domain.picker_model import PickerModel, walk_pick_indices
from sealed.domain.scorer_model import SetTransformerScorer


@dataclass
class DistribSummary:
    colors_mean: float
    creatures_mean: float
    type_creature_share: float
    cmc_hist: list[float]  # 5 bins: CMC<=2, 3, 4, 5, 6+


def _feat(emb: np.ndarray, slot: int) -> float:
    return float(emb[-FEATURE_COUNT + slot])


def _distrib_summaries(
    pools: list[PreparedPool], chosen: list[list[int]],
) -> DistribSummary:
    if not pools:
        return DistribSummary(0.0, 0.0, 0.0, [0.0] * 5)
    color_start = -FEATURE_COUNT + COLOR_FLAGS.start
    color_stop = -FEATURE_COUNT + COLOR_FLAGS.stop
    colors_per_deck: list[int] = []
    creatures_per_deck: list[int] = []
    shares: list[float] = []
    hist = np.zeros(5, dtype=np.float64)
    n_decks = 0
    for i, idxs in enumerate(chosen):
        if not idxs:
            continue
        n_decks += 1
        embs = pools[i].embeddings[idxs]
        color_union = (embs[:, color_start:color_stop] > 0.5).any(axis=0).sum()
        colors_per_deck.append(int(color_union))
        is_creature = (embs[:, -FEATURE_COUNT + POWER] > 0) | (
            embs[:, -FEATURE_COUNT + TOUGHNESS] > 0
        )
        n_creatures = int(is_creature.sum())
        creatures_per_deck.append(n_creatures)
        shares.append(n_creatures / len(idxs))
        cmc = embs[:, -FEATURE_COUNT + MANA_VALUE]
        for v in cmc:
            if v <= 2:
                hist[0] += 1
            elif v <= 3:
                hist[1] += 1
            elif v <= 4:
                hist[2] += 1
            elif v <= 5:
                hist[3] += 1
            else:
                hist[4] += 1
    if n_decks == 0:
        return DistribSummary(0.0, 0.0, 0.0, [0.0] * 5)
    return DistribSummary(
        colors_mean=float(np.mean(colors_per_deck)),
        creatures_mean=float(np.mean(creatures_per_deck)),
        type_creature_share=float(np.mean(shares)),
        cmc_hist=[float(x / n_decks) for x in hist],
    )


@dataclass
class ValidationReport:
    mean_reward: float
    audit_corr: float | None
    distrib: DistribSummary


def _deterministic_chosen(
    model: PickerModel, pools: list[PreparedPool], batch_size: int, device: torch.device,
) -> list[list[int]]:
    """Run deterministic inference (FR-006) over every pool, return chosen indices."""
    chosen: list[list[int]] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(pools), batch_size):
            group = pools[start:start + batch_size]
            cards, mask, _land = _collate(group)
            cards_t = torch.from_numpy(cards).to(device)
            mask_t = torch.from_numpy(mask).to(device)
            logits, _ = model(cards_t, mask_t)
            for i, p in enumerate(group):
                n = p.num_cards
                row = logits[i, :n].cpu()
                chosen.append(walk_pick_indices(row, p.is_land))
    return chosen


def _validate(
    model: PickerModel,
    scorer: SetTransformerScorer,
    auditor: SetTransformerScorer | None,
    pools: list[PreparedPool],
    batch_size: int,
    device: torch.device,
) -> ValidationReport:
    if not pools:
        return ValidationReport(0.0, None, DistribSummary(0.0, 0.0, 0.0, [0.0] * 5))
    chosen = _deterministic_chosen(model, pools, batch_size, device)
    train_scores = _score_chosen_decks(scorer, pools, chosen, device)
    audit_corr: float | None = None
    if auditor is not None:
        audit_scores = _score_chosen_decks(auditor, pools, chosen, device)
        audit_corr = _audit_correlation(train_scores, audit_scores)
    distrib = _distrib_summaries(pools, chosen)
    return ValidationReport(float(train_scores.mean()), audit_corr, distrib)


def _audit_correlation(train_scores: np.ndarray, audit_scores: np.ndarray) -> float:
    """Spearman rank correlation between two scorers on the val decks (FR-030)."""
    from scipy.stats import spearmanr

    # A constant input has no ranks to correlate; say so rather than letting
    # scipy warn and return the same nan.
    if len(train_scores) < 2 or np.ptp(train_scores) == 0 or np.ptp(audit_scores) == 0:
        return float("nan")
    corr, _ = spearmanr(train_scores, audit_scores)
    return float(corr)
