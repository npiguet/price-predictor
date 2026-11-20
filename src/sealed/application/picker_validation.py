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


_COLOR_START = -FEATURE_COUNT + COLOR_FLAGS.start
_COLOR_STOP = -FEATURE_COUNT + COLOR_FLAGS.stop


def _empty_distrib() -> DistribSummary:
    return DistribSummary(0.0, 0.0, 0.0, [0.0] * 5)


def _cmc_bin(v: float) -> int:
    """The CMC histogram bin of one mana value: <=2, 3, 4, 5, 6+."""
    if v <= 2:
        return 0
    if v <= 3:
        return 1
    if v <= 4:
        return 2
    if v <= 5:
        return 3
    return 4


class _DistribTally:
    """The per-deck counts ``_distrib_summaries`` averages over the decks."""

    def __init__(self) -> None:
        self.colors_per_deck: list[int] = []
        self.creatures_per_deck: list[int] = []
        self.shares: list[float] = []
        self.hist = np.zeros(5, dtype=np.float64)
        self.n_decks = 0

    def add(self, embs: np.ndarray, n_cards: int) -> None:
        """Count one non-empty chosen deck, its cards' embeddings ``embs``."""
        self.n_decks += 1
        color_union = (embs[:, _COLOR_START:_COLOR_STOP] > 0.5).any(axis=0).sum()
        self.colors_per_deck.append(int(color_union))
        is_creature = (embs[:, -FEATURE_COUNT + POWER] > 0) | (
            embs[:, -FEATURE_COUNT + TOUGHNESS] > 0
        )
        n_creatures = int(is_creature.sum())
        self.creatures_per_deck.append(n_creatures)
        self.shares.append(n_creatures / n_cards)
        for v in embs[:, -FEATURE_COUNT + MANA_VALUE]:
            self.hist[_cmc_bin(v)] += 1

    def summary(self) -> DistribSummary:
        if self.n_decks == 0:
            return _empty_distrib()
        return DistribSummary(
            colors_mean=float(np.mean(self.colors_per_deck)),
            creatures_mean=float(np.mean(self.creatures_per_deck)),
            type_creature_share=float(np.mean(self.shares)),
            cmc_hist=[float(x / self.n_decks) for x in self.hist],
        )


def _distrib_summaries(
    pools: list[PreparedPool], chosen: list[list[int]],
) -> DistribSummary:
    if not pools:
        return _empty_distrib()
    tally = _DistribTally()
    for i, idxs in enumerate(chosen):
        if idxs:
            tally.add(pools[i].embeddings[idxs], len(idxs))
    return tally.summary()


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


@dataclass(frozen=True)
class ValidationContext:
    """What every validation scores with: the training scorer, the optional
    auditor, the inference batch size and the device."""

    scorer: SetTransformerScorer
    auditor: SetTransformerScorer | None
    batch_size: int
    device: torch.device


def _validate(
    model: PickerModel, pools: list[PreparedPool], context: ValidationContext,
) -> ValidationReport:
    scorer, auditor = context.scorer, context.auditor
    batch_size, device = context.batch_size, context.device
    if not pools:
        return ValidationReport(0.0, None, _empty_distrib())
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
    if _has_no_ranks(train_scores, audit_scores):
        return float("nan")
    corr, _ = spearmanr(train_scores, audit_scores)
    return float(corr)


def _has_no_ranks(train_scores: np.ndarray, audit_scores: np.ndarray) -> bool:
    """Fewer than two decks, or either scorer constant over them."""
    return len(train_scores) < 2 or np.ptp(train_scores) == 0 or np.ptp(audit_scores) == 0
