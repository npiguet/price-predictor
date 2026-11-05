"""The scores a selection set is read on, their common scale and their worst-case
standard errors (FR-068b; research E1, E4; contracts/selection-set.md).

Four components make a stratum's score, each item put on a scale where 0 is
chance or a constant prediction and 1 is perfect:

=========================  =============================================
item                       rescaled
=========================  =============================================
AUC (probe, rule test)     ``2·AUC − 1`` (:func:`rescaled_auc`)
R² (probe)                 as is
explained deviance         as is
grid                       ``1 − MAE / MAE(grid mean)`` (:func:`grid_score`)
=========================  =============================================

and every item is clipped below at −1 (:func:`clipped`), so one item a model
gets badly wrong cannot outweigh the rest of its component.

The standard errors are worst cases, computed at the freeze from labels alone,
before any model has scored an item; each is on the rescaled scale:

- an AUC: Hanley–McNeil at an AUC of 0.5, times two for the rescaling
  (:func:`auc_standard_error`);
- a grid: each cell mean of predictions in [0, 1] varies by at most
  ``1/(2√N_c)`` and the MAE is 1-Lipschitz in each, averaged over the cells
  (which an edit grid makes fully correlated) and divided by the grid mean's
  MAE (:func:`grid_standard_error`);
- an explained deviance: the loss ratio, clustered by game
  (:func:`loss_ratio_standard_error`).

A component averages its m items equally, so its standard error is
``√(Σ SEᵢ²) / m`` (:func:`component_standard_error`).

Torch-free, and free of scikit-learn: ``build-corpus`` and the freeze import
it, and the probe scripts score through it so a probe and a control score
compute one number one way.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

#: The floor every rescaled item is clipped to.
SCORE_FLOOR = -1.0


# ── scores ──────────────────────────────────────────────────────────────


def auc(labels, scores, weight=None) -> float:
    """ROC AUC of ``scores`` against yes/no ``labels``: the (weighted) share of
    (yes, no) pairs the scores rank yes first, a tie counting half.

    Rows whose score or weight is not finite are dropped first. NaN when what
    is left lacks either label (or either label's weight).
    """
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    weight = (np.ones(len(scores)) if weight is None
              else np.asarray(weight, dtype=np.float64))
    keep = np.isfinite(scores) & np.isfinite(weight)
    labels, scores, weight = labels[keep], scores[keep], weight[keep]
    positive = np.where(labels, weight, 0.0)
    negative = np.where(labels, 0.0, weight)
    total_positive, total_negative = positive.sum(), negative.sum()
    if total_positive <= 0 or total_negative <= 0:
        return float("nan")
    unique, at = np.unique(scores, return_inverse=True)
    positive_at = np.bincount(at, weights=positive, minlength=len(unique))
    negative_at = np.bincount(at, weights=negative, minlength=len(unique))
    negative_below = np.cumsum(negative_at) - negative_at
    ranked_first = (positive_at * (negative_below + 0.5 * negative_at)).sum()
    return float(ranked_first / (total_positive * total_negative))


def r_squared(y, prediction, weight=None) -> float:
    """Weighted ``1 − SS_res / SS_tot`` about the weighted mean of ``y``; NaN
    when ``y`` does not vary."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weight = np.ones_like(y) if weight is None else np.asarray(weight, dtype=np.float64)
    mean = float((weight * y).sum() / weight.sum())
    ss_res = float((weight * (y - prediction) ** 2).sum())
    ss_tot = float((weight * (y - mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def grid_mean_error(outcomes) -> float:
    """``B = (1/C) Σ_c |ȳ − y_c|``: the mean absolute error of predicting every
    cell the grid's mean rule outcome."""
    outcomes = np.asarray(outcomes, dtype=np.float64)
    return float(np.abs(outcomes.mean() - outcomes).mean())


def grid_score(predicted, outcomes) -> float:
    """A grid on the common scale, ``1 − MAE / B``: ``predicted`` holds each
    cell's mean predicted probability, ``outcomes`` the rule's outcome in that
    cell, and ``B`` is :func:`grid_mean_error`. NaN when the outcome is
    constant over the cells (``B = 0``), a grid the precision target leaves
    out (:func:`grid_standard_error` is infinite there)."""
    predicted = np.asarray(predicted, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    baseline = grid_mean_error(outcomes)
    if baseline <= 0:
        return float("nan")
    return 1.0 - float(np.abs(predicted - outcomes).mean()) / baseline


def rescaled_auc(value: float) -> float:
    """An AUC on the common scale: ``2·AUC − 1``, 0 at chance."""
    return 2.0 * value - 1.0


def clipped(value: float) -> float:
    """``value`` clipped below at :data:`SCORE_FLOOR`; NaN stays NaN."""
    return value if math.isnan(value) else max(value, SCORE_FLOOR)


# ── worst-case standard errors (research E4) ────────────────────────────


def auc_standard_error(positives: int, negatives: int) -> float:
    """A rule test's or an AUC probe's worst-case standard error on the
    rescaled scale: Hanley–McNeil at an AUC of 0.5, times two,
    ``2·√((n₊ + n₋ + 1) / (12·n₊·n₋))``; infinite without both labels."""
    if positives <= 0 or negatives <= 0:
        return math.inf
    return 2 * math.sqrt((positives + negatives + 1) / (12 * positives * negatives))


def grid_standard_error(cell_records: Sequence[int], outcomes: Sequence[float]) -> float:
    """A grid's worst-case standard error on the rescaled scale,
    ``(1/C) Σ_c 1/(2√N_c) / B``, over cells of ``cell_records`` records each
    and rule outcomes ``outcomes``; infinite when a cell is empty or the
    outcome is constant (``B = 0``)."""
    if len(cell_records) != len(outcomes):
        raise ValueError(f"{len(cell_records)} cell sizes against {len(outcomes)} outcomes")
    if not cell_records or min(cell_records) <= 0:
        return math.inf
    baseline = grid_mean_error(outcomes)
    if baseline <= 0:
        return math.inf
    spread = sum(1 / (2 * math.sqrt(n)) for n in cell_records) / len(cell_records)
    return spread / baseline


def loss_ratio_standard_error(
    games: Iterable[Hashable], losses: Iterable[float], weights: Iterable[float] | None = None,
) -> float:
    """An explained deviance's worst-case standard error, ``√(Σ_g S_g²) / Σ_g S_g``,
    ``S_g`` summing ``w_i·ℓ_i`` over game g's records: ``ℓ_i`` the constant
    prediction's loss on record i and ``w_i`` its weight in the component.
    Infinite when the records carry no weighted loss."""
    games, losses = list(games), list(losses)
    weights = [1.0] * len(losses) if weights is None else list(weights)
    by_game: dict[Hashable, float] = defaultdict(float)
    for game, loss, weight in zip(games, losses, weights, strict=True):
        by_game[game] += weight * loss
    total = sum(by_game.values())
    if total <= 0:
        return math.inf
    return math.sqrt(sum(s * s for s in by_game.values())) / total


def component_standard_error(errors: Sequence[float]) -> float:
    """The standard error of a component averaging its items equally,
    ``√(Σ SEᵢ²) / m``; infinite with no item."""
    if not errors:
        return math.inf
    return math.sqrt(sum(e * e for e in errors)) / len(errors)
