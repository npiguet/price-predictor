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

A grid is scored record by record: each reading of a record edited into a
cell is compared with the rule's answer for that record there, so a model
reading each cell's base rate errs wherever the answer differs between the
cell's records.

The standard errors are worst cases, computed at the freeze from labels alone,
before any model has scored an item; each is on the rescaled scale:

- an AUC: Hanley–McNeil at an AUC of 0.5, times two for the rescaling
  (:func:`auc_standard_error`);
- a grid: each cell's mean absolute error, over ``N_c`` records each erring
  by a value in [0, 1], varies by at most ``1/(2√N_c)``, averaged over the
  cells (which an edit grid makes fully correlated) and divided by the grid
  mean's MAE (:func:`grid_standard_error`);
- an explained deviance: the loss ratio, clustered by game
  (:func:`loss_ratio_standard_error`).

A component averages its m items equally unless its items group (the grid
component's variants, into the grid they are of): then it is the mean of its
groups' means, as :func:`grouped_mean` scores it. Its standard error is
``√(Σ SEᵢ²) / m`` for equal items (:func:`component_standard_error`) and
``√(Σ_g (Σ_{i∈g} SEᵢ²) / m_g²) / G`` for G groups of m_g items
(:func:`grouped_standard_error`), the items' errors taken as independent.

Torch-free, and free of scikit-learn: ``build-corpus`` and the freeze import
it, and the probe scripts score through it so a probe and a control score
compute one number one way.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

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


def _cells(values) -> list[np.ndarray]:
    """A grid's per-cell values as one array a cell."""
    return [np.asarray(cell, dtype=np.float64).ravel() for cell in values]


def grid_mean_error(answers) -> float:
    """``B = (1/C) Σ_c (1/N_c) Σ_r |ȳ − y_rc|``: the mean absolute error of
    predicting every record in every cell the grid's mean rule outcome ``ȳ``
    (the mean of its cells' mean answers). ``answers`` hold, per cell, the
    rule's answer for each of the grid's records; NaN with no answer."""
    cells = [cell for cell in _cells(answers) if cell.size]
    if not cells:
        return float("nan")
    mean = float(np.mean([cell.mean() for cell in cells]))
    return float(np.mean([np.abs(mean - cell).mean() for cell in cells]))


def grid_error(predicted, answers) -> float:
    """A grid's MAE, ``(1/C) Σ_c mean_r |p_rc − y_rc|``: ``predicted`` and
    ``answers`` hold, per cell, each record's reading and the rule's answer
    for that record. Infinite when any reading is not finite (a model that
    overflows is worse than any finite error); NaN when a cell holds no
    record, which is nothing to score."""
    errors, unreadable = [], False
    for read, answer in zip(_cells(predicted), _cells(answers), strict=True):
        if not read.size:
            return float("nan")
        unreadable = unreadable or not np.isfinite(read).all()
        errors.append(float(np.abs(read - answer).mean()))
    if not errors:
        return float("nan")
    return float("inf") if unreadable else float(np.mean(errors))


def grid_score(predicted, answers) -> float:
    """A grid on the common scale, ``1 − MAE / B``: the MAE is
    :func:`grid_error`, each record's reading against that record's answer,
    and ``B`` is :func:`grid_mean_error` over the same answers. NaN when the
    answer is the same for every record in every cell (``B = 0``), a grid the
    precision target leaves out (:func:`grid_standard_error` is infinite
    there); :data:`SCORE_FLOOR` when a reading is not finite."""
    baseline = grid_mean_error(answers)
    if not baseline > 0:
        return float("nan")
    error = grid_error(predicted, answers)
    return SCORE_FLOOR if math.isinf(error) else 1.0 - error / baseline


def non_finite(values) -> int:
    """How many of a model's readings ``values`` are not finite."""
    return int(np.count_nonzero(~np.isfinite(np.asarray(values, dtype=np.float64))))


def grouped_mean(scores: Mapping[str, float], group: Callable[[str], str]) -> float:
    """The mean over ``group``'s groups of each group's mean score, so a
    group counts once whatever its number of items; NaN with no score."""
    by_group: dict[str, list[float]] = defaultdict(list)
    for name, value in scores.items():
        by_group[group(name)].append(float(value))
    if not by_group:
        return float("nan")
    return float(np.mean([np.mean(values) for values in by_group.values()]))


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


def clustered_auc_standard_error(positive_games: Iterable[Hashable],
                                 negative_games: Iterable[Hashable]) -> float:
    """:func:`auc_standard_error` clustered by game (research E4): ``n₊`` and
    ``n₋`` are the distinct games among the positive rows and among the
    negative rows, whose game ids are given, so the records of one game count
    once."""
    return auc_standard_error(len(set(positive_games)), len(set(negative_games)))


def clustered_mean_standard_error(values: Iterable[float], games: Iterable[Hashable]) -> float:
    """The standard error of the mean of ``values``, clustered by game (CR0):
    ``√(Σ_g (Σ_{i∈g} (x_i − x̄))²) / N``, ``games`` giving each value's game,
    so the values of one game, which share a deck and a draw, count as one
    cluster. NaN with no value."""
    values, games = [float(value) for value in values], list(games)
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    by_game: dict[Hashable, float] = defaultdict(float)
    for game, value in zip(games, values, strict=True):
        by_game[game] += value - mean
    return math.sqrt(sum(s * s for s in by_game.values())) / len(values)


def grid_standard_error(answers) -> float:
    """A grid's worst-case standard error on the rescaled scale,
    ``(1/C) Σ_c 1/(2√N_c) / B``: ``answers`` hold, per cell, the rule's
    answer for each of its ``N_c`` records, and ``B`` is
    :func:`grid_mean_error` over them. Infinite when a cell is empty or the
    answer is the same everywhere (``B = 0``)."""
    sizes = [cell.size for cell in _cells(answers)]
    if not sizes or min(sizes) <= 0:
        return math.inf
    baseline = grid_mean_error(answers)
    if not baseline > 0:
        return math.inf
    return sum(1 / (2 * math.sqrt(n)) for n in sizes) / len(sizes) / baseline


def loss_ratio_standard_error(
    games: Iterable[Hashable], losses: Iterable[float], weights: Iterable[float] | None = None,
) -> float:
    """An explained deviance's worst-case standard error, ``√(Σ_g S_g²) / Σ_g S_g``,
    ``S_g`` summing ``w_i·ℓ_i`` over game g's records: ``ℓ_i`` the constant
    prediction's loss on record i and ``w_i`` its weight in the component
    (a deviance copy's rule-mask weight times its class weight). It is
    ``1/√n_eff``, ``n_eff = (Σ_g S_g)² / Σ_g S_g²`` Kish's effective number
    of games. Infinite when the records carry no weighted loss."""
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


def grouped_standard_error(errors: Mapping[str, float], group: Callable[[str], str]) -> float:
    """The standard error of :func:`grouped_mean` over items whose standard
    errors are ``errors``, taken as independent: each group's mean has
    ``√(Σ_{i∈g} SEᵢ²) / m_g``, and the mean of the G groups has
    ``√(Σ_g SE_g²) / G``. Infinite with no item."""
    by_group: dict[str, list[float]] = defaultdict(list)
    for name, error in errors.items():
        by_group[group(name)].append(error)
    if not by_group:
        return math.inf
    return math.sqrt(sum(component_standard_error(group_errors) ** 2
                         for group_errors in by_group.values())) / len(by_group)
