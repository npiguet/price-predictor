"""The control score's items on the common scale: a probe type's, a rule
test's and a deviance term's score, a component's mean of its clipped items,
and the two strata's mix (``control_score``; contracts/selection-set.md
§ Score)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from effects.domain.grid_edits import grid_of
from effects.domain.probe_readouts import item_score
from effects.domain.score_scales import (
    SCORE_FLOOR,
    auc,
    clipped,
    grouped_mean,
    non_finite,
    rescaled_auc,
)
from effects.domain.selection_set import BINARY, GRIDS
from effects.domain.strata import Half

#: The half selection reads; the reporting half is the selected checkpoint's.
SELECTION, REPORTING = str(Half.SELECTION), str(Half.REPORTING)


class ControlScoreError(RuntimeError):
    """A component with nothing to score in one stratum's half (FR-068c)."""

    def __init__(self, stratum: str, half: str, component: str, detail: str) -> None:
        self.stratum, self.half, self.component = stratum, half, component
        super().__init__(f"{stratum} {half}, {component}: {detail}")


def mixed(card_disjoint: float, game_disjoint: float, weight: float) -> float:
    """The control score: ``weight · card_disjoint + (1 − weight) · game_disjoint``."""
    return weight * card_disjoint + (1.0 - weight) * game_disjoint


#: How a component's items group before they are averaged: a grid's
#: variants into the grid they are of, so the damage, combat and evasion grids
#: weigh equally (root spec § 14.1); every other item is a group of its own.
ITEM_GROUPS: dict[str, Callable[[str], str]] = {GRIDS: grid_of}


@dataclass(frozen=True)
class StratumComponent:
    """One component of one stratum's half: what a :class:`ControlScoreError`
    names."""

    stratum: str
    component: str
    half: str = SELECTION

    def error(self, detail: str) -> ControlScoreError:
        return ControlScoreError(self.stratum, self.half, self.component, detail)


def component_score(items: Mapping[str, float], of: StratumComponent, *,
                    strict: bool = True) -> float:
    """The mean of a component's items, each clipped below at −1, grouped
    first by :data:`ITEM_GROUPS` (a group with no scored item left out);
    raises :class:`ControlScoreError` naming ``of`` when no item has a
    score, or is NaN then when not ``strict``."""
    scores = {name: clipped(value) for name, value in items.items() if not math.isnan(value)}
    if not scores:
        if not strict:
            return math.nan
        raise of.error("nothing to score this epoch")
    return grouped_mean(scores, ITEM_GROUPS.get(of.component, str))


def probe_item(kind: str, labels: Sequence[float], predictions: Sequence[float]) -> float:
    """A probe type on the common scale: ``2·AUC − 1`` for a binary one, R²
    for an amount; :data:`SCORE_FLOOR` when any prediction is not finite (an
    overflowing model is scored at the floor, never on the rows it can read);
    NaN where it cannot be scored."""
    if non_finite(predictions):
        return SCORE_FLOOR
    score = item_score(kind, labels, predictions)
    return rescaled_auc(score) if kind == BINARY and not math.isnan(score) else score


def rule_item(outputs: Iterable[tuple[Sequence[float], Sequence[float]]]) -> float:
    """A rule test on the common scale: the mean over its outputs of
    ``2·AUC − 1``, ``fires`` ranked above ``does_not_fire``; each output is
    ``(fires readings, does_not_fire readings)``, and one without a reading
    of both polarities is left out. NaN with none left; :data:`SCORE_FLOOR`
    when any reading is not finite."""
    scores = []
    for fires, does_not in outputs:
        if non_finite(fires) or non_finite(does_not):
            return SCORE_FLOOR
        value = auc([True] * len(fires) + [False] * len(does_not), [*fires, *does_not])
        if not math.isnan(value):
            scores.append(rescaled_auc(value))
    return float(np.mean(scores)) if scores else math.nan


def explained_deviance(parts: Mapping[str, float], floor: Mapping[str, float], *,
                       kept: Iterable[str], deviance_floor: float) -> dict[str, float]:
    """``1 − loss / constant loss`` per item of ``kept`` the pass scored (an
    output term of any shipped head, ``selection_set.deviance_item``), leaving
    out each whose constant-prediction loss is at or below ``deviance_floor``
    (FR-068). A term whose loss is not finite scores :data:`SCORE_FLOOR`."""
    out = {}
    for name in kept:
        loss, constant = parts.get(name), floor.get(name)
        if _unscored(loss, constant, deviance_floor):
            continue
        out[name] = 1.0 - loss / constant if math.isfinite(loss) else SCORE_FLOOR
    return out


def _unscored(loss: float | None, constant: float | None, deviance_floor: float) -> bool:
    """Whether a term is left out: the pass scored no loss or no constant
    loss for it, or its constant loss is at or below the floor."""
    return loss is None or constant is None or constant <= deviance_floor
