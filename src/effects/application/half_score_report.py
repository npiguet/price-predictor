"""A half's control score as report lines, for ``evaluate-effect-model``.

:func:`half_score_checks` renders what ``control_score.score_half`` scored
(:func:`evaluate_effect_model.run_reporting_half` calls both); the grouped
breakdowns beside it are ``breakdowns``'.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from effects.domain.check_label import CheckLabel
from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus
from effects.domain.selection_set import COMPONENTS, DEVIANCE, RULE_TESTS, deviance_by_head

if TYPE_CHECKING:
    from effects.application.control_score import ControlScore, StratumScore

#: Left-out item names a report line spells out before it counts the rest.
NAMED_LEFT_OUT = 12


@dataclass(frozen=True, slots=True)
class HalfReading:
    """Which half a control score read and what the report sets beside it.

    ``games`` is each stratum's games in the half; ``selection_control`` the
    control score of the epoch the checkpoint holds, ``note`` a remark the
    control line carries.
    """

    half: str
    games: Mapping[str, int]
    selection_control: float | None = None
    note: str = ""

    @property
    def prefix(self) -> str:
        return f"{self.half}-half"


def half_score_checks(score: ControlScore, reading: HalfReading) -> list[CheckResult]:
    """One half of every stratum, scored as the control score scores the
    selection half (FR-070): per stratum its four components and their mean,
    then each component's items, the rule tests per rule (FR-074), and last
    the control score the strata mix to beside the selection half's of the
    epoch the checkpoint holds."""
    checks: list[CheckResult] = []
    for stratum, scored in sorted(score.strata.items()):
        checks += _stratum_checks(stratum, scored, reading)
    checks += _control_checks(score, reading)
    return checks


def _stratum_checks(stratum: str, scored: StratumScore, reading: HalfReading) -> list[CheckResult]:
    """One stratum's line, then a line per component it scored."""
    where = CheckLabel.of(reading.prefix, stratum)
    checks = [CheckResult(
        str(where), CheckStatus.REPORTED,
        _stratum_detail(scored, reading.games.get(stratum, 0)),
        {**dict(scored.components), "score": scored.score})]
    for component in COMPONENTS:
        if component in scored.components:
            checks.append(_component_check(stratum, scored, component, reading))
    return checks


def _stratum_detail(scored: StratumScore, games: int) -> str:
    missing = [name for name in COMPONENTS if name in scored.components
               and math.isnan(scored.components[name])]
    detail = f"{games} games; score = the mean of the four components"
    if missing:
        detail += f"; nothing to score in {', '.join(missing)}"
    if scored.non_finite:
        detail += f"; {scored.non_finite} non-finite reading(s) scored at -1"
    return detail


def _component_check(
    stratum: str, scored: StratumScore, component: str, reading: HalfReading,
) -> CheckResult:
    """One component's items; the rule tests are named by the rule-tests check."""
    items = dict(scored.items.get(component, {}))
    left_out = tuple(scored.left_out.get(component, ()))
    if component == RULE_TESTS:
        name = CheckLabel.of(CheckName.RULE_TESTS, stratum, reading.half)
    else:
        name = CheckLabel.of(reading.prefix, stratum).then(component)
    return CheckResult(str(name), CheckStatus.REPORTED,
                       _items_detail(component, items, left_out), items)


def _control_checks(score: ControlScore, reading: HalfReading) -> list[CheckResult]:
    """The control score the strata mix to, beside the selection half's."""
    values = {"control": score.control} if len(score.strata) == 2 else {}
    if reading.selection_control is not None:
        values["selection_control"] = float(reading.selection_control)
    if not values:
        return []
    return [CheckResult(
        str(CheckLabel.of(reading.prefix, "control")), CheckStatus.REPORTED,
        _control_detail(score, reading), values)]


def _control_detail(score: ControlScore, reading: HalfReading) -> str:
    detail = f"stratum weight {score.stratum_weight:g} on the card-disjoint score"
    if reading.selection_control is not None:
        detail += "; selection_control is the control score of the epoch the checkpoint holds"
    if reading.note:
        detail += f"; {reading.note}"
    return detail


def _items_detail(component: str, items: Mapping[str, float], left_out: tuple[str, ...]) -> str:
    """How many items a component scored, each rescaled and clipped at -1, and
    which it left out."""
    unit = "rules" if component == RULE_TESTS else "items"
    detail = f"{len(items)} {unit} scored, rescaled and clipped below at -1"
    if component == DEVIANCE and items:
        heads = deviance_by_head(items)
        detail += "; by head: " + ", ".join(
            f"{head} {mean:.3f} ({count})" for head, (mean, count) in heads.items())
    if left_out:
        named = ", ".join(left_out[:NAMED_LEFT_OUT])
        more = len(left_out) - NAMED_LEFT_OUT
        detail += f"; {len(left_out)} left out: {named}" + (f" and {more} more" if more > 0 else "")
    return detail
