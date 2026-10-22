"""What one ``evaluate-effect-model`` run reports: named checks and their verdicts.

Every gate and every reported check produces a :class:`CheckResult`, and the
report renders them one line each, in the order they ran. Kept in the domain
and free of torch, because the gates, the geometry checks and the breakdowns
all build these values, and the report scripts (``scripts/effect_gen2_report``)
parse what :meth:`EvaluationReport.render` prints.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    #: The records or the cache this check reads are absent from this run.
    SKIPPED = "skipped"
    #: Reported, gates nothing.
    REPORTED = "reported"
    #: Specified but never built; the line says what it would measure.
    NOT_IMPLEMENTED = "not-implemented"

    @property
    def marker(self) -> str:
        """The four-character tag a report line opens with."""
        return _MARKERS[self]


_MARKERS = {
    CheckStatus.PASS: "PASS",
    CheckStatus.FAIL: "FAIL",
    CheckStatus.SKIPPED: "skip",
    CheckStatus.REPORTED: "----",
    CheckStatus.NOT_IMPLEMENTED: "todo",
}


class CheckName(StrEnum):
    """The report's fixed check names.

    The values are what the rendered report prints and what the report scripts
    (``scripts/effect_gen2_report``) parse, so a member is renamed freely and a
    value never is. Breakdown lines compose their names from the stratum and
    group, and a baseline comparison from the baseline
    (:meth:`baseline_comparison`).
    """

    GATE_1 = "gate-1"
    GATE_2 = "gate-2"
    GATE_3 = "gate-3"
    WARD_CANARY = "ward-canary"
    NEAREST_NEIGHBOUR = "nearest-neighbour"
    UMAP = "umap"
    DECODABILITY = "decodability"
    BASELINE_COMPARISON = "comparison"
    BREAKDOWNS = "breakdowns"
    MEMORIZATION_GAP = "memorization-gap"
    ZERO_SHOT_KEYWORD = "zero-shot-keyword"
    MATCHED_REAL_VS_FORK = "matched-real-vs-fork"
    PROBE_DIFF = "probe-diff"
    ROLE_POLARITY = "role-polarity"

    @classmethod
    def baseline_comparison(cls, baseline: str) -> str:
        """``no-state-comparison``: one baseline cache's geometry beside the shipping one's."""
        return f"{baseline}-{cls.BASELINE_COMPARISON}"


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    status: CheckStatus
    detail: str = ""
    values: dict[str, float] = field(default_factory=dict)

    @property
    def blocks(self) -> bool:
        return self.status is CheckStatus.FAIL


@dataclass(frozen=True, slots=True)
class KeywordVerdict:
    """One damage-step keyword's routing verdict from gate 2. Blocks nothing.

    ``qualifying_records`` counts distinct records and is what the threshold
    reads; ``observations`` counts the (record, carrier) perturbations the
    agreement averages, more than the records wherever one combat holds two
    carriers of the keyword.
    """

    keyword: str
    qualifying_records: int
    direction_agreement: float
    routed_to_probe: bool
    reason: str
    observations: int = 0

    @property
    def passed(self) -> bool:
        return not self.routed_to_probe


def margin(label: str, value: float, breached: bool, comparison: str) -> str | None:
    """One margin's failure, or None where it holds.

    A margin that could not be measured fails rather than passes: NaN compares
    False against every threshold, so without this an empty stratum or an
    all-zero embedding read as a gate that held.
    """
    if math.isnan(value):
        return f"{label} not measured"
    return f"{label} {comparison}" if breached else None


def breaches(*margins: str | None) -> list[str]:
    """The margins that failed, in the order given."""
    return [found for found in margins if found is not None]


@dataclass
class EvaluationReport:
    """Everything one evaluation run produced."""

    checks: list[CheckResult] = field(default_factory=list)
    keyword_verdicts: list[KeywordVerdict] = field(default_factory=list)

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        return result

    def extend(self, results: Iterable[CheckResult]) -> None:
        self.checks.extend(results)

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.blocks]

    @property
    def ships(self) -> bool:
        """Whether gates 1 and 3 both hold.

        Only gates 1 and 3 can answer no. Gate 2 is routing information and the
        remaining checks are diagnostics. The verdict is reported; it never
        stops the command.
        """
        return not self.blocking_failures

    def render(self) -> str:
        lines = []
        for check in self.checks:
            lines.append(f"[{check.status.marker}] {check.name}: {check.detail}")
            # Every figure a check measured, whatever its verdict: the design
            # record's Outcome section needs a passing margin as much as a
            # failing one.
            if check.values:
                lines.append("       " + "  ".join(
                    f"{name}={value:+.4f}"
                    for name, value in check.values.items()
                ))
        for verdict in self.keyword_verdicts:
            state = "pass" if verdict.passed else "→ probe"
            lines.append(
                f"    {verdict.keyword:<16} {state:<8} "
                f"n={verdict.qualifying_records:<6} {verdict.reason}"
            )
        lines.append(
            "SHIPS" if self.ships
            else "GATES FAILED: " + ", ".join(c.name for c in self.blocking_failures)
        )
        return "\n".join(lines)
