"""What one ``evaluate-effect-model`` run reports: named checks and their verdicts.

The report scripts (``scripts/effect_gen2_report``) parse what
:meth:`EvaluationReport.render` prints.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
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
    """The report's fixed check names. The report scripts parse the values, so
    a member is renamed freely and a value never is."""

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

    ``qualifying_records`` (distinct records) is what the threshold reads;
    ``observations`` the (record, carrier) perturbations the agreement averages.
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


class Bound(StrEnum):
    """Which side of its limit a margin's value must stay on."""

    FLOOR = "floor"
    CEILING = "ceiling"


@dataclass(frozen=True, slots=True)
class Margin:
    """One measured quantity a gate holds against a fixed limit.

    ``key`` names the value in the figures line, ``label`` in the prose;
    ``verb`` reads before the value there (``explains 41.0% > 30%``).
    """

    key: str
    label: str
    value: float
    limit: float
    bound: Bound
    value_format: str = ""
    limit_format: str = ""
    verb: str = ""

    @property
    def breached(self) -> bool:
        """Whether the value is on the wrong side of the limit; an unmeasured
        (NaN) margin fails rather than comparing False and passing."""
        if math.isnan(self.value):
            return True
        if self.bound is Bound.FLOOR:
            return self.value < self.limit
        return self.value > self.limit

    @property
    def failure(self) -> str:
        """The report's account of this margin breached."""
        if math.isnan(self.value):
            return f"{self.label} not measured"
        breach = "<" if self.bound is Bound.FLOOR else ">"
        verb = f"{self.verb} " if self.verb else ""
        return (
            f"{self.label} {verb}{self.value:{self.value_format}} {breach} "
            f"{self.limit:{self.limit_format}}"
        )


def gate_verdict(
    name: str, margins: Sequence[Margin], *, fail_prefix: str, pass_detail: str,
) -> CheckResult:
    """A gate that holds only when every one of its margins does; every value
    is reported and every breach named, in the order given."""
    values = {margin.key: margin.value for margin in margins}
    failures = [margin.failure for margin in margins if margin.breached]
    if failures:
        return CheckResult(
            name, CheckStatus.FAIL, f"{fail_prefix}: " + "; ".join(failures), values,
        )
    return CheckResult(name, CheckStatus.PASS, pass_detail, values)


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
        """Whether gates 1 and 3, the only checks that can fail, both hold.
        Reported; it never stops the command."""
        return not self.blocking_failures

    def render(self) -> str:
        lines = []
        for check in self.checks:
            lines.append(f"[{check.status.marker}] {check.name}: {check.detail}")
            # Every figure, whatever the verdict: a passing margin is a result too.
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
