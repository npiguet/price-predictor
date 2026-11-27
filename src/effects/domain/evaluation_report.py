"""What one ``evaluate-effect-model`` run reports: named checks and their verdicts.

The report scripts (``scripts/effect_gen2_report``) parse what
:meth:`EvaluationReport.render` prints.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import chain

from effects.domain.check_label import TableRow


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
    REPORTING_HALF = "reporting-half"
    RULE_TESTS = "rule-tests"
    MATCHED_REAL_VS_FORK = "matched-real-vs-fork"
    PROBE_DIFF = "probe-diff"
    ROLE_POLARITY = "role-polarity"
    SCALING_CALIBRATION = "scaling-calibration"
    IDENTITY_GAME_DISJOINT = "identity-game-disjoint"
    STATE_ONLY_FLOOR = "state-only-floor"
    SHARED_TEXT_STRATUM = "shared-text-stratum"
    NOVEL_COMBINATION_STRATUM = "novel-combination-stratum"
    NUMERIC_EXTRAPOLATION_STRATUM = "numeric-extrapolation-stratum"

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
    #: Figures the line carries beside its values: written to the run's table
    #: file and never rendered.
    table: tuple[TableRow, ...] = ()

    @property
    def blocks(self) -> bool:
        return self.status is CheckStatus.FAIL


@dataclass(frozen=True, slots=True)
class EffectMoves:
    """How much of the right move one gate-2 effect's prediction made.

    ``mean_share`` averages each combat's share, the predicted move over the
    right one (1 the full move, 0 the keyword ignored, negative the wrong
    way), over ``combats``; ``standard_error`` is clustered by game and
    ``quartiles`` are the shares'. A combat whose right move is 0 has no
    share: ``zero_size`` counts those, and ``zero_size_move`` is the mean
    ``|m|`` the prediction moved on them anyway.
    """

    label: str
    combats: int
    mean_share: float
    standard_error: float
    quartiles: tuple[float, float, float]
    zero_size: int = 0
    zero_size_move: float = float("nan")

    def render(self) -> str:
        """``label 0.35 ± 0.03 (q 0.18/0.39/0.66) over 812, t=0: 14 (mean |m| 0.60)``."""
        if self.combats:
            q1, q2, q3 = self.quartiles
            text = (f"{self.label} {self.mean_share:.2f} ± {self.standard_error:.2f} "
                    f"(q {q1:.2f}/{q2:.2f}/{q3:.2f}) over {self.combats}")
        else:
            text = f"{self.label} no share"
        if not self.zero_size:
            return text
        return f"{text}, t=0: {self.zero_size} (mean |m| {self.zero_size_move:.2f})"


@dataclass(frozen=True, slots=True)
class KeywordMoves:
    """Gate 2's move report for one keyword: its effects' shares. It routes
    nothing; the verdict reads the direction alone (decision D11)."""

    effects: tuple[EffectMoves, ...] = ()

    @property
    def share(self) -> float | None:
        """The mean of the effects' mean shares, each effect counting once;
        None when no effect has a combat with a right move other than 0."""
        shares = [effect.mean_share for effect in self.effects if effect.combats]
        return sum(shares) / len(shares) if shares else None

    def render(self) -> str:
        """The report's second line for the keyword, holding no ``n=``: the
        report scripts read gate 2's lines by it."""
        share = self.share
        head = "share none" if share is None else f"share {share:.2f}"
        return "; ".join([f"moves: {head}", *(effect.render() for effect in self.effects)])


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
    #: Perturbations left out because the strip could not remove the keyword's
    #: statement (it is shared with other keywords).
    unqualified: int = 0
    #: How much of the right move each effect made; reported, routes nothing.
    moves: KeywordMoves | None = None

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
    ``verb`` reads before the value there (``explains 41.0% > 30%``);
    ``unmeasured`` says why a NaN value could not be measured.
    """

    key: str
    label: str
    value: float
    limit: float
    bound: Bound
    value_format: str = ""
    limit_format: str = ""
    verb: str = ""
    unmeasured: str = ""

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
            why = f" ({self.unmeasured})" if self.unmeasured else ""
            return f"{self.label} not measured{why}"
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


#: The checks that block shipping (FR-118, FR-123); gate 2 only routes.
BLOCKING_GATES = (CheckName.GATE_1, CheckName.GATE_3)


@dataclass
class EvaluationReport:
    """Everything one evaluation run produced.

    ``echo``, when given, receives each check's lines the moment the check is
    added, so a run that dies late has already printed what it measured.
    """

    checks: list[CheckResult] = field(default_factory=list)
    keyword_verdicts: list[KeywordVerdict] = field(default_factory=list)
    echo: Callable[[str], None] | None = None

    def add(self, result: CheckResult) -> CheckResult:
        self.checks.append(result)
        if self.echo is not None:
            self.echo(self.render_check(result))
        return result

    def extend(self, results: Iterable[CheckResult]) -> None:
        for result in results:
            self.add(result)

    @property
    def blocking_failures(self) -> list[CheckResult]:
        return [check for check in self.checks if check.blocks]

    @property
    def unmeasured_gates(self) -> list[str]:
        """The blocking gates no check of this run passed or failed: skipped,
        or never run."""
        measured = {
            check.name for check in self.checks
            if check.status in (CheckStatus.PASS, CheckStatus.FAIL)
        }
        return [gate for gate in BLOCKING_GATES if gate not in measured]

    @property
    def ships(self) -> bool:
        """Whether gates 1 and 3 were both measured and both hold. Reported; it
        never stops the command."""
        return not self.blocking_failures and not self.unmeasured_gates

    def verdict_line(self) -> str:
        failed = [check.name for check in self.blocking_failures]
        unmeasured = self.unmeasured_gates
        if failed:
            line = "GATES FAILED: " + ", ".join(failed)
            return f"{line} (not measured: {', '.join(unmeasured)})" if unmeasured else line
        if unmeasured:
            return "GATES NOT MEASURED: " + ", ".join(unmeasured)
        return "SHIPS"

    def render_check(self, check: CheckResult) -> str:
        """One check's lines; gate 2's are followed by its keyword verdicts."""
        lines = [f"[{check.status.marker}] {check.name}: {check.detail}"]
        # Every figure, whatever the verdict: a passing margin is a result too.
        if check.values:
            lines.append("       " + "  ".join(
                f"{name}={value:+.4f}" for name, value in check.values.items()
            ))
        if check.name == CheckName.GATE_2:
            lines.extend(chain.from_iterable(
                _keyword_lines(verdict) for verdict in self.keyword_verdicts))
        return "\n".join(lines)

    def render(self) -> str:
        return "\n".join(
            [*(self.render_check(check) for check in self.checks), self.verdict_line()]
        )


def _keyword_lines(verdict: KeywordVerdict) -> list[str]:
    """A keyword's verdict line, then its moves line when it has one."""
    state = "pass" if verdict.passed else "→ probe"
    lines = [
        f"    {verdict.keyword:<16} {state:<8} "
        f"n={verdict.qualifying_records:<6} {verdict.reason}"
    ]
    if verdict.moves is not None:
        lines.append(f"      {verdict.moves.render()}")
    return lines
