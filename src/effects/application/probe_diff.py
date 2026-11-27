"""Probe diff: the model's move beside Forge's on the probed combats (FR-116).

A damage-step probe re-ran one real combat in Forge with one keyword stripped
from one carrier, so its record and the real one differ by the keyword as
Forge played it. Gate 2's stream holds both, the stratum taking whole games:
each probe fork is joined to its real combat by ``mirror_of`` and the probed
keyword and carrier, and only when gate 2 scored that combat for that keyword
and carrier.

Forge's move ``f`` is the real record's target minus the fork's
(:func:`effects.domain.effect_targets.derive_targets` on both), on each
effect's field at each of its subjects; the model's move ``m`` is gate 2's
own, the real record built as collected minus its stripped twin. Each pair
scores ``m / f`` as gate 2's moves line scores ``m / t``, an effect's subjects
averaged first, and a pair whose ``f`` is 0 counts apart with its mean
``|m|`` (:class:`effects.application.gate_two_moves.MoveTally`).

A fork may come before or after its real record in shard order, so both
sides wait in a buffer keyed by the join until the other arrives. First and
double strike are not measured: their fork deals both damage steps
(``DamageStepProbe.stepsDealt``) where the real record is the first-strike
step alone. The check is reported, routes nothing and the control score reads
none of it; one that raises is a skip line and leaves gate 2 standing.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

from effects.application.gate_two_moves import MoveTally, PredictedMove
from effects.domain.check_label import CheckLabel
from effects.domain.damage_step_keywords import (
    CombatParticipant,
    DamageStepKeyword,
    KeywordEffect,
)
from effects.domain.effect_head_layout import ZONE_OUTCOMES, HeadField
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.evaluation_report import CheckName, CheckResult, CheckStatus

logger = logging.getLogger(__name__)

#: Keywords whose fork deals both damage steps where the real record is one.
NOT_MEASURED = frozenset({"first_strike", "double_strike"})


class ProbeKey(NamedTuple):
    """What joins a fork to its real combat: the real record, the probed
    keyword and its carrier."""

    record_id: str
    keyword: str
    carrier: str


class _RealSide(NamedTuple):
    """One scored observation waiting for its fork: the real record's game,
    gate 2's predicted moves and the real record's value at each."""

    game: str
    moves: tuple[PredictedMove, ...]
    values: tuple[float | None, ...]


def observed_value(targets: RecordTargets, effect: KeywordEffect, subject: str) -> float | None:
    """The record's value of ``effect``'s field at ``subject``, as gate 2
    reads the prediction: 0 where the subject is unaffected, for a zone
    outcome 1 where it is the effect's class and 0 otherwise; None where the
    record declares the field unknown."""
    entry = targets.get(subject)
    if entry is None or not entry.affected:
        return 0.0
    if effect.field in entry.unknown:
        return None
    value = entry.fields.get(effect.field)
    if effect.field == HeadField.ZONE_OUTCOME:
        return 1.0 if value == ZONE_OUTCOMES.index(effect.category) else 0.0
    return float(value or 0)


def forge_moves(real: _RealSide, fork: RecordTargets) -> dict[str, list[tuple[float, float]]]:
    """Each effect label's ``(m, f)`` at every subject both records can say."""
    by_label: dict[str, list[tuple[float, float]]] = {}
    for move, value in zip(real.moves, real.values, strict=True):
        forked = observed_value(fork, move.effect, move.subject)
        if value is not None and forked is not None:
            by_label.setdefault(move.effect.label, []).append((move.predicted, value - forked))
    return by_label


@dataclass(slots=True)
class ProbeJoin:
    """The probe forks and gate 2's scored observations, joined while the
    pass reads, and the model's move beside Forge's on each pair."""

    forks: dict[ProbeKey, RecordTargets] = field(default_factory=dict)
    observed: dict[ProbeKey, _RealSide] = field(default_factory=dict)
    moves: dict[str, MoveTally] = field(default_factory=dict)
    paired: Counter[str] = field(default_factory=Counter)
    not_measured: Counter[str] = field(default_factory=Counter)
    #: Why the join stopped, when one of its steps raised.
    failure: str | None = None

    def fork(self, record) -> None:
        """Take one probe fork: pair it with its scored observation, or keep
        it until that arrives."""
        self._guarded(self._fork, record)

    def observe(self, record, row: DamageStepKeyword, participant: CombatParticipant,
                moves: Sequence[PredictedMove]) -> None:
        """Take one observation gate 2 scored: pair it with its fork, or keep
        it until that arrives."""
        self._guarded(self._observe, record, row, participant, moves)

    def _guarded(self, step, *args) -> None:
        if self.failure is not None:
            return
        try:
            step(*args)
        except Exception as error:
            logger.warning("the reported check %s raised", CheckName.PROBE_DIFF, exc_info=True)
            self.failure = f"{type(error).__name__}: {error}"

    def _fork(self, record) -> None:
        payload = record.payload
        keyword = payload.probed_keyword
        if keyword in NOT_MEASURED:
            self.not_measured[keyword] += 1
            return
        key = ProbeKey(record.mirror_of, keyword, payload.probed_entity)
        targets = derive_targets(record)
        real = self.observed.pop(key, None)
        if real is None:
            self.forks[key] = targets
        else:
            self._pair(keyword, real, targets)

    def _observe(self, record, row, participant, moves) -> None:
        if row.keyword in NOT_MEASURED:
            return
        key = ProbeKey(record.record_id, row.keyword, participant.carrier.id)
        targets = derive_targets(record)
        real = _RealSide(record.game_id, tuple(moves), tuple(
            observed_value(targets, move.effect, move.subject) for move in moves))
        fork = self.forks.pop(key, None)
        if fork is None:
            self.observed[key] = real
        else:
            self._pair(row.keyword, real, fork)

    def _pair(self, keyword: str, real: _RealSide, fork: RecordTargets) -> None:
        self.paired[keyword] += 1
        self.moves.setdefault(keyword, MoveTally()).add(real.game, forge_moves(real, fork))

    def checks(self, rows: Iterable[DamageStepKeyword]) -> list[CheckResult]:
        """The check's lines: a summary, then one per measured keyword."""
        if self.failure is not None:
            return [CheckResult(CheckName.PROBE_DIFF, CheckStatus.SKIPPED, self.failure)]
        unpaired = Counter(key.keyword for key in self.forks)
        lines = [CheckResult(CheckName.PROBE_DIFF, CheckStatus.REPORTED,
                             self._summary(sum(unpaired.values())))]
        for row in rows:
            if row.keyword not in NOT_MEASURED:
                lines.append(self._keyword_line(row, unpaired[row.keyword]))
        return lines

    def _summary(self, unpaired: int) -> str:
        skipped = ", ".join(f"{keyword} {count}" for keyword, count in
                            sorted(self.not_measured.items())) or "none"
        return (
            f"the model's move over Forge's (f: the real combat's targets minus its "
            f"probe fork's), {self.paired.total()} probed combats paired, {unpaired} "
            f"forks unpaired; routes nothing; first and double strike not measured "
            f"(forks: {skipped}): a fork deals both damage steps where the real "
            f"record is one"
        )

    def _keyword_line(self, row: DamageStepKeyword, unpaired: int) -> CheckResult:
        moves = self.moves.get(row.keyword, MoveTally()).summary(row)
        return CheckResult(
            str(CheckLabel.of(CheckName.PROBE_DIFF, row.keyword)), CheckStatus.REPORTED,
            f"{moves.figures('f')}; paired {self.paired[row.keyword]}, "
            f"forks unpaired {unpaired}",
        )
