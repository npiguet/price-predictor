"""A rule-test output's targets, and the direction they settle (finding 12.9).

A rule test reads each of its rule's effective outputs at the tag's outcome
entities (a per-entity field or the affected gate) or at the record (a
verdict, global or created-objects output). The record's own targets say
which of the output's categories (or members) the rule moves: the one whose
mean indicator differs most between the ``fires`` and the ``does_not_fire``
rows, read as its complement where the difference is negative. An output no
``fires`` row observes, or one whose mean targets differ by less than
:data:`MIN_DIFFERENCE`, decides nothing about the rule: its AUC would enter
the rule's mean at full weight.

The control score settles each output's reading from these indicators, and
the freeze leaves out every output they do not settle, over the same frozen
rows, so both read one definition of an output's target. Each indicator is
what training reads (``effect_targets.derive_targets``, the verdict, global
and created-objects rows of ``aux_head_targets``), at an entity the record's
surface gives a slot (``effect_head_input.has_card_slot``, every player).

Torch-free: the freeze reads it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from effects.domain.aux_head_targets import (
    created_objects_targets,
    global_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.effect_head_input import slotted_entities
from effects.domain.effect_head_layout import (
    CREATED_OBJECT_SLOTS,
    CREATED_SLOT_WIDTH,
    FIELDS_BY_NAME,
    GLOBAL_LAYOUT,
    VERDICT_LAYOUT,
    CreatedSlot,
    FieldSpec,
    RecordHeadLayout,
)
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.field_types import FieldType
from effects.domain.records import EffectRecord
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, GATE_OUTPUT
from effects.domain.selection_set import (
    NO_DIFFERING_OUTPUT,
    SAME_TARGETS,
    UNOBSERVED_TARGETS,
    TaggedRecord,
)

#: The least a rule-test output's mean target must differ between the
#: polarities (in its most different category) for the output to be read.
MIN_DIFFERENCE = 1e-6

Indicators = list[float]


def reads_output(output: str) -> bool:
    """Whether a shipped head carries ``output``, so a rule test can read it."""
    return (output in (GATE_OUTPUT, CREATED_OBJECTS_OUTPUT) or output in FIELDS_BY_NAME
            or output in VERDICT_LAYOUT.by_name or output in GLOBAL_LAYOUT.by_name)


def entity_indicators(output: str, targets: RecordTargets, subject: str) -> Indicators | None:
    """A per-entity output's target indicators at ``subject``, one per
    category or member, or one value; None where the record does not say
    what the output is there."""
    if output == GATE_OUTPUT:
        return _gate_indicators(targets, subject)
    return _field_indicators(FIELDS_BY_NAME[output], targets, subject)


def _gate_indicators(targets: RecordTargets, subject: str) -> Indicators | None:
    if not targets.gated:
        return None
    return [1.0 if _affected(targets, subject) else 0.0]


def _affected(targets: RecordTargets, subject: str) -> bool:
    entry = targets.get(subject)
    return entry is not None and entry.affected


def _field_indicators(spec: FieldSpec, targets: RecordTargets,
                      subject: str) -> Indicators | None:
    """A gated field at an entity nothing affected is nothing happening
    where the record trains the gate, unread where it trains none."""
    if spec.gated and not _affected(targets, subject):
        return [0.0] * _width(spec) if targets.gated else None
    entry = targets.get(subject)
    value = None if entry is None else entry.supervised(spec)
    if value is None:
        return None
    if spec.type is FieldType.CATEGORICAL:
        return _one_hot(spec.arity, value)
    if spec.type is FieldType.MULTI_BINARY:
        members = [float(v) for v in value]
        return members + [0.0] * (spec.arity - len(members))
    return [float(value)]


def _width(spec: FieldSpec) -> int:
    return 1 if spec.type.has_magnitude or spec.type is FieldType.BINARY else spec.arity


def _one_hot(arity: int, value) -> Indicators:
    hot = [0.0] * arity
    hot[int(value)] = 1.0
    return hot


def record_indicators(output: str, record: EffectRecord) -> Indicators | None:
    """A verdict, global or created-objects output's target indicators at the
    record; None where the record supervises no target for it."""
    if output == CREATED_OBJECTS_OUTPUT:
        return _created_indicators(record)
    if output in VERDICT_LAYOUT.by_name:
        return _row_indicators(VERDICT_LAYOUT, verdict_targets, output, record)
    return _row_indicators(GLOBAL_LAYOUT, global_targets, output, record)


def _created_indicators(record: EffectRecord) -> Indicators | None:
    if not supervises_created_objects(record):
        return None
    target, _known = created_objects_targets(record)
    return [float(sum(target[slot * CREATED_SLOT_WIDTH + CreatedSlot.PRESENT]
                      for slot in range(CREATED_OBJECT_SLOTS)))]


def _row_indicators(layout: RecordHeadLayout, derive: Callable, output: str,
                    record: EffectRecord) -> Indicators | None:
    row = derive(record)
    if row is None or not row[1][layout.index[output]]:
        return None
    first, last = layout.target_slices[output]
    given = row[0][first:last]
    spec = layout.by_name[output]
    if spec.type is FieldType.CATEGORICAL:
        return _one_hot(spec.arity, given[0])
    if spec.type is FieldType.BINARY:
        return [float(v) for v in given]
    return [float(sum(given))]


class RecordIndicators:
    """One rule-test record's indicators for each output it is read for:
    the targets and the slotted subjects derived once, the first time a
    per-entity output needs them."""

    def __init__(self, record: EffectRecord) -> None:
        self.record = record
        self._targets: RecordTargets | None = None
        self._slotted: frozenset[str] | None = None

    def of(self, output: str, entities: Sequence[str]) -> Indicators | None:
        """``output``'s indicators on this record for a row read at
        ``entities``: a per-entity output's mean over the entities that have
        a slot and say what it is, else the record's; None where nothing is
        read, as for an output no shipped head carries."""
        if not reads_output(output):
            return None
        if output != GATE_OUTPUT and output not in FIELDS_BY_NAME:
            return record_indicators(output, self.record)
        seen = [found for found in (self._at(output, entity) for entity in entities)
                if found is not None]
        return [float(v) for v in np.mean(np.asarray(seen), axis=0)] if seen else None

    def _at(self, output: str, entity: str) -> Indicators | None:
        if entity not in self.slotted:
            return None
        return entity_indicators(output, self.targets, entity)

    @property
    def targets(self) -> RecordTargets:
        if self._targets is None:
            self._targets = derive_targets(self.record)
        return self._targets

    @property
    def slotted(self) -> frozenset[str]:
        """The subjects the record's surface gives a slot: every player and
        every entity but a library card below the top."""
        if self._slotted is None:
            self._slotted = frozenset(
                [player.id for player in self.record.state.players]
                + [entity.id for entity in slotted_entities(self.record)])
        return self._slotted


@dataclass(frozen=True)
class Direction:
    """What an output's targets settle: the category (or member) read, and
    +1, or −1 where it is read as its complement."""

    candidate: int
    sign: float


class TargetMeans:
    """One output's indicators summed per polarity over a rule's rows."""

    def __init__(self) -> None:
        self.sums: dict[bool, np.ndarray | None] = {True: None, False: None}
        self.counts = {True: 0, False: 0}

    def observe(self, fires: bool, indicators: Sequence[float]) -> None:
        vector = np.asarray(indicators, dtype=np.float64)
        held = self.sums[fires]
        self.sums[fires] = vector if held is None else held + vector
        self.counts[fires] += 1

    def observe_all(self, fires: bool, rows: Iterable[Sequence[float] | None]) -> None:
        """Observe each of ``rows`` that is read; None is a row read nowhere."""
        for indicators in rows:
            if indicators is not None:
                self.observe(fires, indicators)

    def means(self, fires: bool) -> np.ndarray | None:
        held = self.sums[fires]
        return None if held is None else held / self.counts[fires]

    def settle(self) -> Direction | None:
        """The category whose mean differs most between the polarities, and
        its direction; None when no ``fires`` row observed the output or no
        category differs by :data:`MIN_DIFFERENCE` (a polarity no row
        observed counts as all zeros)."""
        fires = self.means(True)
        if fires is None:
            return None
        does_not = self.means(False)
        difference = fires - (np.zeros_like(fires) if does_not is None else does_not)
        candidate = int(np.argmax(np.abs(difference)))
        if abs(difference[candidate]) < MIN_DIFFERENCE:
            return None
        return Direction(candidate, -1.0 if difference[candidate] < 0 else 1.0)

    @property
    def observed(self) -> bool:
        """Whether a ``fires`` row observed the output."""
        return bool(self.counts[True])


# ── the freeze's reading of a rule's drawn rows ─────────────────────────

#: ``row -> output -> its indicators on the row``.
RowTargets = Mapping[TaggedRecord, Mapping[str, Indicators | None]]
Drawn = Mapping[Polarity, list[TaggedRecord]]


def informative_rows(
    rule: str, drawn: Drawn,
    targets: Mapping[tuple[str, TaggedRecord], Mapping[str, Indicators | None]],
) -> tuple[dict[Polarity, list[TaggedRecord]] | None, list[tuple[str, str]]]:
    """The drawn rows a kept output of ``rule`` reads (None when none is
    kept), and each item left out with why: an output whose targets settle
    no direction over the drawn rows (``rule/output``), or the rule when
    none is left. ``targets`` is keyed ``(rule, row)``; a rule it holds no
    row of reads every drawn row."""
    held = {row: targets[(rule, row)] for rows in drawn.values() for row in rows
            if (rule, row) in targets}
    if not held:
        return dict(drawn), []
    means = _output_means(drawn, held)
    kept = frozenset(output for output, found in means.items() if found.settle() is not None)
    if not kept:
        return None, [(rule, NO_DIFFERING_OUTPUT)]
    return _rows_reading(drawn, held, kept), _left_outputs(rule, means, kept)


def _output_means(drawn: Drawn, held: RowTargets) -> dict[str, TargetMeans]:
    """Each output's indicators summed per polarity over the drawn rows."""
    means = {output: TargetMeans() for by_output in held.values() for output in by_output}
    for polarity, rows in drawn.items():
        for row in rows:
            _observe_row(means, polarity is Polarity.FIRES, held[row])
    return means


def _observe_row(means: Mapping[str, TargetMeans], fires: bool,
                 by_output: Mapping[str, Indicators | None]) -> None:
    for output, indicators in by_output.items():
        means[output].observe_all(fires, (indicators,))


def _rows_reading(drawn: Drawn, held: RowTargets,
                  kept: frozenset[str]) -> dict[Polarity, list[TaggedRecord]]:
    return {polarity: [row for row in rows if _reads(held[row], kept)]
            for polarity, rows in drawn.items()}


def _reads(by_output: Mapping[str, Indicators | None], kept: frozenset[str]) -> bool:
    return any(by_output.get(output) is not None for output in kept)


def _left_outputs(rule: str, means: Mapping[str, TargetMeans],
                  kept: frozenset[str]) -> list[tuple[str, str]]:
    return [(f"{rule}/{output}", SAME_TARGETS if found.observed else UNOBSERVED_TARGETS)
            for output, found in means.items() if output not in kept]
