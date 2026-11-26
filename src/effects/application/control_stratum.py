"""One stratum's half of the control score, prepared once at startup
(:func:`prepare_stratum`): every item's records read, their reads cached on
the host (:class:`StratumCache`), each rule-test output's direction settled
from the frozen records' targets (``control_score``)."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from effects.application.control_chunks import Chunk, ChunkBuilder, Request
from effects.application.control_items import SELECTION, ControlScoreError
from effects.application.control_reads import (
    DOES_NOT_FIRE,
    FIRES,
    Observation,
    OutputReading,
    Read,
    readout_of,
)
from effects.application.host_cache import pack
from effects.application.step_preparation import DeviceStep
from effects.application.stratum_records import StratumReader, selection_records
from effects.application.surface_batching import SurfaceBatcher
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.grid_edits import CellRead
from effects.domain.head_decoding import decoder_of
from effects.domain.rule_output_targets import record_indicators
from effects.domain.selection_set import (
    COMPONENTS,
    DEVIANCE,
    GRIDS,
    PROBES,
    RULE_TESTS,
    FrozenGrid,
    SelectionSet,
    TaggedRecord,
    error_of,
    item_key,
)

if TYPE_CHECKING:
    from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)

#: Records per prepared chunk of the probes', grids' and rule tests' forward,
#: and per deviance step: forward passes only, so a chunk this large fits the
#: card beside a training run's state, and runs several times faster than
#: chunks of a training batch's size (T160).
CHUNK_RECORDS = 128


class GridEditor(Protocol):
    """Edits a frozen grid's record into each of the grid's cells.

    ``cells`` returns one :class:`CellRead` per cell, in the frozen grid's
    cell order; ``entities`` are the ones the selection set froze for the
    record.
    """

    def cells(self, grid: str, record: EffectRecord,
              entities: tuple[str, ...]) -> Sequence[CellRead]: ...


class ScoringPreparer(Protocol):
    """What the scorer asks of a preparer: a training run's ``StepPreparer``,
    or a saved checkpoint's :class:`CheckpointPreparer`."""

    def scoring_batcher(self) -> SurfaceBatcher: ...

    def prepare_scoring_step(self, records: Sequence[EffectRecord],
                             fields: tuple[FieldSpec, ...], *,
                             class_weighted: bool = False) -> DeviceStep: ...


@dataclass
class _ProbeItem:
    kind: str
    labels: list[float] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)


@dataclass
class GridItem:
    #: Per cell, the value positions of the grid's records.
    cells: list[list[int]]
    #: Per cell, the rule's answer for each record at the same index of ``cells``.
    answers: list[list[float]]

    def positions(self) -> list[int]:
        return [position for cell in self.cells for position in cell]

    def add(self, cell: int, position: int, answer: float) -> None:
        self.cells[cell].append(position)
        self.answers[cell].append(answer)


@dataclass
class StratumCache:
    """One stratum's half, prepared."""

    stratum: str
    half: str = SELECTION
    #: The components prepared and scored, in :data:`COMPONENTS` order.
    components: tuple[str, ...] = COMPONENTS
    values: int = 0
    chunks: list[Chunk] = field(default_factory=list)
    probes: dict[str, _ProbeItem] = field(default_factory=dict)
    grids: dict[str, GridItem] = field(default_factory=dict)
    rules: dict[str, dict[str, OutputReading]] = field(default_factory=dict)
    steps: list = field(default_factory=list)
    kept_items: frozenset[str] = frozenset()
    left_out: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    floors: dict[tuple, dict[str, float]] = field(default_factory=dict)
    #: The deviance terms the last epoch found with a loss that is not finite.
    unreadable_terms: int = 0

    def position(self) -> int:
        self.values += 1
        return self.values - 1


@dataclass(frozen=True)
class HalfScope:
    """What a control score prepares and scores: ``half`` of each stratum
    (the selection half for a training run, which never reads the reporting
    half), the ``components`` chosen, and whether a component with nothing
    to score raises (``strict``) or is NaN and named."""

    half: str = SELECTION
    components: Sequence[str] = COMPONENTS
    strict: bool = True


@dataclass(frozen=True)
class ScoringSource:
    """What a control score's records are read through: the ``preparer`` of
    its chunks and steps, every field set the run's epochs score
    (``fields``: the curriculum's before and after its switch; the deviance
    steps are prepared over their union), each rule's effective outputs, the
    grid ``editor`` (None where no half holds a grid), and the records per
    chunk and per deviance step (``batch_size``: a chunk's when None)."""

    preparer: ScoringPreparer
    fields: Sequence[tuple[FieldSpec, ...]]
    rule_outputs: Mapping[str, Sequence[str]]
    editor: GridEditor | None = None
    chunk_records: int = CHUNK_RECORDS
    batch_size: int | None = None


@dataclass(frozen=True)
class StratumPlan:
    """How each stratum's half is prepared: through ``source``, the deviance
    steps over ``union`` (its field sets' union) under ``mask``, within
    ``scope`` (its components in :data:`COMPONENTS` order)."""

    source: ScoringSource
    union: tuple[FieldSpec, ...]
    mask: str
    scope: HalfScope

    @property
    def step_records(self) -> int:
        """The deviance-sample copies per scoring step."""
        return self.source.batch_size or self.source.chunk_records


def prepare_stratum(selection: SelectionSet, stratum: str, read: StratumReader,
                    plan: StratumPlan) -> StratumCache:
    """One stratum's half prepared in three steps: what each record is read
    for, collected (:func:`_collect_asks`); the half read once
    (:class:`_HalfReading`); each rule's outputs settled and every component
    checked (:func:`_settle_and_check`)."""
    scope = plan.scope
    cache = StratumCache(stratum, half=scope.half, components=tuple(scope.components))
    frozen = _FrozenHalf.of(selection, stratum, scope)
    if frozen.grids and plan.source.editor is None:
        raise ControlScoreError(stratum, scope.half, GRIDS,
                                "the selection set holds grids and no grid editor was given")
    asks = _collect_asks(cache, selection, frozen, plan.source.rule_outputs)
    reading = _HalfReading(cache, asks, plan)
    reading.read(selection_records(read, games=frozen.games, records=asks.records(),
                                   copies=frozen.copies()))
    _settle_and_check(cache, selection, plan)
    return cache


# ── collect the requests ──


@dataclass(frozen=True)
class _FrozenHalf:
    """What the selection set froze in one stratum's half, each component
    out of scope empty, and the half's games."""

    probes: Mapping[str, Sequence]
    grids: Mapping[str, FrozenGrid]
    rule_tests: Mapping[str, Mapping[str, Sequence[TaggedRecord]]]
    samples: Sequence
    games: frozenset[str]

    @classmethod
    def of(cls, selection: SelectionSet, stratum: str, scope: HalfScope) -> _FrozenHalf:
        half, components = scope.half, scope.components

        def frozen(by_stratum, component, empty):
            if component not in components:
                return empty
            return by_stratum.get(stratum, {}).get(half, empty)

        return cls(frozen(selection.probes, PROBES, {}), frozen(selection.grids, GRIDS, {}),
                   frozen(selection.rule_tests, RULE_TESTS, {}),
                   frozen(selection.deviance_samples, DEVIANCE, ()),
                   frozenset(selection.halves.get(stratum, {}).get(half, ())))

    def copies(self) -> frozenset[tuple[str, int]]:
        """The deviance sample's copies, ``(record_id, copy)``."""
        return frozenset((row.record_id, row.copy) for row in self.samples)


@dataclass(frozen=True)
class _ProbeAsk:
    """A probe item's read of a record at ``entity``, into ``position``."""

    entity: str | None
    read: Read
    position: int

    def requests(self, record: EffectRecord, cache: StratumCache) -> list[Request]:
        return [Request(self.entity, self.read, self.position)]


@dataclass(frozen=True, eq=False)
class _RuleAsk:
    """A rule-test output's read of a record of ``polarity`` (``row``: the
    tag's outcome entities), each read's positions held by ``held`` (the
    reading's ``fires`` or ``does_not``)."""

    reading: OutputReading
    polarity: str
    row: TaggedRecord
    held: list[list[int]]

    def requests(self, record: EffectRecord, cache: StratumCache) -> list[Request]:
        """At the tag's outcome entities, with the target observation the
        rule's category is settled from; at the record for a record-level
        output, not read where the record supervises no target for it (a
        per-entity one is dropped the same way once its targets are built,
        ``ChunkBuilder``)."""
        if not self.reading.decoder.per_entity:
            return self._at_record(record, cache)
        return self._at_entities(cache)

    def _at_record(self, record: EffectRecord, cache: StratumCache) -> list[Request]:
        indicators = record_indicators(self.reading.decoder.output, record)
        if indicators is None:
            return []
        position = cache.position()
        self.held.append([position])
        request = Request(None, self.reading, position)
        self.reading.observe(self.polarity, indicators)
        return [request]

    def _at_entities(self, cache: StratumCache) -> list[Request]:
        observation = Observation(self.reading, self.polarity)
        out = [Request(entity, self.reading, cache.position(), observation)
               for entity in self.row.entities]
        if out:
            self.held.append([request.position for request in out])
        return out


#: ``record_id -> [(grid, entities, the rule's answer in each cell)]``.
_GridRequests = dict[str, list[tuple[str, tuple[str, ...], tuple[float, ...]]]]

#: ``record_id -> its probe and rule-test asks``, in the order asked.
_ReadAsks = dict[str, list["_ProbeAsk | _RuleAsk"]]


@dataclass(frozen=True)
class _Asks:
    """What each record of the half is read for: its probe and rule-test
    asks (``reads``), and its grids' (``grids``), each by ``record_id``."""

    reads: _ReadAsks
    grids: _GridRequests

    def records(self) -> frozenset[str]:
        return frozenset(self.reads) | frozenset(self.grids)


def _collect_asks(cache: StratumCache, selection: SelectionSet, frozen: _FrozenHalf,
                  rule_outputs: Mapping[str, Sequence[str]]) -> _Asks:
    """What each record is read for, each item opened in ``cache``: the
    probes', then the rule tests', then the grids'."""
    asks: _ReadAsks = defaultdict(list)
    _ask_probes(cache, selection.probe_kinds, frozen.probes, asks)
    for rule, sides in frozen.rule_tests.items():
        cache.rules[rule] = _ask_rule(rule_outputs.get(rule, ()), sides, asks)
    return _Asks(asks, _grid_requests(cache, frozen.grids))


def _ask_probes(cache: StratumCache, kinds: Mapping[str, str], probes: Mapping[str, Sequence],
                asks: _ReadAsks) -> None:
    for probe_type, rows in probes.items():
        item = cache.probes[probe_type] = _ProbeItem(kinds[probe_type])
        readout = readout_of(probe_type)
        for row in rows:
            position = cache.position()
            item.labels.append(row.label)
            item.positions.append(position)
            asks[row.record_id].append(_ProbeAsk(row.entity, readout, position))


def _ask_rule(outputs: Sequence[str], sides: Mapping[str, Sequence[TaggedRecord]],
              asks: _ReadAsks) -> dict[str, OutputReading]:
    """A rule's readings, one per output a head carries, each asked of the
    rule's records of both polarities."""
    readings = {}
    for output in outputs:
        decoder = decoder_of(output)
        if decoder is None:
            continue
        reading = readings[output] = OutputReading(decoder)
        _ask_polarities(reading, sides, asks)
    return readings


def _ask_polarities(reading: OutputReading, sides: Mapping[str, Sequence[TaggedRecord]],
                    asks: _ReadAsks) -> None:
    """One output's reading asked of the rule's ``fires`` records, then of
    its ``does_not_fire`` ones."""
    for polarity, held in ((FIRES, reading.fires), (DOES_NOT_FIRE, reading.does_not)):
        for row in sides.get(polarity, ()):
            asks[row.record_id].append(_RuleAsk(reading, polarity, row, held))


def _grid_requests(cache: StratumCache, grids: Mapping[str, FrozenGrid]) -> _GridRequests:
    """What each record of the half's frozen grids is read for, each grid's
    item opened in ``cache``."""
    out: _GridRequests = defaultdict(list)
    for name, grid in grids.items():
        cache.grids[name] = GridItem([[] for _ in grid.cells], [[] for _ in grid.cells])
        answers = grid.record_answers()
        for row in grid.records:
            out[row.record_id].append((name, row.entities, answers[row.record_id]))
    return out


# ── read the half ──


class _HalfReading:
    """The half's records read once, in the reader's order: each distinct
    record's asks into the plain chunks and its grid cells into chunks of
    their own, each deviance-sample copy into a scoring step."""

    def __init__(self, cache: StratumCache, asks: _Asks, plan: StratumPlan) -> None:
        self.cache, self.asks, self.plan = cache, asks, plan
        preparer, size = plan.source.preparer, plan.source.chunk_records
        self.plain = ChunkBuilder(preparer.scoring_batcher(), size)
        self.edited = ChunkBuilder(preparer.scoring_batcher(), size)
        self.sampled: list = []
        self.steps: list = []

    def read(self, records: Iterable[tuple[EffectRecord, bool, bool]]) -> None:
        """Every ``(record, distinct, copy)`` read, then the chunks and steps
        closed into the cache."""
        for record, distinct, copy in records:
            if distinct:
                self._distinct(record)
            if copy:
                self._copy(record)
        self._close()

    def _distinct(self, record: EffectRecord) -> None:
        asked = self.asks.reads.get(record.record_id, ())
        self.plain.add(record, [request for ask in asked
                                for request in ask.requests(record, self.cache)])
        self._edit_grid_cells(record)
        # One grid record's cells to a chunk: near copies, kept as their
        # differences (``host_cache``).
        self.edited.flush()

    def _edit_grid_cells(self, record: EffectRecord) -> None:
        """The record edited into each cell of every grid it is asked for,
        each cell read at its entity, its position and the record's answer
        there held by the grid's item. With no editor the half holds no grid
        (:func:`prepare_stratum` refuses one), so nothing is asked."""
        editor = self.plan.source.editor
        if editor is None:
            return
        for name, entities, answers in self.asks.grids.get(record.record_id, ()):
            cells = editor.cells(name, record, entities)
            for index, (cell, answer) in enumerate(zip(cells, answers, strict=True)):
                position = self.cache.position()
                self.cache.grids[name].add(index, position, answer)
                self.edited.add(cell.record,
                                [Request(cell.entity, readout_of(cell.readout), position)],
                                cell.texts)

    def _copy(self, record: EffectRecord) -> None:
        self.sampled.append(record)
        if len(self.sampled) >= self.plan.step_records:
            self._step()

    def _step(self) -> None:
        self.steps.append(_deviance_step(self.plan.source.preparer, self.sampled,
                                         self.plan.union))
        self.sampled = []

    def _close(self) -> None:
        self.plain.flush()
        self.edited.flush()
        if self.sampled:
            self._step()
        self._log_unread()
        self.cache.chunks = self.plain.chunks + self.edited.chunks
        self.cache.steps = self.steps

    def _log_unread(self) -> None:
        cache, plain, edited = self.cache, self.plain, self.edited
        unplaced = plain.unplaced + edited.unplaced
        if unplaced:
            logger.info("Control score, %s %s: %d read(s) at an entity the record gives no "
                        "slot, read nowhere (e.g. %s).", cache.stratum, cache.half, unplaced,
                        ", ".join((plain.unplaced_refs + edited.unplaced_refs)[:5]))
        if plain.untargeted:
            logger.info("Control score, %s %s: %d rule-test read(s) at an entity where the "
                        "record defines no target for the output, read nowhere.",
                        cache.stratum, cache.half, plain.untargeted)


# ── settle and check ──


def _settle_and_check(cache: StratumCache, selection: SelectionSet, plan: StratumPlan) -> None:
    """The deviance items the freeze kept, each rule's outputs settled, and
    every component checked for an item to read (raising on an empty one
    when strict, else logging it)."""
    cache.kept_items = _kept_items(selection, cache.stratum, plan.mask, half=cache.half)
    _settle_rules(cache)
    empty = _empty_components(cache)
    if empty and plan.scope.strict:
        detail = ("no deviance sample or no item the freeze kept" if empty[0] == DEVIANCE
                  else "no item to read")
        raise ControlScoreError(cache.stratum, cache.half, empty[0], detail)
    for component in empty:
        logger.info("Control score, %s %s: no %s item to read.", cache.stratum, cache.half,
                    component)
    left_out = cache.left_out[RULE_TESTS]
    if left_out:
        logger.info("Control score, %s: %d rule(s) or output(s) are left out (no output a head "
                    "reads, or a target the same in both polarities): %s", cache.stratum,
                    len(left_out), ", ".join(left_out[:12]))


def _empty_components(cache: StratumCache) -> list[str]:
    """The components in the cache's scope that hold nothing to read."""
    held = ((PROBES, cache.probes), (GRIDS, cache.grids), (RULE_TESTS, cache.rules),
            (DEVIANCE, cache.steps and cache.kept_items))
    return [component for component, items in held
            if component in cache.components and not items]


def _settle_rules(cache: StratumCache) -> None:
    """Keep each rule's outputs whose targets settle a direction; name in
    ``left_out`` a rule with none (``rule``) and an output of a rule that
    keeps others (``rule/output``)."""
    for rule in list(cache.rules):
        outputs = cache.rules[rule]
        settled = {output: reading for output, reading in outputs.items() if reading.settle()}
        if settled:
            cache.rules[rule] = settled
            cache.left_out[RULE_TESTS] += [f"{rule}/{output}" for output in outputs
                                           if output not in settled]
        else:
            del cache.rules[rule]
            cache.left_out[RULE_TESTS].append(rule)


def _deviance_step(preparer: ScoringPreparer, sampled: list,
                   union: tuple[FieldSpec, ...]) -> DeviceStep:
    """One packed scoring step of deviance-sample copies, each weighed by its
    ``class_weight`` (FR-059a)."""
    return pack(preparer.prepare_scoring_step(sampled, union, class_weighted=True))


def _kept_items(selection: SelectionSet, stratum: str, mask: str, *,
                 half: str = SELECTION) -> frozenset[str]:
    """The deviance items (every shipped head's output terms) the freeze
    kept in ``half`` under ``mask``: those whose standard error is at most
    the freeze's item target."""
    limit = float(selection.settings.get("max_item_error", 0.1))
    prefix = item_key(stratum, half, DEVIANCE, "")
    kept = set()
    for key, error in selection.standard_errors.items():
        if not key.startswith(prefix) or not isinstance(error, dict):
            continue
        if error_of(error.get(mask)) <= limit:
            kept.add(key[len(prefix):])
    return frozenset(kept)
