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
    error_of,
    item_key,
)

if TYPE_CHECKING:
    from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)


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


def prepare_stratum(
    selection: SelectionSet, stratum: str, read: StratumReader, *,
    preparer: ScoringPreparer, union: tuple[FieldSpec, ...],
    rule_outputs: Mapping[str, Sequence[str]], mask: str, editor: GridEditor | None,
    chunk_records: int, batch_size: int, half: str = SELECTION,
    components: tuple[str, ...] = COMPONENTS, strict: bool = True,
) -> StratumCache:
    cache = StratumCache(stratum, half=half, components=components)

    def frozen(by_stratum, component, empty):
        return by_stratum.get(stratum, {}).get(half, empty) if component in components else empty

    probes = frozen(selection.probes, PROBES, {})
    grids = frozen(selection.grids, GRIDS, {})
    rule_tests = frozen(selection.rule_tests, RULE_TESTS, {})
    samples = frozen(selection.deviance_samples, DEVIANCE, ())
    if grids and editor is None:
        raise ControlScoreError(stratum, half, GRIDS,
                                "the selection set holds grids and no grid editor was given")

    # What each record is read for.
    requests: dict[str, list] = defaultdict(list)
    for probe_type, rows in probes.items():
        item = cache.probes[probe_type] = _ProbeItem(selection.probe_kinds[probe_type])
        readout = readout_of(probe_type)
        for row in rows:
            position = cache.position()
            item.labels.append(row.label)
            item.positions.append(position)
            requests[row.record_id].append(("probe", row.entity, readout, position))
    for rule, sides in rule_tests.items():
        readings = {}
        for output in rule_outputs.get(rule, ()):
            decoder = decoder_of(output)
            if decoder is None:
                continue
            reading = readings[output] = OutputReading(decoder)
            for polarity, held in ((FIRES, reading.fires), (DOES_NOT_FIRE, reading.does_not)):
                for row in sides.get(polarity, ()):
                    requests[row.record_id].append(("rule", reading, polarity, row, held))
        cache.rules[rule] = readings
    grid_records = _grid_requests(cache, grids)

    batcher = preparer.scoring_batcher()
    plain = ChunkBuilder(batcher, chunk_records)
    edited = ChunkBuilder(preparer.scoring_batcher(), chunk_records)
    sampled: list = []
    steps: list = []
    games = frozenset(selection.halves.get(stratum, {}).get(half, ()))
    wanted = frozenset(requests) | frozenset(grid_records)
    copies = frozenset((row.record_id, row.copy) for row in samples)
    for record, distinct, copy in selection_records(read, games=games, records=wanted,
                                                    copies=copies):
        if distinct:
            plain.add(record, _requests(record, requests.get(record.record_id, ()), cache))
            _edit_grid_cells(cache, edited, editor, record, grid_records.get(record.record_id, ()))
            # One grid record's cells to a chunk: near copies, kept as their
            # differences (``host_cache``).
            edited.flush()
        if copy:
            sampled.append(record)
            if len(sampled) >= batch_size:
                steps.append(_deviance_step(preparer, sampled, union))
                sampled = []
    plain.flush()
    edited.flush()
    if sampled:
        steps.append(_deviance_step(preparer, sampled, union))
    unplaced = plain.unplaced + edited.unplaced
    if unplaced:
        logger.info("Control score, %s %s: %d read(s) at an entity the record gives no slot, "
                    "read nowhere (e.g. %s).", stratum, half, unplaced,
                    ", ".join((plain.unplaced_refs + edited.unplaced_refs)[:5]))
    if plain.untargeted:
        logger.info("Control score, %s %s: %d rule-test read(s) at an entity where the record "
                    "defines no target for the output, read nowhere.", stratum, half,
                    plain.untargeted)
    cache.chunks = plain.chunks + edited.chunks
    cache.steps = steps
    cache.kept_items = _kept_items(selection, stratum, mask, half=half)
    _settle_rules(cache)
    empty = [component for component, held in (
        (PROBES, cache.probes), (GRIDS, cache.grids), (RULE_TESTS, cache.rules),
        (DEVIANCE, cache.steps and cache.kept_items)) if component in components and not held]
    if empty and strict:
        detail = ("no deviance sample or no item the freeze kept" if empty[0] == DEVIANCE
                  else "no item to read")
        raise ControlScoreError(stratum, half, empty[0], detail)
    for component in empty:
        logger.info("Control score, %s %s: no %s item to read.", stratum, half, component)
    if cache.left_out[RULE_TESTS]:
        logger.info("Control score, %s: %d rule(s) or output(s) are left out (no output a head "
                    "reads, or a target the same in both polarities): %s", stratum,
                    len(cache.left_out[RULE_TESTS]), ", ".join(cache.left_out[RULE_TESTS][:12]))
    return cache


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


#: ``record_id -> [(grid, entities, the rule's answer in each cell)]``.
_GridRequests = dict[str, list[tuple[str, tuple[str, ...], tuple[float, ...]]]]


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


def _edit_grid_cells(cache: StratumCache, edited: ChunkBuilder, editor: GridEditor | None,
                     record: EffectRecord,
                     asked: Iterable[tuple[str, tuple[str, ...], tuple[float, ...]]]) -> None:
    """One record edited into each cell of every grid ``asked`` names, each
    cell read at its entity, its position and the record's answer there
    held by the grid's item. With no ``editor`` the half holds no grid
    (:func:`prepare_stratum` refuses one), so nothing is asked."""
    if editor is None:
        return
    for name, entities, answers in asked:
        cells = editor.cells(name, record, entities)
        for index, (cell, answer) in enumerate(zip(cells, answers, strict=True)):
            position = cache.position()
            cache.grids[name].add(index, position, answer)
            edited.add(cell.record, [Request(cell.entity, readout_of(cell.readout), position)],
                       cell.texts)


def _requests(record: EffectRecord, asked: Iterable[tuple], cache: StratumCache) -> list:
    """One record's reads: each probe item at its entity, each rule-test
    output at the tag's outcome entities or at the record, with the target
    observations the rule's category is settled from; a record-level output
    the record supervises no target for is not read (a per-entity one is
    dropped the same way once its targets are built, ``ChunkBuilder``)."""
    out: list[Request] = []
    for ask in asked:
        if ask[0] == "probe":
            _, entity, read, position = ask
            out.append(Request(entity, read, position))
            continue
        _, reading, polarity, row, held = ask
        decoder = reading.decoder
        if not decoder.per_entity:
            indicators = record_indicators(decoder.output, record)
            if indicators is None:
                continue
            position = cache.position()
            held.append([position])
            out.append(Request(None, reading, position))
            reading.observe(polarity, indicators)
            continue
        positions = []
        observation = Observation(reading, polarity)
        for entity in row.entities:
            position = cache.position()
            positions.append(position)
            out.append(Request(entity, reading, position, observation))
        if positions:
            held.append(positions)
    return out


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
