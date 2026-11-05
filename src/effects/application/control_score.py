"""The control score: each epoch's selection number, from the frozen selection
set's selection halves (FR-068, FR-068c, FR-069; research E2, E6;
contracts/selection-set.md).

Built once, at startup (:meth:`ControlScorer.prepare`): the selection half of
each validation stratum is read once, and the host half of every item it
holds is prepared and cached —

- the probe records, each read at its entity's slot (``[ACT]`` for a verdict
  probe) through :data:`~effects.domain.probe_readouts.READOUTS`;
- the grid edits, each grid record edited into every cell by a
  :class:`GridEditor` and read at the cell's entity;
- the rule-test records, read for each output the manifest's map names for the
  rule (E6): at the tag's entities (their mean), or at the record for a
  verdict, global or created-objects output;
- the deviance sample's copies, as scoring steps under the run's
  ``--rule-mask``, the way validation scores its own records.

A record several items read is prepared once. The cache stays on the host,
packed (``host_cache``: each tensor in its smallest exact form), and is
unpacked a chunk at a time as the chunk is run. The cells of one grid record
fill chunks of their own, so a chunk's rows are near copies of each other and
are kept as their differences.

Each epoch (:meth:`ControlScorer.epoch`), on whatever weights the caller has
swapped in (the averaged ones), only the device forward runs, through
``record_forward``, and each stratum's four components are scored:

============  ===============================================  ===============
component     item score                                       rescaled
============  ===============================================  ===============
probes        rung-3 AUC (binary) or R² (amount), per type     2·AUC − 1; R²
grids         MAE of the cell means against the rule's cells   1 − MAE / B
rule tests    per rule, mean over its outputs of the AUC        2·AUC − 1
              ranking ``fires`` above ``does_not_fire``
deviance      per field, 1 − loss / constant loss              as is
============  ===============================================  ===============

Every item is clipped below at −1, a component is the mean of its items, a
stratum score the mean of its four components, and the control score
``W · card_disjoint + (1 − W) · game_disjoint`` with ``W = --stratum-weight``.
A component with nothing to score — no item kept, none readable, or a
deviance component every field of which falls at or below
``--deviance-floor`` — raises :class:`ControlScoreError` naming stratum,
half and component, which ``train-effect-model`` turns into exit 1 (FR-068c).

**A rule test's readout** (E6). For a binary or categorical output the
reading is the probability of the scenario's category, for a count or delta
its expected value under the negative binomial; per-entity readings are the
unconditional ones gate 2 and the probes read (``HeadReading``: the gate's
probability times the field's, a legality field ungated). The selection set
names a rule's records and their polarity, not the scenario's category, so
the category is read off the frozen records' own targets once, at startup:
the category (or, for a multi-member output, the member) whose observed share
differs most between the ``fires`` and the ``does_not_fire`` records, and the
direction of that difference, which turns the reading into its complement
where the scenario's outcome is an absence (a creature that does not die). A
count or delta takes the sign of the difference of the two polarities' mean
targets. An output no ``fires`` record observes, an output name no head
carries (``new:…``, ``planned:…``) and a per-entity output on a record whose
tag names no entity on the board are not read; a rule with no output left is
left out of its component.

**The deviance component** reads the fields the frozen set kept under the
run's mask (each output field's standard error at most the freeze's item
target), the per-entity head's alone: the freeze computes no item for the
verdict, global and created-objects heads, so neither is an item here.

The reporting half is never read: only the selection half's games and the
records its items name are taken from a stratum.

In a training run the validator (``validation_pass.Validator.prepare``)
builds one :class:`ControlScorer` at startup from the run's
:class:`FrozenSelection`, handing it every field set the run's epochs use
and ``grid_edits.GridEdits`` as its editor, and scores it at each epoch's
close inside the averaged weights' swap; the training loop keeps the epoch
of the best :attr:`ControlScore.control` (FR-069), and the CLI turns a
:class:`ControlScoreError` into exit 1.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from effects.application.host_cache import host_bytes, pack, unpack
from effects.application.validation_pass import FieldParts, KeptTargets, score_steps
from effects.domain.aux_head_targets import (
    created_objects_targets,
    global_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.effect_head_layout import (
    CREATED_OBJECT_SLOTS,
    CREATED_SLOT_WIDTH,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    GLOBAL_LAYOUT,
    VERDICT_LAYOUT,
    CreatedSlot,
    FieldSpec,
    RecordHeadLayout,
)
from effects.domain.field_types import FieldType
from effects.domain.grid_edits import CellRead, TextOverride
from effects.domain.head_decoding import HeadReading, direction_balance, expected_magnitude
from effects.domain.loss_floor import constant_predictor_floor
from effects.domain.probe_readouts import READOUTS, item_score
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, GATE_OUTPUT
from effects.domain.score_scales import (
    auc,
    clipped,
    grid_score,
    rescaled_auc,
)
from effects.domain.selection_set import (
    BINARY,
    COMPONENTS,
    DEVIANCE,
    GRIDS,
    PROBES,
    RULE_TESTS,
    SelectionSet,
    error_of,
    item_key,
)
from effects.domain.strata import Half, Stratum
from effects.domain.surface_collation import per_entity_targets

if TYPE_CHECKING:
    from effects.application.record_forward import ForwardOutputs, RecordForward
    from effects.application.step_loss import StepLossComputer
    from effects.application.step_preparation import StepPreparer
    from effects.domain.entity_field_targets import EntityTargetBatch
    from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)

#: The half selection reads; the reporting half is the selected checkpoint's.
SELECTION = str(Half.SELECTION)
#: The polarities a rule test ranks, ``fires`` first.
FIRES, DOES_NOT_FIRE = "fires", "does_not_fire"
#: Records per prepared chunk of the probes', grids' and rule tests' forward,
#: and per deviance step: forward passes only, so a chunk this large fits the
#: card beside a training run's state, and runs several times faster than
#: chunks of a training batch's size (T160).
CHUNK_RECORDS = 128


class ControlScoreError(RuntimeError):
    """A component with nothing to score in one stratum's half (FR-068c)."""

    def __init__(self, stratum: str, half: str, component: str, detail: str) -> None:
        self.stratum, self.half, self.component = stratum, half, component
        super().__init__(f"{stratum} {half}, {component}: {detail}")


# ── the grid port ───────────────────────────────────────────────────────


class GridEditor(Protocol):
    """Edits a frozen grid's record into each of the grid's cells.

    ``cells`` returns one :class:`CellRead` per cell, in the frozen grid's
    cell order; ``entities`` are the ones the selection set froze for the
    record.
    """

    def cells(self, grid: str, record: EffectRecord,
              entities: tuple[str, ...]) -> Sequence[CellRead]: ...


# ── scores ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StratumScore:
    """One stratum's four components, each the mean of its clipped items."""

    components: Mapping[str, float]
    #: ``component -> item -> clipped rescaled score``, what each mean read.
    items: Mapping[str, Mapping[str, float]]
    #: ``component -> items read but left out this epoch, with why``.
    left_out: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """The mean of the four components."""
        return float(np.mean([self.components[name] for name in COMPONENTS]))


@dataclass(frozen=True)
class ControlScore:
    """Both strata's scores and the control score they mix to (FR-069)."""

    strata: Mapping[str, StratumScore]
    stratum_weight: float

    @property
    def control(self) -> float:
        """``W · card_disjoint + (1 − W) · game_disjoint``."""
        return mixed(self.strata[str(Stratum.CARD_DISJOINT)].score,
                     self.strata[str(Stratum.GAME_DISJOINT)].score, self.stratum_weight)


def mixed(card_disjoint: float, game_disjoint: float, weight: float) -> float:
    """The control score: ``weight · card_disjoint + (1 − weight) · game_disjoint``."""
    return weight * card_disjoint + (1.0 - weight) * game_disjoint


def component_score(items: Mapping[str, float], *, stratum: str, component: str,
                    half: str = SELECTION) -> float:
    """The mean of a component's items, each clipped below at −1; raises
    :class:`ControlScoreError` when no item has a score."""
    scores = [clipped(value) for value in items.values() if not math.isnan(value)]
    if not scores:
        raise ControlScoreError(stratum, half, component, "nothing to score this epoch")
    return float(np.mean(scores))


def probe_item(kind: str, labels: Sequence[float], predictions: Sequence[float]) -> float:
    """A probe type on the common scale: ``2·AUC − 1`` for a binary one, R²
    for an amount; NaN where it cannot be scored."""
    score = item_score(kind, labels, predictions)
    return rescaled_auc(score) if kind == BINARY and not math.isnan(score) else score


def rule_item(outputs: Iterable[tuple[Sequence[float], Sequence[float]]]) -> float:
    """A rule test on the common scale: the mean over its outputs of
    ``2·AUC − 1``, ``fires`` ranked above ``does_not_fire``; each output is
    ``(fires readings, does_not_fire readings)``, and one without a finite
    reading of both polarities is left out. NaN with none left."""
    scores = []
    for fires, does_not in outputs:
        value = auc([True] * len(fires) + [False] * len(does_not), [*fires, *does_not])
        if not math.isnan(value):
            scores.append(rescaled_auc(value))
    return float(np.mean(scores)) if scores else math.nan


def explained_deviance(parts: Mapping[str, float], floor: Mapping[str, float], *,
                       kept: Iterable[str], deviance_floor: float) -> dict[str, float]:
    """``1 − loss / constant loss`` per field of ``kept`` the pass scored,
    leaving out each field whose constant-prediction loss is at or below
    ``deviance_floor`` (FR-068)."""
    out = {}
    for name in kept:
        loss, constant = parts.get(name), floor.get(name)
        if loss is None or constant is None or constant <= deviance_floor:
            continue
        out[name] = 1.0 - loss / constant
    return out


# ── reading the heads ───────────────────────────────────────────────────


class _Heads:
    """One chunk's outputs, each head read once: the per-entity head at the
    chunk's asked slots, the record heads on first use."""

    def __init__(self, outputs: ForwardOutputs, index: torch.Tensor) -> None:
        self.outputs = outputs
        self.gathered = outputs.per_entity_at(index)
        self._verdict = self._global = self._created = None

    @property
    def verdict(self) -> torch.Tensor:
        if self._verdict is None:
            self._verdict = self.outputs.verdict()
        return self._verdict

    @property
    def global_(self) -> torch.Tensor:
        if self._global is None:
            self._global = self.outputs.global_outputs()
        return self._global

    @property
    def created(self) -> torch.Tensor:
        if self._created is None:
            self._created = self.outputs.created_objects()
        return self._created


#: ``read(heads, rows, columns) -> (n,) tensor``: one value per asked chunk
#: row and gathered column, every ask of one read in a chunk at once.
Read = Callable[[_Heads, torch.Tensor, torch.Tensor], torch.Tensor]


class _ReadoutRead:
    """A :data:`READOUTS` answer at the asked rows and gathered columns (a
    verdict readout ignores the columns)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.readout = READOUTS[name]

    def __call__(self, heads: _Heads, rows: torch.Tensor, columns: torch.Tensor):
        return self.readout(heads.gathered, heads.verdict, rows, columns)


#: One read per readout, so a chunk's asks of one readout are read together.
_READOUT_READS: dict[str, _ReadoutRead] = {}


def _readout(name: str) -> Read:
    read = _READOUT_READS.get(name)
    if read is None:
        read = _READOUT_READS[name] = _ReadoutRead(name)
    return read


class _Decoder:
    """One rule-test output's candidate readings and the matching target
    indicators: a candidate per category (or member), or one expected value."""

    #: Whether the output is read at the tag's entities rather than the record.
    per_entity = False
    #: Whether the candidates are probabilities of categories (else amounts).
    categorical = True

    def candidates(self, heads: _Heads, rows: torch.Tensor,
                   columns: torch.Tensor) -> torch.Tensor:
        """``(asks, candidates)``: each ask's reading of every candidate."""
        raise NotImplementedError

    def entity_indicators(self, targets: EntityTargetBatch, row: int,
                          column: int) -> list[float] | None:
        return None

    def record_indicators(self, record: EffectRecord) -> list[float] | None:
        return None


class _GateDecoder(_Decoder):
    per_entity = True

    def candidates(self, heads, rows, columns):
        return torch.sigmoid(heads.gathered[rows, columns, GATE_INDEX].float()).unsqueeze(-1)

    def entity_indicators(self, targets, row, column):
        if not bool(targets.gate_scored[row, column]):
            return None
        return [float(targets.gate[row, column])]


class _EntityFieldDecoder(_Decoder):
    """A per-entity field, read unconditionally (gated by the affected gate,
    a legality field ungated)."""

    per_entity = True

    def __init__(self, spec: FieldSpec) -> None:
        self.spec = spec
        self.categorical = not spec.type.has_magnitude

    def candidates(self, heads, rows, columns):
        outputs = heads.gathered[rows, columns]
        spec = self.spec
        if spec.type.has_magnitude:
            value = HeadReading(outputs).expected(spec.name, conditional=not spec.gated)
            return value.unsqueeze(-1)
        start, end = FIELD_SLICES[spec.name]
        read = outputs[:, start:end].float()
        if spec.type is FieldType.CATEGORICAL:
            value = torch.softmax(read, dim=-1)
        else:
            value = torch.sigmoid(read)
        if spec.gated:
            value = HeadReading(outputs).p_affected.unsqueeze(-1) * value
        return value

    def entity_indicators(self, targets, row, column):
        spec = self.spec
        present = (spec.name in targets.fields
                   and bool(targets.fields.present[spec.name][row, column]))
        scored = present and bool(targets.scored(spec.name)[row, column])
        if spec.gated:
            if not bool(targets.gate_scored[row, column]):
                return None
            if not bool(targets.affected[row, column]):
                return [0.0] * _width(spec)
            if not scored:
                return None
        elif not scored:
            return None
        value = targets.fields.values[spec.name][row, column]
        if spec.type is FieldType.CATEGORICAL:
            hot = [0.0] * spec.arity
            hot[int(value)] = 1.0
            return hot
        if spec.type is FieldType.MULTI_BINARY:
            return [float(v) for v in value.reshape(-1)]
        return [float(value)]


def _width(spec: FieldSpec) -> int:
    return 1 if spec.type.has_magnitude or spec.type is FieldType.BINARY else spec.arity


class _RecordFieldDecoder(_Decoder):
    """A verdict or global field, read at the record."""

    def __init__(self, layout: RecordHeadLayout, name: str, head: str, derive) -> None:
        self.layout, self.spec, self.head, self.derive = layout, layout.by_name[name], head, derive
        self.categorical = not self.spec.type.has_magnitude

    def _outputs(self, heads: _Heads, rows: torch.Tensor) -> torch.Tensor:
        """``(asks, members, member width)``."""
        row_outputs = heads.verdict[rows] if self.head == "verdict" else heads.global_[rows]
        start, end = self.layout.slices[self.spec.name]
        spec = self.spec
        return row_outputs[:, start:end].float().reshape(
            -1, spec.count, spec.type.width(spec.arity))

    def candidates(self, heads, rows, columns):
        spec, members = self.spec, self._outputs(heads, rows)
        if spec.type is FieldType.CATEGORICAL:
            return torch.softmax(members[:, 0], dim=-1)
        if spec.type is FieldType.BINARY:
            return torch.sigmoid(members[..., 0])
        value = expected_magnitude(_Magnitude(spec.type), members)
        if spec.type.has_direction:
            value = direction_balance(_Magnitude(spec.type), members) * value
        return value.sum(-1, keepdim=True)

    def record_indicators(self, record):
        row = self.derive(record)
        if row is None:
            return None
        values, mask = row
        if not mask[self.layout.index[self.spec.name]]:
            return None
        first, last = self.layout.target_slices[self.spec.name]
        given = values[first:last]
        spec = self.spec
        if spec.type is FieldType.CATEGORICAL:
            hot = [0.0] * spec.arity
            hot[int(given[0])] = 1.0
            return hot
        if spec.type is FieldType.BINARY:
            return [float(v) for v in given]
        return [float(sum(given))]


@dataclass(frozen=True)
class _Magnitude:
    """The type alone, standing in for a ``FieldSpec`` in the per-entity
    head's decoders: a record field's member reads its columns the same way."""

    type: FieldType
    name: str = "record field"


class _CreatedDecoder(_Decoder):
    """The created-objects head: its expected number of created groups, the
    sum over its slots of the probability each is present."""

    categorical = False

    def candidates(self, heads, rows, columns):
        present = heads.created[rows][:, [slot * CREATED_SLOT_WIDTH + CreatedSlot.PRESENT
                                          for slot in range(CREATED_OBJECT_SLOTS)]]
        return torch.sigmoid(present.float()).sum(-1, keepdim=True)

    def record_indicators(self, record):
        if not supervises_created_objects(record):
            return None
        target, _known = created_objects_targets(record)
        return [float(sum(target[slot * CREATED_SLOT_WIDTH + CreatedSlot.PRESENT]
                          for slot in range(CREATED_OBJECT_SLOTS)))]


def decoder_of(output: str) -> _Decoder | None:
    """The decoder of an output the manifest's map names; None for a name no
    shipped head carries."""
    if output == GATE_OUTPUT:
        return _GateDecoder()
    if output == CREATED_OBJECTS_OUTPUT:
        return _CreatedDecoder()
    if output in FIELDS_BY_NAME:
        return _EntityFieldDecoder(FIELDS_BY_NAME[output])
    if output in VERDICT_LAYOUT.by_name:
        return _RecordFieldDecoder(VERDICT_LAYOUT, output, "verdict", verdict_targets)
    if output in GLOBAL_LAYOUT.by_name:
        return _RecordFieldDecoder(GLOBAL_LAYOUT, output, "global", global_targets)
    return None


class _OutputReading:
    """One rule's output in one stratum: the decoder, the target statistics
    gathered at startup and the candidate and direction they settle on."""

    def __init__(self, decoder: _Decoder) -> None:
        self.decoder = decoder
        self.sums: dict[str, np.ndarray | None] = {FIRES: None, DOES_NOT_FIRE: None}
        self.counts = {FIRES: 0, DOES_NOT_FIRE: 0}
        self.candidate = 0
        self.sign = 1.0
        self.fires: list[list[int]] = []
        self.does_not: list[list[int]] = []

    def observe(self, polarity: str, indicators: Sequence[float]) -> None:
        vector = np.asarray(indicators, dtype=np.float64)
        held = self.sums[polarity]
        self.sums[polarity] = vector if held is None else held + vector
        self.counts[polarity] += 1

    def settle(self) -> bool:
        """Choose the category and direction from the targets; False when no
        ``fires`` record observed the output."""
        if not self.counts[FIRES]:
            return False
        fires = self.sums[FIRES] / self.counts[FIRES]
        does_not = (self.sums[DOES_NOT_FIRE] / self.counts[DOES_NOT_FIRE]
                    if self.counts[DOES_NOT_FIRE] else np.zeros_like(fires))
        difference = fires - does_not
        self.candidate = int(np.argmax(np.abs(difference)))
        self.sign = -1.0 if difference[self.candidate] < 0 else 1.0
        return True

    def __call__(self, heads: _Heads, rows: torch.Tensor,
                 columns: torch.Tensor) -> torch.Tensor:
        """The settled reading at every ask of one chunk: a :data:`Read`."""
        return self.sign * self.decoder.candidates(heads, rows, columns)[:, self.candidate]


class _Observation:
    """One record's target indicators for one output, averaged over the
    tag's entities once every entity of the record is read."""

    def __init__(self, reading: _OutputReading, polarity: str) -> None:
        self.reading, self.polarity = reading, polarity
        self.seen: list[list[float]] = []

    def add(self, indicators: list[float] | None) -> None:
        if indicators is not None:
            self.seen.append(indicators)

    def close(self) -> None:
        if self.seen:
            self.reading.observe(self.polarity, np.mean(np.asarray(self.seen), axis=0))


# ── the host cache ──────────────────────────────────────────────────────


def _stripped(batch):
    """A prepared batch with the host intermediates no device half reads
    dropped, as a scoring step's is."""
    return replace(batch, texts={}, staged=replace(batch.staged, prepared=[]), surfaces=[])


@dataclass(frozen=True)
class _ReadGroup:
    """Every ask of one read in a chunk: rows, gathered columns, and the value
    positions they fill."""

    read: Read
    rows: torch.Tensor
    columns: torch.Tensor
    positions: np.ndarray


def _grouped(reads: Sequence[tuple[int, int, Read, int]]) -> tuple[_ReadGroup, ...]:
    by_read: dict[int, list] = {}
    for row, column, read, position in reads:
        by_read.setdefault(id(read), [read, [], [], []])
        held = by_read[id(read)]
        held[1].append(row)
        held[2].append(column)
        held[3].append(position)
    return tuple(_ReadGroup(read, torch.tensor(rows, dtype=torch.long),
                            torch.tensor(columns, dtype=torch.long),
                            np.asarray(positions, dtype=np.int64))
                 for read, rows, columns, positions in by_read.values())


@dataclass
class _Chunk:
    """One prepared chunk: its packed batch, the slots its per-entity reads
    gather, and its reads grouped by read, each with the value positions it
    fills."""

    batch: object
    index: torch.Tensor
    groups: tuple[_ReadGroup, ...]

    @property
    def reads(self) -> int:
        return sum(len(group.positions) for group in self.groups)

    @property
    def positions(self) -> np.ndarray:
        return np.concatenate([group.positions for group in self.groups]) if self.groups else (
            np.zeros(0, dtype=np.int64))


@dataclass
class _Request:
    """A read one record asks for: at ``subject`` (None at the record), into
    value ``position``, with the target observation it feeds, if any."""

    subject: str | None
    read: Read
    position: int
    observation: _Observation | None = None


class _ChunkBuilder:
    """Collects records with their requests and prepares them a chunk at a time."""

    def __init__(self, batcher, size: int) -> None:
        self.batcher = batcher
        self.size = size
        self.pending: list[tuple[EffectRecord, list[_Request]]] = []
        self.chunks: list[_Chunk] = []

    def add(self, record: EffectRecord, requests: list[_Request],
            texts: Sequence[TextOverride] = ()) -> None:
        for edit in texts:
            self.batcher.override_text(edit.key, edit.text, edit.like, script=edit.script,
                                       role_spans=edit.role_spans)
        if requests:
            self.pending.append((record, requests))
        if len(self.pending) >= self.size:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        records = [record for record, _ in self.pending]
        batch = self.batcher.prepare(records)
        surfaces = batch.surfaces
        targets = None
        if any(r.observation is not None and r.subject is not None
               for _, requests in self.pending for r in requests):
            targets = per_entity_targets(records, surfaces, tuple(FIELDS_BY_NAME.values()))
        columns = [{surface.slots[slot].subject_id: column
                    for column, slot in enumerate(surface.entity_slots)} for surface in surfaces]
        asked: list[dict[int, int]] = [{} for _ in records]
        reads: list[tuple[int, int, Read, int]] = []
        for row, (_, requests) in enumerate(self.pending):
            observations = []
            for request in requests:
                if request.subject is None:
                    reads.append((row, 0, request.read, request.position))
                    continue
                column = columns[row].get(request.subject)
                if column is None:
                    continue
                gather = asked[row].setdefault(column, len(asked[row]))
                reads.append((row, gather, request.read, request.position))
                if request.observation is not None:
                    request.observation.add(
                        request.observation.reading.decoder.entity_indicators(
                            targets, row, column))
                    observations.append(request.observation)
            for observation in {id(o): o for o in observations}.values():
                observation.close()
        width = max([len(a) for a in asked] + [1])
        index = torch.zeros((len(records), width), dtype=torch.long)
        for row, (surface, gathered) in enumerate(zip(surfaces, asked)):
            for column, at in gathered.items():
                index[row, at] = surface.entity_slots[column]
        self.chunks.append(_Chunk(pack(_stripped(batch), rows=len(records)), index,
                                  _grouped(reads)))
        self.pending = []


# ── items ───────────────────────────────────────────────────────────────


@dataclass
class _ProbeItem:
    kind: str
    labels: list[float] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)


@dataclass
class _GridItem:
    outcomes: list[float]
    #: Per cell, the value positions of the grid's records.
    cells: list[list[int]]


@dataclass
class _StratumCache:
    """One stratum's selection half, prepared."""

    stratum: str
    values: int = 0
    chunks: list[_Chunk] = field(default_factory=list)
    probes: dict[str, _ProbeItem] = field(default_factory=dict)
    grids: dict[str, _GridItem] = field(default_factory=dict)
    rules: dict[str, dict[str, _OutputReading]] = field(default_factory=dict)
    steps: list = field(default_factory=list)
    kept_fields: frozenset[str] = frozenset()
    left_out: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    floors: dict[tuple, dict[str, float]] = field(default_factory=dict)

    def position(self) -> int:
        self.values += 1
        return self.values - 1


#: Reads one stratum: a fresh iterable of its shards in path order, each an
#: iterable of its records in line order (``freeze_selection_set.stratum_reader``).
StratumReader = Callable[[], Iterable[Iterable["EffectRecord"]]]


@dataclass(frozen=True)
class FrozenSelection:
    """What a training run's control score is built from: the corpus's
    frozen selection set, and a reader of each validation stratum it halves."""

    selection: SelectionSet
    strata: Mapping[str, StratumReader]

    @property
    def digest(self) -> str:
        return self.selection.digest


def selection_records(
    read: StratumReader, *, games: frozenset[str], records: frozenset[str],
    copies: frozenset[tuple[str, int]],
) -> Iterator[tuple[EffectRecord, bool, bool]]:
    """``(record, distinct, copy)`` for every record of ``games`` that is the
    first occurrence of a record in ``records`` (``distinct``) or a copy in
    ``copies`` (``copy``), copies counted as the freeze counts them: a
    record's earlier occurrences in the stratum, read in shard path order."""
    occurrences: dict[str, int] = defaultdict(int)
    for shard in read():
        for record in shard:
            copy = occurrences[record.record_id]
            occurrences[record.record_id] += 1
            if record.game_id not in games:
                continue
            distinct = copy == 0 and record.record_id in records
            sampled = (record.record_id, copy) in copies
            if distinct or sampled:
                yield record, distinct, sampled


class ControlScorer:
    """The control score's cache and its per-epoch scoring."""

    def __init__(self, strata: dict[str, _StratumCache], *, stratum_weight: float,
                 deviance_floor: float) -> None:
        self.strata = strata
        self.stratum_weight = stratum_weight
        self.deviance_floor = deviance_floor

    # ── startup ──

    @classmethod
    def prepare(
        cls, selection: SelectionSet, readers: Mapping[str, StratumReader], *,
        preparer: StepPreparer, fields: Sequence[tuple[FieldSpec, ...]],
        rule_outputs: Mapping[str, Sequence[str]], mask: str | None,
        stratum_weight: float, deviance_floor: float, editor: GridEditor | None = None,
        chunk_records: int = CHUNK_RECORDS, batch_size: int | None = None,
    ) -> ControlScorer:
        """Read each stratum's selection half once and prepare every item's
        host half (research E2).

        ``fields`` are every field set the run's epochs score (the curriculum's
        before and after its switch): the deviance steps are prepared over
        their union. ``mask`` is the run's ``--rule-mask`` (None reads every
        copy at full weight). Raises :class:`ControlScoreError` on a component
        with no item to read.
        """
        union = _field_union(fields)
        strata = {}
        for stratum in sorted(selection.halves):
            strata[stratum] = _prepare_stratum(
                selection, stratum, readers[stratum], preparer=preparer, union=union,
                rule_outputs=rule_outputs, mask=mask or "hard", editor=editor,
                chunk_records=chunk_records, batch_size=batch_size or chunk_records,
            )
        scorer = cls(strata, stratum_weight=stratum_weight, deviance_floor=deviance_floor)
        logger.info("Control score: %s, %.1f MB on the host.", scorer.describe(),
                    scorer.host_bytes() / 2**20)
        return scorer

    def describe(self) -> str:
        parts = []
        for name, cache in sorted(self.strata.items()):
            parts.append(
                f"{name} {len(cache.probes)} probe type(s), {len(cache.grids)} grid(s), "
                f"{len(cache.rules)} rule(s), {len(cache.kept_fields)} field(s) over "
                f"{sum(c.reads for c in cache.chunks)} read(s) and "
                f"{sum(int(s.targets.gate.shape[0]) for s in cache.steps)} deviance copies")
        return "; ".join(parts)

    def host_bytes(self) -> int:
        """The cache's bytes on the host."""
        return sum(host_bytes(chunk.batch) + chunk.index.nbytes
                   for cache in self.strata.values() for chunk in cache.chunks) + sum(
            host_bytes(step) for cache in self.strata.values() for step in cache.steps)

    # ── each epoch ──

    def epoch(self, forward: RecordForward, losses: StepLossComputer,
              fields: tuple[FieldSpec, ...]) -> ControlScore:
        """Both strata scored on the weights ``forward`` runs; ``fields`` is
        the epoch's field set. Raises :class:`ControlScoreError` on a
        component with nothing to score."""
        return ControlScore(
            {name: self._stratum(cache, forward, losses, fields)
             for name, cache in sorted(self.strata.items())},
            self.stratum_weight,
        )

    def _stratum(self, cache: _StratumCache, forward: RecordForward,
                 losses: StepLossComputer, fields: tuple[FieldSpec, ...]) -> StratumScore:
        values = self._values(cache, forward)
        items = {
            PROBES: {name: probe_item(item.kind, item.labels, values[item.positions])
                     for name, item in sorted(cache.probes.items())},
            GRIDS: {name: _grid_item(item, values) for name, item in sorted(cache.grids.items())},
            RULE_TESTS: {rule: rule_item(
                (_record_values(reading.fires, values), _record_values(reading.does_not, values))
                for reading in outputs.values()) for rule, outputs in sorted(cache.rules.items())},
            DEVIANCE: self._deviance(cache, forward, losses, fields),
        }
        left_out = {component: tuple(sorted(name for name, value in by_item.items()
                                            if math.isnan(value)))
                    for component, by_item in items.items()}
        components = {component: component_score(items[component], stratum=cache.stratum,
                                                  component=component)
                      for component in COMPONENTS}
        return StratumScore(
            components,
            {component: {name: clipped(value) for name, value in by_item.items()
                         if not math.isnan(value)} for component, by_item in items.items()},
            {component: names for component, names in left_out.items() if names},
        )

    def _values(self, cache: _StratumCache, forward: RecordForward) -> np.ndarray:
        values = np.full(cache.values, np.nan)
        if not cache.chunks:
            return values
        batches = (unpack(chunk.batch) for chunk in cache.chunks)
        for chunk, outputs in zip(cache.chunks, forward.prepared(batches)):
            if not chunk.groups:
                continue
            heads = _Heads(outputs, chunk.index)
            read = torch.cat([group.read(heads, group.rows, group.columns).float().reshape(-1)
                              for group in chunk.groups])
            values[chunk.positions] = read.cpu().numpy()
        return values

    def _deviance(self, cache: _StratumCache, forward: RecordForward,
                  losses: StepLossComputer, fields: tuple[FieldSpec, ...]) -> dict[str, float]:
        if not cache.steps:
            return {}
        names = {spec.name for spec in fields}
        prepared = {spec.name for spec in cache.steps[0].fields}
        if not names <= prepared:
            raise ValueError(f"the deviance sample was prepared without {sorted(names - prepared)}")
        key = tuple(spec.name for spec in fields)
        parts = FieldParts()
        kept = KeptTargets() if key not in cache.floors else None
        collectors = (parts,) if kept is None else (parts, kept)
        score_steps((unpack(replace(step, fields=fields)) for step in cache.steps),
                    forward, losses, collectors, device=_device_of(forward))
        if kept is not None:
            cache.floors[key] = constant_predictor_floor(kept.sample, fields=fields)
        explained = explained_deviance(parts.means(), cache.floors[key],
                                       kept=cache.kept_fields, deviance_floor=self.deviance_floor)
        if not explained:
            raise ControlScoreError(
                cache.stratum, SELECTION, DEVIANCE,
                f"every field's constant-prediction loss is at or below the deviance "
                f"floor {self.deviance_floor}, or no kept field is scored this epoch")
        return explained


def _device_of(forward: RecordForward) -> torch.device:
    materializer = forward.materializer
    return materializer.device if materializer is not None else torch.device("cpu")


def _record_values(records: list[list[int]], values: np.ndarray) -> list[float]:
    """Each record's reading, the mean over its entities' finite readings;
    a record with none is dropped."""
    out = []
    for positions in records:
        read = values[positions]
        read = read[np.isfinite(read)]
        if read.size:
            out.append(float(read.mean()))
    return out


def _grid_item(item: _GridItem, values: np.ndarray) -> float:
    means = []
    for positions in item.cells:
        read = values[positions]
        read = read[np.isfinite(read)]
        if not read.size:
            return math.nan
        means.append(float(read.mean()))
    return grid_score(means, item.outcomes)


def _field_union(sets: Sequence[tuple[FieldSpec, ...]]) -> tuple[FieldSpec, ...]:
    seen: dict[str, FieldSpec] = {}
    for specs in sets:
        for spec in specs:
            seen.setdefault(spec.name, spec)
    return tuple(seen.values())


def _prepare_stratum(
    selection: SelectionSet, stratum: str, read: StratumReader, *, preparer: StepPreparer,
    union: tuple[FieldSpec, ...], rule_outputs: Mapping[str, Sequence[str]], mask: str,
    editor: GridEditor | None, chunk_records: int, batch_size: int,
) -> _StratumCache:
    cache = _StratumCache(stratum)
    probes = selection.probes.get(stratum, {}).get(SELECTION, {})
    grids = selection.grids.get(stratum, {}).get(SELECTION, {})
    rule_tests = selection.rule_tests.get(stratum, {}).get(SELECTION, {})
    samples = selection.deviance_samples.get(stratum, {}).get(SELECTION, ())
    if grids and editor is None:
        raise ControlScoreError(stratum, SELECTION, GRIDS,
                                "the selection set holds grids and no grid editor was given")

    # What each record is read for.
    requests: dict[str, list] = defaultdict(list)
    for probe_type, rows in probes.items():
        item = cache.probes[probe_type] = _ProbeItem(selection.probe_kinds[probe_type])
        readout = _readout(probe_type)
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
            reading = readings[output] = _OutputReading(decoder)
            for polarity, held in ((FIRES, reading.fires), (DOES_NOT_FIRE, reading.does_not)):
                for row in sides.get(polarity, ()):
                    requests[row.record_id].append(("rule", reading, polarity, row, held))
        cache.rules[rule] = readings
    grid_records: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
    for name, grid in grids.items():
        cache.grids[name] = _GridItem([cell.outcome for cell in grid.cells],
                                      [[] for _ in grid.cells])
        for row in grid.records:
            grid_records[row.record_id].append((name, row.entities))

    batcher = preparer.scoring_batcher()
    plain = _ChunkBuilder(batcher, chunk_records)
    edited = _ChunkBuilder(preparer.scoring_batcher(), chunk_records)
    sampled: list = []
    steps: list = []
    games = frozenset(selection.halves.get(stratum, {}).get(SELECTION, ()))
    wanted = frozenset(requests) | frozenset(grid_records)
    copies = frozenset((row.record_id, row.copy) for row in samples)
    for record, distinct, copy in selection_records(read, games=games, records=wanted,
                                                    copies=copies):
        if distinct:
            plain.add(record, _requests(record, requests.get(record.record_id, ()), cache))
            for name, entities in grid_records.get(record.record_id, ()):
                for cell, asked in zip(cache.grids[name].cells,
                                       editor.cells(name, record, entities), strict=True):
                    position = cache.position()
                    cell.append(position)
                    edited.add(asked.record, [_Request(asked.entity, _readout(asked.readout),
                                                       position)], asked.texts)
            # One grid record's cells to a chunk: near copies, kept as their
            # differences (``host_cache``).
            edited.flush()
        if copy:
            sampled.append(record)
            if len(sampled) >= batch_size:
                steps.append(pack(preparer.prepare_scoring_step(sampled, union)))
                sampled = []
    plain.flush()
    edited.flush()
    if sampled:
        steps.append(pack(preparer.prepare_scoring_step(sampled, union)))
    cache.chunks = plain.chunks + edited.chunks
    cache.steps = steps
    cache.kept_fields = _kept_fields(selection, stratum, mask)
    for rule in list(cache.rules):
        settled = {output: reading for output, reading in cache.rules[rule].items()
                   if reading.settle()}
        if settled:
            cache.rules[rule] = settled
        else:
            del cache.rules[rule]
            cache.left_out[RULE_TESTS].append(rule)
    for component, held in ((PROBES, cache.probes), (GRIDS, cache.grids),
                            (RULE_TESTS, cache.rules)):
        if not held:
            raise ControlScoreError(stratum, SELECTION, component, "no item to read")
    if not cache.steps or not cache.kept_fields:
        raise ControlScoreError(stratum, SELECTION, DEVIANCE,
                                "no deviance sample or no field the freeze kept")
    if cache.left_out[RULE_TESTS]:
        logger.info("Control score, %s: %d rule(s) have no output a head reads: %s", stratum,
                    len(cache.left_out[RULE_TESTS]), ", ".join(cache.left_out[RULE_TESTS][:12]))
    return cache


def _requests(record: EffectRecord, asked: Iterable[tuple], cache: _StratumCache) -> list:
    """One record's reads: each probe item at its entity, each rule-test
    output at the tag's entities or at the record, with the target
    observations the rule's category is settled from."""
    out: list[_Request] = []
    for ask in asked:
        if ask[0] == "probe":
            _, entity, read, position = ask
            out.append(_Request(entity, read, position))
            continue
        _, reading, polarity, row, held = ask
        decoder = reading.decoder
        if not decoder.per_entity:
            position = cache.position()
            held.append([position])
            out.append(_Request(None, reading, position))
            indicators = decoder.record_indicators(record)
            if indicators is not None:
                reading.observe(polarity, indicators)
            continue
        positions = []
        observation = _Observation(reading, polarity)
        for entity in row.entities:
            position = cache.position()
            positions.append(position)
            out.append(_Request(entity, reading, position, observation))
        if positions:
            held.append(positions)
    return out


def _kept_fields(selection: SelectionSet, stratum: str, mask: str) -> frozenset[str]:
    """The output fields the freeze kept in this half under ``mask``: those
    whose standard error is at most the freeze's item target."""
    limit = float(selection.settings.get("max_item_error", 0.1))
    prefix = item_key(stratum, SELECTION, DEVIANCE, "")
    kept = set()
    for key, error in selection.standard_errors.items():
        if not key.startswith(prefix) or not isinstance(error, dict):
            continue
        if error_of(error.get(mask)) <= limit:
            kept.add(key[len(prefix):])
    return frozenset(kept)


__all__ = [
    "CHUNK_RECORDS", "CellRead", "ControlScore", "ControlScoreError", "ControlScorer",
    "FrozenSelection", "GridEditor", "StratumScore", "TextOverride", "component_score",
    "decoder_of", "explained_deviance", "host_bytes", "mixed", "probe_item", "rule_item",
    "selection_records",
]
