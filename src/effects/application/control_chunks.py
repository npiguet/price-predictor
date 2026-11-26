"""The control score's host cache: records with the reads they ask for,
prepared and packed a chunk at a time (:class:`ChunkBuilder`), each chunk's
reads grouped by read (``host_cache``; contracts/selection-set.md § Score)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from effects.application.control_reads import Observation, Read
from effects.application.host_cache import pack
from effects.application.step_preparation import stripped
from effects.domain.effect_head_layout import FIELDS_BY_NAME
from effects.domain.effect_targets import RecordTargets, derive_targets
from effects.domain.grid_edits import TextOverride
from effects.domain.rule_output_targets import entity_indicators
from effects.domain.surface_collation import entity_target_tensors

if TYPE_CHECKING:
    from effects.domain.entity_field_targets import EntityTargetBatch
    from effects.domain.records import EffectRecord


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
class Chunk:
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
class Request:
    """A read one record asks for: at ``subject`` (None at the record), into
    value ``position``, with the target observation it feeds, if any."""

    subject: str | None
    read: Read
    position: int
    observation: Observation | None = None


class ChunkBuilder:
    """Collects records with their requests and prepares them a chunk at a time."""

    def __init__(self, batcher, size: int) -> None:
        self.batcher = batcher
        self.size = size
        self.pending: list[tuple[EffectRecord, list[Request]]] = []
        self.chunks: list[Chunk] = []
        #: Requests at an entity the record's surface gives no slot, each
        #: read nowhere: their count and a few of their refs.
        self.unplaced = 0
        self.unplaced_refs: list[str] = []
        #: Rule-test requests at an entity where the record defines no target
        #: for the output, each read nowhere.
        self.untargeted = 0

    def add(self, record: EffectRecord, requests: list[Request],
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
        targets = _RowTargets.of(records, surfaces) if _observes_entities(self.pending) else None
        placing = _Placing(targets, [_columns_of(surface) for surface in surfaces],
                           [{} for _ in records])
        for row, (_, requests) in enumerate(self.pending):
            self._place_row(placing, row, requests)
        self.chunks.append(Chunk(pack(stripped(batch), rows=len(records)),
                                 _gather_index(surfaces, placing.asked),
                                 _grouped(placing.reads)))
        self.pending = []

    def _place_row(self, placing: _Placing, row: int, requests: list[Request]) -> None:
        """One record's requests placed in the chunk; each target observation
        they fed closed once."""
        observations = []
        for request in requests:
            observation = self._place(placing, row, request)
            if observation is not None:
                observations.append(observation)
        for observation in {id(o): o for o in observations}.values():
            observation.close()

    def _place(self, placing: _Placing, row: int, request: Request) -> Observation | None:
        """One request placed at its record's row: at the record, or at its
        entity's gathered column (counted unplaced where the surface gives the
        entity no slot). Returns the target observation it fed, if any."""
        if request.subject is None:
            placing.reads.append((row, 0, request.read, request.position))
            return None
        column = placing.columns[row].get(request.subject)
        if column is None:
            self._unplaced(request)
            return None
        read = request.read
        if request.observation is not None:
            read = self._observed(request, placing.targets, row, column)
            if read is None:
                return None
        gather = placing.asked[row].setdefault(column, len(placing.asked[row]))
        placing.reads.append((row, gather, read, request.position))
        return request.observation

    def _unplaced(self, request: Request) -> None:
        self.unplaced += 1
        if len(self.unplaced_refs) < 5:
            self.unplaced_refs.append(str(request.subject))

    def _observed(self, request: Request, targets: _RowTargets,
                  row: int, column: int) -> Read | None:
        """A rule test's read of one entity, its target observed; None where
        the record does not say what the output is there, so the reading
        would compare nothing."""
        observation = request.observation
        reading = observation.reading
        indicators = entity_indicators(reading.decoder.output, targets.derived[row],
                                       request.subject)
        if indicators is None:
            self.untargeted += 1
            return None
        observation.add(indicators)
        return reading.at(targets.batch, row, column)


@dataclass(frozen=True)
class _Placing:
    """A chunk's reads as its requests are placed: each record's entity
    columns (``subject_id -> column``), the columns each record's reads
    gather (``column -> gathered index``), and the reads themselves
    (``(row, gathered index, read, value position)``)."""

    targets: _RowTargets | None
    columns: list[dict[str, int]]
    asked: list[dict[int, int]]
    reads: list[tuple[int, int, Read, int]] = field(default_factory=list)


def _observes_entities(pending: Sequence[tuple[EffectRecord, list[Request]]]) -> bool:
    """Whether a pending request observes a target at an entity, so the
    chunk's targets are built."""
    return any(r.observation is not None and r.subject is not None
               for _, requests in pending for r in requests)


def _columns_of(surface) -> dict[str, int]:
    return {surface.slots[slot].subject_id: column
            for column, slot in enumerate(surface.entity_slots)}


def _gather_index(surfaces: Sequence, asked: list[dict[int, int]]) -> torch.Tensor:
    """Each record's gathered columns as the slots they read, padded with
    slot 0 to the widest record's (at least one column)."""
    width = max([len(a) for a in asked] + [1])
    index = torch.zeros((len(asked), width), dtype=torch.long)
    for row, (surface, gathered) in enumerate(zip(surfaces, asked)):
        for column, at in gathered.items():
            index[row, at] = surface.entity_slots[column]
    return index


@dataclass(frozen=True)
class _RowTargets:
    """A chunk's targets: the per-entity batch a read chooses its gate by,
    and each record's derived targets its indicators are read from."""

    batch: EntityTargetBatch
    derived: list[RecordTargets]

    @classmethod
    def of(cls, records: list[EffectRecord], surfaces: list) -> _RowTargets:
        derived = [derive_targets(record) for record in records]
        return cls(entity_target_tensors(surfaces, derived, tuple(FIELDS_BY_NAME.values())),
                   derived)
