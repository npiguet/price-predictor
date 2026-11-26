"""The control score's reads of one chunk's outputs: the heads read once
(:class:`Heads`), a probe or grid readout (:func:`readout_of`), and a rule-test
output's settled reading with the target observations it settles from
(:class:`OutputReading`, :class:`Observation`)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from effects.domain.head_decoding import Decoder
from effects.domain.probe_readouts import READOUTS
from effects.domain.rule_output_targets import TargetMeans

if TYPE_CHECKING:
    from effects.application.record_forward import ForwardOutputs
    from effects.domain.entity_field_targets import EntityTargetBatch

#: The polarities a rule test ranks, ``fires`` first.
FIRES, DOES_NOT_FIRE = "fires", "does_not_fire"


class Heads:
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
Read = Callable[[Heads, torch.Tensor, torch.Tensor], torch.Tensor]


class _ReadoutRead:
    """A :data:`READOUTS` answer at the asked rows and gathered columns (a
    verdict readout ignores the columns)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.readout = READOUTS[name]

    def __call__(self, heads: Heads, rows: torch.Tensor, columns: torch.Tensor):
        return self.readout(heads.gathered, heads.verdict, rows, columns)


#: One read per readout, so a chunk's asks of one readout are read together.
_READOUT_READS: dict[str, _ReadoutRead] = {}


def readout_of(name: str) -> Read:
    read = _READOUT_READS.get(name)
    if read is None:
        read = _READOUT_READS[name] = _ReadoutRead(name)
    return read


class OutputReading:
    """One rule's output in one stratum: the decoder, the target statistics
    gathered at startup and the candidate and direction they settle on."""

    def __init__(self, decoder: Decoder) -> None:
        self.decoder = decoder
        self.means = TargetMeans()
        self.candidate = 0
        self.sign = 1.0
        self.fires: list[list[int]] = []
        self.does_not: list[list[int]] = []
        self._conditional: _ConditionalRead | None = None

    def observe(self, polarity: str, indicators: Sequence[float]) -> None:
        self.means.observe(polarity == FIRES, indicators)

    def positions(self) -> list[int]:
        """Every value position this output's records are read at."""
        return [position for record in (*self.fires, *self.does_not) for position in record]

    def settle(self) -> bool:
        """Choose the category and direction from the targets; False when no
        ``fires`` record observed the output, or when no category's mean
        target differs between the polarities by ``rule_output_targets.MIN_DIFFERENCE``: such
        an output decides nothing about the rule, and its AUC would enter the
        rule's mean at full weight (``TargetMeans.settle``, which the freeze
        reads too)."""
        direction = self.means.settle()
        if direction is None:
            return False
        self.candidate, self.sign = direction.candidate, direction.sign
        return True

    def __call__(self, heads: Heads, rows: torch.Tensor,
                 columns: torch.Tensor) -> torch.Tensor:
        """The settled reading at every ask of one chunk: a :data:`Read`."""
        return self.sign * self.decoder.candidates(heads, rows, columns)[:, self.candidate]

    def at(self, targets: EntityTargetBatch, row: int, column: int) -> Read:
        """The read of one entity: this one, or :attr:`conditional` where the
        decoder leaves the gate out there."""
        if self.decoder.reads_conditionally(targets, row, column):
            return self.conditional
        return self

    @property
    def conditional(self) -> Read:
        """The settled reading without the affected gate, one per output so
        its asks gather into one group."""
        if self._conditional is None:
            self._conditional = _ConditionalRead(self)
        return self._conditional


class _ConditionalRead:
    """An output's settled reading without the affected gate: a :data:`Read`."""

    def __init__(self, reading: OutputReading) -> None:
        self.reading = reading

    def __call__(self, heads: Heads, rows: torch.Tensor,
                 columns: torch.Tensor) -> torch.Tensor:
        reading = self.reading
        read = reading.decoder.conditional_candidates(heads, rows, columns)
        return reading.sign * read[:, reading.candidate]


class Observation:
    """One record's target indicators for one output, averaged over the
    tag's entities once every entity of the record is read."""

    def __init__(self, reading: OutputReading, polarity: str) -> None:
        self.reading, self.polarity = reading, polarity
        self.seen: list[list[float]] = []

    def add(self, indicators: list[float] | None) -> None:
        if indicators is not None:
            self.seen.append(indicators)

    def close(self) -> None:
        if self.seen:
            self.reading.observe(self.polarity, np.mean(np.asarray(self.seen), axis=0))
