"""Decoding the per-entity head's raw outputs into the quantities rules name,
in one place so the gates and the probes read the same columns; and the
control score's decoders (:func:`decoder_of`), one rule-test output's
candidate readings over whichever shipped head carries it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

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
    FieldType,
    HeadField,
    RecordHeadLayout,
    ZoneOutcome,
)
from effects.domain.field_types import DIRECTIONS, SIGNED_DIRECTIONS, Direction
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, GATE_OUTPUT

if TYPE_CHECKING:
    from effects.domain.entity_field_targets import EntityTargetBatch


def field_outputs(outputs: torch.Tensor, spec: FieldSpec) -> torch.Tensor:
    """``spec``'s columns of a per-entity output, over any leading dims."""
    start, end = FIELD_SLICES[spec.name]
    return outputs[..., start:end]


def magnitude_log_rate(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """The predicted magnitude's log rate, from ``spec``'s columns."""
    column = spec.type.magnitude_column
    if column is None:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no magnitude")
    return slice_[..., column]


def expected_magnitude(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """The predicted magnitude itself: the negative binomial mean, ``exp`` of
    its log."""
    return torch.exp(magnitude_log_rate(spec, slice_))


def nonzero_direction_balance(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """``(P(+) − P(−)) / (P(+) + P(−))`` of a signed delta, in ``[-1, 1]``: the
    sign's lean among the non-zero outcomes, the zero class left out."""
    if not spec.type.has_direction:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no direction")
    probabilities = torch.softmax(slice_[..., SIGNED_DIRECTIONS], dim=-1)
    positive = probabilities[..., DIRECTIONS.index(Direction.POSITIVE)]
    negative = probabilities[..., DIRECTIONS.index(Direction.NEGATIVE)]
    return (positive - negative) / (positive + negative).clamp_min(torch.finfo(positive.dtype).tiny)


@dataclass(frozen=True, slots=True)
class HeadReading:
    """One entity's per-entity output, over any leading dims, read as the rules ask.

    Unconditional by default, ``P(affected) ×`` the field's reading, because
    each field head trained only on affected entities and a board change can
    move either factor. ``conditional=True`` reads the field head alone. A
    field with no silent value also reads conditional on the record having
    written it.
    """

    outputs: torch.Tensor

    @property
    def p_affected(self) -> torch.Tensor:
        return torch.sigmoid(self.outputs[..., GATE_INDEX].float())

    def _columns(self, spec: FieldSpec) -> torch.Tensor:
        return field_outputs(self.outputs, spec).float()

    def _gated(self, value: torch.Tensor, conditional: bool) -> torch.Tensor:
        return value if conditional else self.p_affected * value

    def expected(self, field: str, *, conditional: bool = False) -> torch.Tensor:
        """A count's expected value, or a signed delta's expected signed value.

        A signed delta's magnitude column is trained on every scored row, zeros
        included, so ``exp(log_rate)`` already is the mean absolute value and
        the zero class is not discounted again: the value is that mean times
        the sign's lean among the non-zero outcomes,
        ``(P(+) − P(−)) / (P(+) + P(−))``. It is exact when both signs share a
        magnitude; with different magnitudes it reads their mean absolute value
        at each sign's share.
        """
        spec = FIELDS_BY_NAME[field]
        columns = self._columns(spec)
        if not spec.type.has_magnitude:
            raise ValueError(f"{field} is a {spec.type} field and has no expected value")
        value = expected_magnitude(spec, columns)
        if spec.type.has_direction:
            value = nonzero_direction_balance(spec, columns) * value
        return self._gated(value, conditional)

    def p_class(
        self, field: str, category: str, *, conditional: bool = False,
    ) -> torch.Tensor:
        """The probability a categorical field takes ``category``."""
        spec = FIELDS_BY_NAME[field]
        if spec.type is not FieldType.CATEGORICAL:
            raise ValueError(f"{field} is a {spec.type} field and has no classes")
        probabilities = torch.softmax(self._columns(spec), dim=-1)
        value = probabilities[..., spec.classes.index(category)]
        return self._gated(value, conditional)

    def p_zone(self, outcome: ZoneOutcome, *, conditional: bool = False) -> torch.Tensor:
        """The probability of one zone outcome: ``died``, ``exiled``, …"""
        return self.p_class(HeadField.ZONE_OUTCOME, outcome, conditional=conditional)

    def zone_probabilities(self, *, conditional: bool = False) -> torch.Tensor:
        """Every zone outcome's probability, in :data:`ZONE_OUTCOMES` order."""
        spec = FIELDS_BY_NAME[HeadField.ZONE_OUTCOME]
        probabilities = torch.softmax(self._columns(spec), dim=-1)
        return probabilities if conditional else (
            self.p_affected[..., None] * probabilities
        )


class HeadOutputs(Protocol):
    """One chunk's head outputs as a decoder reads them: the per-entity head
    gathered at the chunk's asked slots, and the record heads
    (``control_reads.Heads``)."""

    gathered: torch.Tensor

    @property
    def verdict(self) -> torch.Tensor: ...

    @property
    def global_(self) -> torch.Tensor: ...

    @property
    def created(self) -> torch.Tensor: ...



class Decoder:
    """One rule-test output's candidate readings and the matching target
    indicators: a candidate per category (or member), or one expected value.
    Its targets are :attr:`output`'s, read by ``rule_output_targets``, the
    definition the freeze leaves an output out by."""

    #: The output the decoder reads, as the manifest's map names it.
    output = ""
    #: Whether the output is read at the tag's outcome entities rather than the record.
    per_entity = False
    #: Whether the candidates are probabilities of categories (else amounts).
    categorical = True

    def candidates(self, heads: HeadOutputs, rows: torch.Tensor,
                   columns: torch.Tensor) -> torch.Tensor:
        """``(asks, candidates)``: each ask's reading of every candidate."""
        raise NotImplementedError

    def conditional_candidates(self, heads: HeadOutputs, rows: torch.Tensor,
                               columns: torch.Tensor) -> torch.Tensor:
        """:meth:`candidates` read without the affected gate, where
        :meth:`reads_conditionally` says so; the same reading elsewhere."""
        return self.candidates(heads, rows, columns)

    def reads_conditionally(self, targets: EntityTargetBatch, row: int, column: int) -> bool:
        """Whether one entity's reading leaves the affected gate out."""
        return False


class _GateDecoder(Decoder):
    output = GATE_OUTPUT
    per_entity = True

    def candidates(self, heads, rows, columns):
        return torch.sigmoid(heads.gathered[rows, columns, GATE_INDEX].float()).unsqueeze(-1)


class _EntityFieldDecoder(Decoder):
    """A per-entity field, read unconditionally (gated by the affected gate,
    a legality field ungated), except on a record that trains no gate (a cost
    half, a playability record, a rewrite whose outcome is partly unknown),
    where a gated field trains at the entities the gate's target marks and is
    read conditionally, as training reads it (task G11b)."""

    per_entity = True

    def __init__(self, spec: FieldSpec) -> None:
        self.spec = spec
        self.output = spec.name
        self.categorical = not spec.type.has_magnitude

    def candidates(self, heads, rows, columns):
        return self._read(heads, rows, columns, conditional=not self.spec.gated)

    def conditional_candidates(self, heads, rows, columns):
        return self._read(heads, rows, columns, conditional=True)

    def _read(self, heads, rows, columns, *, conditional: bool) -> torch.Tensor:
        outputs = heads.gathered[rows, columns]
        spec = self.spec
        if spec.type.has_magnitude:
            return HeadReading(outputs).expected(spec.name, conditional=conditional).unsqueeze(-1)
        start, end = FIELD_SLICES[spec.name]
        read = outputs[:, start:end].float()
        if spec.type is FieldType.CATEGORICAL:
            value = torch.softmax(read, dim=-1)
        else:
            value = torch.sigmoid(read)
        if not conditional:
            value = HeadReading(outputs).p_affected.unsqueeze(-1) * value
        return value

    def reads_conditionally(self, targets, row, column):
        return self.spec.gated and not bool(targets.gate_scored[row, column])


class _RecordFieldDecoder(Decoder):
    """A verdict or global field, read at the record."""

    def __init__(self, layout: RecordHeadLayout, name: str, head: str) -> None:
        self.layout, self.spec, self.head, self.output = layout, layout.by_name[name], head, name
        self.categorical = not self.spec.type.has_magnitude

    def _outputs(self, heads: HeadOutputs, rows: torch.Tensor) -> torch.Tensor:
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
            value = nonzero_direction_balance(_Magnitude(spec.type), members) * value
        return value.sum(-1, keepdim=True)


@dataclass(frozen=True)
class _Magnitude:
    """The type alone, standing in for a ``FieldSpec`` in the per-entity
    head's decoders: a record field's member reads its columns the same way."""

    type: FieldType
    name: str = "record field"


class _CreatedDecoder(Decoder):
    """The created-objects head: its expected number of created groups, the
    sum over its slots of the probability each is present."""

    output = CREATED_OBJECTS_OUTPUT
    categorical = False

    def candidates(self, heads, rows, columns):
        present = heads.created[rows][:, [slot * CREATED_SLOT_WIDTH + CreatedSlot.PRESENT
                                          for slot in range(CREATED_OBJECT_SLOTS)]]
        return torch.sigmoid(present.float()).sum(-1, keepdim=True)


def decoder_of(output: str) -> Decoder | None:
    """The decoder of an output the manifest's map names; None for a name no
    shipped head carries."""
    if output == GATE_OUTPUT:
        return _GateDecoder()
    if output == CREATED_OBJECTS_OUTPUT:
        return _CreatedDecoder()
    if output in FIELDS_BY_NAME:
        return _EntityFieldDecoder(FIELDS_BY_NAME[output])
    if output in VERDICT_LAYOUT.by_name:
        return _RecordFieldDecoder(VERDICT_LAYOUT, output, "verdict")
    if output in GLOBAL_LAYOUT.by_name:
        return _RecordFieldDecoder(GLOBAL_LAYOUT, output, "global")
    return None
