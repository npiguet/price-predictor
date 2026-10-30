"""Decoding the per-entity head's raw outputs into the quantities rules name,
in one place so the gates and the probes read the same columns."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    FieldSpec,
    FieldType,
    HeadField,
    ZoneOutcome,
)
from effects.domain.field_types import DIRECTIONS, SIGNED_DIRECTIONS, Direction


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
    """The predicted magnitude itself: the Poisson rate, ``exp`` of its log."""
    return torch.exp(magnitude_log_rate(spec, slice_))


def direction_balance(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """``P(positive) − P(negative)`` of a signed delta, in ``[-1, 1]``."""
    if not spec.type.has_direction:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no direction")
    probabilities = torch.softmax(slice_[..., SIGNED_DIRECTIONS], dim=-1)
    return (
        probabilities[..., DIRECTIONS.index(Direction.POSITIVE)]
        - probabilities[..., DIRECTIONS.index(Direction.NEGATIVE)]
    )


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
        """A count's expected value, or a signed delta's expected signed value."""
        spec = FIELDS_BY_NAME[field]
        columns = self._columns(spec)
        if not spec.type.has_magnitude:
            raise ValueError(f"{field} is a {spec.type} field and has no expected value")
        value = expected_magnitude(spec, columns)
        if spec.type.has_direction:
            value = direction_balance(spec, columns) * value
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
