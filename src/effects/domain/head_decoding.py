"""Decoding the per-entity head's raw outputs into the quantities rules name.

No field of the head is a bare value. A count predicts a log rate; a signed
delta predicts three direction logits (negative, zero, positive) and then the
same log-rate magnitude; a categorical predicts logits over its classes. Every
reader that wants "how much damage" or "how likely to die" has to undo that
layout, and doing it in one place is what keeps gate 1's deviance, gate 2's
direction test and the knowledge probes reading the same column.

And no field head stands alone. Each is supervised only on entities the gate's
target marks affected, so it answers "*if* this entity is affected, how much"
and says nothing about whether it is. :class:`HeadReading` is the one place
that combines the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from effects.domain.effect_model import (
    DURATIONS,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    ZONE_OUTCOMES,
    FieldSpec,
    FieldType,
)

#: Field types that predict a magnitude under the count loss — "how much
#: damage", "how many cards", "how far the life total moved".
MAGNITUDE_TYPES: frozenset[FieldType] = frozenset(
    {FieldType.COUNT, FieldType.SIGNED_DELTA},
)

#: Each categorical field's classes, in the head's column order.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "zone_outcome": ZONE_OUTCOMES,
    "pt_duration": DURATIONS,
    "type_color_duration": DURATIONS,
}


def field_outputs(outputs: torch.Tensor, spec: FieldSpec) -> torch.Tensor:
    """``spec``'s columns of a per-entity output, over any leading dims."""
    start, end = FIELD_SLICES[spec.name]
    return outputs[..., start:end]


def magnitude_log_rate(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """The predicted magnitude's log rate, from ``spec``'s columns.

    A count predicts only that column and a signed delta predicts direction
    then magnitude, so it is the last column either way.
    """
    if spec.type not in MAGNITUDE_TYPES:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no magnitude")
    return slice_[..., -1]


def expected_magnitude(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """The predicted magnitude itself: the Poisson rate, ``exp`` of its log."""
    return torch.exp(magnitude_log_rate(spec, slice_))


def direction_balance(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """``P(positive) − P(negative)`` of a signed delta, in ``[-1, 1]``."""
    if spec.type is not FieldType.SIGNED_DELTA:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no direction")
    probabilities = torch.softmax(slice_[..., :3], dim=-1)
    return probabilities[..., 2] - probabilities[..., 0]


@dataclass(frozen=True, slots=True)
class HeadReading:
    """One entity's per-entity output, over any leading dims, read as the rules ask.

    Unconditional by default: an expected value is ``P(affected) × E[field |
    affected]`` and a class probability ``P(affected) × P(class | affected)``,
    because a change to the board can move either factor. Removing first strike
    leaves the creature it fights unaffected in that step: the gate falls while
    the damage head, which never trained on an unaffected entity, still reads
    how much a hit would deal. ``conditional=True`` reads the field head alone,
    for a question already restricted to affected entities.
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
        match spec.type:
            case FieldType.COUNT:
                value = expected_magnitude(spec, columns)
            case FieldType.SIGNED_DELTA:
                value = direction_balance(spec, columns) * expected_magnitude(spec, columns)
            case _:
                raise ValueError(
                    f"{field} is a {spec.type} field and has no expected value"
                )
        return self._gated(value, conditional)

    def p_class(
        self, field: str, category: str, *, conditional: bool = False,
    ) -> torch.Tensor:
        """The probability a categorical field takes ``category``."""
        spec = FIELDS_BY_NAME[field]
        if spec.type is not FieldType.CATEGORICAL:
            raise ValueError(f"{field} is a {spec.type} field and has no classes")
        probabilities = torch.softmax(self._columns(spec), dim=-1)
        value = probabilities[..., CATEGORIES[field].index(category)]
        return self._gated(value, conditional)

    def p_zone(self, outcome: str, *, conditional: bool = False) -> torch.Tensor:
        """The probability of one zone outcome: ``died``, ``exiled``, …"""
        return self.p_class("zone_outcome", outcome, conditional=conditional)

    def zone_probabilities(self, *, conditional: bool = False) -> torch.Tensor:
        """Every zone outcome's probability, in :data:`ZONE_OUTCOMES` order."""
        spec = FIELDS_BY_NAME["zone_outcome"]
        probabilities = torch.softmax(self._columns(spec), dim=-1)
        return probabilities if conditional else (
            self.p_affected[..., None] * probabilities
        )
