"""What an output field's type fixes about its columns and its target.

Torch-free, because the target derivation every corpus worker runs reads the
layout these facts build; what each type does with a tensor — its loss, its
best constant, its magnitude — is :mod:`effects.domain.field_kinds`.
"""

from __future__ import annotations

from enum import StrEnum

#: A signed delta's direction classes, in the head's column order; a target's
#: sign plus one is its class.
DIRECTIONS: tuple[str, ...] = ("negative", "zero", "positive")
#: A signed delta's columns: a logit per direction, then the magnitude's log
#: rate under the count loss.
SIGNED_DIRECTIONS = slice(0, len(DIRECTIONS))
SIGNED_MAGNITUDE = len(DIRECTIONS)
SIGNED_WIDTH = SIGNED_MAGNITUDE + 1


class FieldType(StrEnum):
    """How a field is predicted and scored (FR-080), shared by the per-entity
    head's fields and the value head's targets."""

    BINARY = "binary"
    MULTI_BINARY = "multi_binary"
    COUNT = "count"
    SIGNED_DELTA = "signed_delta"
    CATEGORICAL = "categorical"

    def width(self, arity: int) -> int:
        """The field's columns, given its ``arity`` classes or members."""
        if self is FieldType.SIGNED_DELTA:
            return SIGNED_WIDTH
        return arity if self in _A_COLUMN_PER_CLASS else 1

    @property
    def per_member(self) -> bool:
        """Whether a target holds one value per member rather than one value:
        a categorical's target stays one class index, since cross-entropy
        takes indices, not one-hots."""
        return self is FieldType.MULTI_BINARY

    @property
    def magnitude_column(self) -> int | None:
        """Where among the field's columns the magnitude's log rate sits, under
        the count loss; None for a type that predicts no magnitude."""
        return _MAGNITUDE_COLUMNS.get(self)

    @property
    def has_magnitude(self) -> bool:
        return self.magnitude_column is not None

    @property
    def has_direction(self) -> bool:
        return self is FieldType.SIGNED_DELTA


_A_COLUMN_PER_CLASS = frozenset({FieldType.MULTI_BINARY, FieldType.CATEGORICAL})
_MAGNITUDE_COLUMNS: dict[FieldType, int] = {
    FieldType.COUNT: 0, FieldType.SIGNED_DELTA: SIGNED_MAGNITUDE,
}
