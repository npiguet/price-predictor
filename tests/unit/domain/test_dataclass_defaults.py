from __future__ import annotations

from dataclasses import dataclass, field, fields

import pytest

from price_predictor.domain.dataclass_defaults import field_default


@dataclass
class _Config:
    required: int
    plain: int = 3
    made: list[int] = field(default_factory=lambda: [1])


def _spec(name: str):
    return next(spec for spec in fields(_Config) if spec.name == name)


def test_a_factory_default_is_built_fresh():
    first, second = field_default(_spec("made")), field_default(_spec("made"))
    assert first == [1] and first is not second


def test_a_plain_default_is_returned_as_is():
    assert field_default(_spec("plain")) == 3


def test_a_required_field_reads_as_otherwise_or_raises():
    assert field_default(_spec("required"), otherwise=None) is None
    with pytest.raises(ValueError, match="'required' has no default"):
        field_default(_spec("required"))
