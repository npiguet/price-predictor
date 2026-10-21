"""Decoding a per-entity head's columns into the quantity a rule names."""

from __future__ import annotations

import math

import pytest
import torch

from effects.domain.effect_head_layout import FIELDS_BY_NAME, PER_ENTITY_WIDTH
from effects.domain.head_decoding import (
    direction_balance,
    expected_magnitude,
    field_outputs,
    magnitude_log_rate,
)


def test_a_count_s_magnitude_is_its_only_column():
    spec = FIELDS_BY_NAME["damage_taken"]
    vector = torch.zeros(PER_ENTITY_WIDTH)
    field_outputs(vector, spec)[0] = math.log(3.0)
    assert float(expected_magnitude(spec, field_outputs(vector, spec))) == pytest.approx(3.0)


def test_a_signed_delta_s_magnitude_is_its_last_column():
    """Three direction logits come first; the magnitude follows them."""
    spec = FIELDS_BY_NAME["life_delta"]
    slice_ = torch.tensor([0.0, 0.0, 0.0, math.log(5.0)])
    assert float(magnitude_log_rate(spec, slice_)) == pytest.approx(math.log(5.0))
    assert float(expected_magnitude(spec, slice_)) == pytest.approx(5.0)


def test_direction_balance_is_positive_minus_negative():
    spec = FIELDS_BY_NAME["life_delta"]
    slice_ = torch.tensor([-10.0, -10.0, 10.0, 0.0])
    assert float(direction_balance(spec, slice_)) == pytest.approx(1.0, abs=1e-6)


def test_decoding_works_over_leading_dimensions():
    spec = FIELDS_BY_NAME["power_delta"]
    outputs = torch.zeros(2, 3, PER_ENTITY_WIDTH)
    assert expected_magnitude(spec, field_outputs(outputs, spec)).shape == (2, 3)


def test_a_field_without_a_magnitude_is_refused():
    spec = FIELDS_BY_NAME["zone_outcome"]
    with pytest.raises(ValueError, match="no magnitude"):
        magnitude_log_rate(spec, torch.zeros(spec.width))
    with pytest.raises(ValueError, match="no direction"):
        direction_balance(spec, torch.zeros(spec.width))
