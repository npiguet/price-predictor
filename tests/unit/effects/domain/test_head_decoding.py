"""Decoding a per-entity head's columns into the quantity a rule names."""

from __future__ import annotations

import math

import pytest
import torch

from effects.domain.effect_model import FIELDS_BY_NAME, PER_ENTITY_WIDTH
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


# ── HeadReading: one entity's output, read the way the head was trained ────


def _outputs(gate_logit: float, **columns: float) -> torch.Tensor:
    from effects.domain.effect_model import FIELD_SLICES, GATE_INDEX

    vector = torch.zeros(PER_ENTITY_WIDTH)
    vector[GATE_INDEX] = gate_logit
    for name, value in columns.items():
        vector[FIELD_SLICES[name][0]] = value
    return vector


class TestHeadReading:
    """Field heads train only where the gate fires, so each reads "if affected"."""

    def test_p_affected_is_the_gate_s_sigmoid(self):
        from effects.domain.head_decoding import HeadReading

        assert float(HeadReading(_outputs(0.0)).p_affected) == pytest.approx(0.5)

    def test_an_expected_count_is_the_gate_times_the_rate(self):
        from effects.domain.head_decoding import HeadReading

        reading = HeadReading(_outputs(0.0, damage_taken=math.log(4.0)))
        assert float(reading.expected("damage_taken")) == pytest.approx(2.0)
        assert float(reading.expected("damage_taken", conditional=True)) == (
            pytest.approx(4.0)
        )

    def test_an_expected_signed_delta_carries_its_direction_and_the_gate(self):
        from effects.domain.effect_model import FIELD_SLICES
        from effects.domain.head_decoding import HeadReading

        vector = _outputs(20.0)
        start, _end = FIELD_SLICES["life_delta"]
        vector[start : start + 4] = torch.tensor([20.0, -20.0, -20.0, math.log(3.0)])
        assert float(HeadReading(vector).expected("life_delta")) == pytest.approx(
            -3.0, abs=1e-4,
        )

    def test_a_class_probability_is_gated_unless_asked_conditional(self):
        from effects.domain.effect_model import FIELD_SLICES, ZONE_OUTCOMES
        from effects.domain.head_decoding import HeadReading

        vector = _outputs(0.0)
        start, end = FIELD_SLICES["zone_outcome"]
        vector[start:end] = -20.0
        vector[start + ZONE_OUTCOMES.index("died")] = 20.0
        reading = HeadReading(vector)
        assert float(reading.p_zone("died")) == pytest.approx(0.5, abs=1e-6)
        assert float(reading.p_zone("died", conditional=True)) == pytest.approx(
            1.0, abs=1e-6,
        )

    def test_it_reads_over_leading_dimensions(self):
        from effects.domain.head_decoding import HeadReading

        reading = HeadReading(torch.zeros(2, 3, PER_ENTITY_WIDTH))
        assert reading.expected("cards_drawn").shape == (2, 3)
        assert reading.p_class("pt_duration", "end_of_turn").shape == (2, 3)

    def test_a_field_with_no_expected_value_is_refused(self):
        from effects.domain.head_decoding import HeadReading

        with pytest.raises(ValueError, match="tap_state"):
            HeadReading(_outputs(0.0)).expected("tap_state")
        with pytest.raises(ValueError, match="damage_taken"):
            HeadReading(_outputs(0.0)).p_class("damage_taken", "died")
