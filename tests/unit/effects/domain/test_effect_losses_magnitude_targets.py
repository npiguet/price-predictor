"""The losses read a magnitude capped and a count nonnegative."""

from __future__ import annotations

import torch

from effects.domain.effect_head_layout import FIELDS_BY_NAME
from effects.domain.effect_losses import field_loss
from effects.domain.loss_floor import constant_predictor_outputs
from tests.unit.effects.builders import fields_everywhere, make_entity_targets

ONE = torch.tensor(0.0)  # a log-dispersion of zero: a dispersion of one


def _floor(value):
    """The constant predictor's outputs for a damage_taken target of ``value``."""
    batch = make_entity_targets(
        gate=torch.ones(1, 2),
        fields=fields_everywhere({"damage_taken": torch.tensor([[2.0, value]])}),
        mask=torch.ones(1, 2),
    )
    return constant_predictor_outputs([batch], fields=(FIELDS_BY_NAME["damage_taken"],))


class TestMagnitudeCap:
    """A magnitude above the cap scores as the cap.

    Above about forty, an amount is the game state compounding — a counter
    doubling run to two million — not what the ability's text says, and one
    such target under a Poisson loss pulls a step a million times harder than
    an ordinary one.
    """

    def test_a_count_above_the_cap_scores_as_the_cap(self):
        from effects.domain.input_layout import MAGNITUDE_CAP
        spec = FIELDS_BY_NAME["damage_taken"]
        prediction = torch.zeros(1, 1)
        assert field_loss(spec, prediction, torch.tensor([2_000_000.0]), log_dispersion=ONE) == \
            field_loss(spec, prediction, torch.tensor([MAGNITUDE_CAP]), log_dispersion=ONE)

    def test_a_signed_delta_keeps_its_direction_and_caps_its_size(self):
        from effects.domain.input_layout import MAGNITUDE_CAP
        spec = FIELDS_BY_NAME["power_delta"]
        prediction = torch.zeros(1, 4)
        for sign in (1.0, -1.0):
            assert field_loss(
                spec, prediction, torch.tensor([sign * 2_098_760.0]), log_dispersion=ONE,
            ) == field_loss(
                spec, prediction, torch.tensor([sign * MAGNITUDE_CAP]), log_dispersion=ONE,
            )

    def test_the_floor_is_computed_over_capped_targets(self):
        from effects.domain.input_layout import MAGNITUDE_CAP

        assert torch.equal(_floor(2_000_000.0), _floor(MAGNITUDE_CAP))

    def test_the_evaluation_deviance_caps_the_observed_amount(self):
        import numpy as np

        from effects.domain.gate_metrics import mean_deviance
        from effects.domain.input_layout import MAGNITUDE_CAP
        predicted = np.array([3.0, 5.0])
        dispersion = np.array(0.0)
        assert mean_deviance(predicted, np.array([1.0, 2_000_000.0]), dispersion) == \
            mean_deviance(predicted, np.array([1.0, MAGNITUDE_CAP]), dispersion)


class TestCountTargetsAreNonnegative:
    """A count below zero has no likelihood (FR-080, FR-081).

    ``damage_taken`` is decremented by a healing event, so a heal with no
    damage before it is a negative count, and a negative target makes a count
    NLL unbounded below: the head could lower the loss without limit by
    raising its rate.
    """

    def test_a_negative_count_scores_as_zero(self):
        spec = FIELDS_BY_NAME["damage_taken"]
        for log_rate in (-2.0, 0.0, 3.0):
            prediction = torch.full((1, 1), log_rate)
            assert field_loss(spec, prediction, torch.tensor([-3.0]), log_dispersion=ONE) == \
                field_loss(spec, prediction, torch.tensor([0.0]), log_dispersion=ONE)

    def test_the_floor_reads_a_negative_count_as_zero(self):
        assert torch.equal(_floor(-5.0), _floor(0.0))
