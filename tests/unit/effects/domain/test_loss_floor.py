"""The constant-predictor floor a field's loss is compared against (FR-127c)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from effects.domain.effect_head_layout import FIELD_SLICES, FIELDS_BY_NAME, PER_ENTITY_WIDTH
from effects.domain.effect_losses import LossParts, per_entity_loss
from effects.domain.entity_field_targets import EntityFieldTargets
from effects.domain.loss_floor import (
    EntityTargetBatch,
    PartsMean,
    constant_predictor_floor,
    constant_predictor_outputs,
)
from effects.domain.state_snapshot import COLORS


class TestConstantPredictorFloor:
    """What a field's loss is compared against (FR-127c).

    A loss alone cannot say whether the head learned the field or is repeating
    its base rate, so every card-disjoint term is scaled by the loss the best
    constant prediction reaches on the same targets under the same masking.
    """

    def _batch(self, gate, fields, mask=None):
        gate = torch.tensor(gate, dtype=torch.float32)
        return EntityTargetBatch(
            gate=gate,
            fields=EntityFieldTargets.everywhere({
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in fields.items()
            }),
            mask=torch.ones_like(gate) if mask is None
            else torch.tensor(mask, dtype=torch.float32),
        )

    def test_a_binary_fields_floor_is_the_entropy_of_its_positive_rate(self):
        """Eight affected entities, a quarter of them positive, two records."""
        values = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        batch = self._batch(gate=[[1.0] * 4] * 2, fields={"tap_state": values})

        floor = constant_predictor_floor(
            [batch], fields=(FIELDS_BY_NAME["tap_state"],),
        )

        rate = 0.25
        entropy = -(rate * math.log(rate) + (1 - rate) * math.log(1 - rate))
        assert floor["tap_state"] == pytest.approx(8 * entropy / 2, abs=1e-5)

    def test_a_count_fields_floor_is_the_poisson_nll_of_its_mean(self):
        values = [[0.0, 1.0, 2.0, 3.0], [4.0, 0.0, 1.0, 2.0]]
        batch = self._batch(gate=[[1.0] * 4] * 2, fields={"damage_taken": values})

        floor = constant_predictor_floor(
            [batch], fields=(FIELDS_BY_NAME["damage_taken"],),
        )

        target = torch.tensor(values).reshape(-1)
        expected = functional.poisson_nll_loss(
            torch.full_like(target, math.log(float(target.mean()))), target,
            log_input=True, full=True, reduction="sum",
        ) / 2
        assert floor["damage_taken"] == pytest.approx(float(expected), abs=1e-5)

    def test_the_constant_prediction_scores_exactly_the_floor(self):
        """Zero explained deviance is a head sitting at the base rate."""
        names = ("tap_state", "damage_taken", "zone_outcome", "life_delta")
        batch = self._batch(
            gate=[[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            fields={
                "tap_state": [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                "damage_taken": [[2.0, 0.0, 0.0], [3.0, 0.0, 1.0]],
                "zone_outcome": [[1.0, 0.0, 0.0], [2.0, 0.0, 1.0]],
                "life_delta": [[-2.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
            },
        )
        fields = tuple(FIELDS_BY_NAME[name] for name in names)

        floor = constant_predictor_floor([batch], fields=fields)
        constant = constant_predictor_outputs([batch], fields=fields)
        _, reported = per_entity_loss(
            constant.expand(2, 3, PER_ENTITY_WIDTH), batch.gate, batch.fields,
            batch.mask, fields=fields, report_parts=True,
        )
        parts = reported.as_dict()

        assert set(parts) == set(floor) == {"gate", *names}
        for name, value in parts.items():
            assert value == pytest.approx(floor[name], abs=1e-5)

    def test_a_head_that_predicts_the_targets_explains_nearly_all_of_it(self):
        values = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        batch = self._batch(gate=[[1.0] * 3] * 2, fields={"tap_state": values})
        fields = (FIELDS_BY_NAME["tap_state"],)
        floor = constant_predictor_floor([batch], fields=fields)

        outputs = torch.zeros(2, 3, PER_ENTITY_WIDTH)
        start, end = FIELD_SLICES["tap_state"]
        outputs[..., start:end] = (
            (torch.tensor(values) * 2 - 1) * 20
        ).unsqueeze(-1)
        _, parts = per_entity_loss(
            outputs, batch.gate, batch.fields, batch.mask, fields=fields,
            report_parts=True,
        )

        assert 1 - parts.as_dict()["tap_state"] / floor["tap_state"] > 0.99

    def test_the_floor_reads_the_gate_over_the_entities_the_mask_keeps(self):
        """A padded entity is not a base rate: it is not an entity at all.

        Two of the four real entities are affected, so the constant gate is a
        coin flip; counting the padding would make it one in three and hand the
        breakdown a floor no prediction was ever scored against.
        """
        batch = self._batch(
            gate=[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            fields={},
            mask=[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        )

        floor = constant_predictor_floor([batch], fields=())

        assert floor["gate"] == pytest.approx(4 * math.log(2) / 2, abs=1e-5)


class TestPartsMean:
    def test_each_field_is_averaged_over_the_batches_that_carried_it(self):
        """Summed on the device in float64 and read once, the means are the
        ones a float read of every part and a Python sum give."""
        batches = [
            LossParts(("gate", "tap_state"), torch.tensor([0.1, 0.7], dtype=torch.float64)),
            LossParts(("gate",), torch.tensor([0.3], dtype=torch.float64)),
            LossParts(),
            LossParts(("gate", "tap_state"), torch.tensor([0.2, 0.9], dtype=torch.float64)),
        ]
        means = PartsMean()
        for parts in batches:
            means.add(parts)

        assert means.means() == {
            "gate": (0.1 + 0.3 + 0.2) / 3, "tap_state": (0.7 + 0.9) / 2,
        }

    def test_nothing_added_reads_as_no_fields(self):
        assert PartsMean().means() == {}


class TestDegenerateFields:
    """A field whose targets never vary has no floor (FR-127c).

    The rate clamp leaves such a field a floor of about a millionth of a nat,
    and a head's arbitrary output against it reads as hundreds of thousands of
    percent. The breakdown exists to make exactly these rare fields legible, so
    the floor is zero and the term reports no deviance at all.
    """

    def _batch(self, gate, fields):
        gate = torch.tensor(gate, dtype=torch.float32)
        return EntityTargetBatch(
            gate=gate,
            fields=EntityFieldTargets.everywhere({
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in fields.items()
            }),
            mask=torch.ones_like(gate),
        )

    def _floor(self, name, values):
        batch = self._batch(gate=[[1.0] * len(values[0])] * len(values),
                            fields={name: values})
        return constant_predictor_floor(
            [batch], fields=(FIELDS_BY_NAME[name],),
        )[name]

    def test_a_count_field_nothing_ever_happens_to_has_no_floor(self):
        assert self._floor("damage_taken", [[0.0, 0.0], [0.0, 0.0]]) == 0.0

    def test_a_binary_field_with_no_positives_has_no_floor(self):
        assert self._floor("tap_state", [[0.0, 0.0], [0.0, 0.0]]) == 0.0

    def test_a_categorical_field_with_one_class_has_no_floor(self):
        assert self._floor("zone_outcome", [[0.0, 0.0], [0.0, 0.0]]) == 0.0

    def test_a_signed_delta_that_never_moves_has_no_floor(self):
        assert self._floor("life_delta", [[0.0, 0.0], [0.0, 0.0]]) == 0.0

    def test_one_nonzero_target_is_enough_to_make_a_floor(self):
        assert self._floor("damage_taken", [[0.0, 0.0], [0.0, 2.0]]) > 0.0

    def test_a_multi_binary_field_is_degenerate_only_when_its_rows_agree(self):
        """A bit that never fires still rides the clamp; the field does not."""
        fires = [1.0] + [0.0] * (len(COLORS) - 1)
        silent = [0.0] * len(COLORS)
        constant = [[fires, fires], [fires, fires]]
        varying = [[fires, silent], [fires, fires]]
        assert self._floor("colors_gained", constant) == 0.0
        assert self._floor("colors_gained", varying) > 0.0

    def test_the_gate_keeps_its_floor_whatever_the_fields_do(self):
        """The gate is supervised on every entity and always has a base rate."""
        batch = self._batch(
            gate=[[1.0, 0.0], [1.0, 0.0]],
            fields={"damage_taken": [[0.0, 0.0], [0.0, 0.0]]},
        )
        floor = constant_predictor_floor(
            [batch], fields=(FIELDS_BY_NAME["damage_taken"],),
        )
        assert floor["gate"] == pytest.approx(4 * math.log(2) / 2, abs=1e-5)
        assert floor["damage_taken"] == 0.0
