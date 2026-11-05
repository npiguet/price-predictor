"""The constant-predictor floor a field's loss is compared against (FR-127c)."""

from __future__ import annotations

import math

import pytest
import torch

from effects.domain.effect_head_layout import FIELD_SLICES, FIELDS_BY_NAME, PER_ENTITY_WIDTH
from effects.domain.effect_losses import LossParts, per_entity_loss
from effects.domain.field_kinds import fit_log_dispersion, negative_binomial
from effects.domain.loss_floor import (
    PartsMean,
    constant_predictor_dispersion,
    constant_predictor_floor,
    constant_predictor_outputs,
)
from effects.domain.state_snapshot import COLORS
from tests.unit.effects.builders import fields_everywhere, make_entity_targets


class TestConstantPredictorFloor:
    """What a field's loss is compared against (FR-127c).

    A loss alone cannot say whether the head learned the field or is repeating
    its base rate, so every card-disjoint term is scaled by the loss the best
    constant prediction reaches on the same targets under the same masking.
    """

    def _batch(self, gate, fields, mask=None):
        gate = torch.tensor(gate, dtype=torch.float32)
        return make_entity_targets(
            gate=gate,
            fields=fields_everywhere({
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

    def test_a_count_fields_floor_is_the_negative_binomial_nll_of_its_fit(self):
        values = [[0.0, 1.0, 2.0, 9.0], [4.0, 0.0, 1.0, 7.0]]
        batch = self._batch(gate=[[1.0] * 4] * 2, fields={"damage_taken": values})

        floor = constant_predictor_floor(
            [batch], fields=(FIELDS_BY_NAME["damage_taken"],),
        )

        target = torch.tensor(values).reshape(-1)
        mean = target.mean()
        expected = negative_binomial(
            torch.full_like(target, math.log(float(mean))), target,
            fit_log_dispersion(target, mean),
        ).sum() / 2
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
            constant.expand(2, 3, PER_ENTITY_WIDTH), batch, fields=fields,
            log_dispersion=constant_predictor_dispersion([batch], fields=fields),
            report_parts=True,
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
            outputs, batch, fields=fields,
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
    def test_each_term_is_per_record_over_every_batch_added(self):
        """A term a batch does not carry counts zero there, and each batch
        weighs its records: the unit of a step's loss term and of the floor."""
        batches = [
            (LossParts(("gate", "tap_state"), torch.tensor([0.1, 0.7], dtype=torch.float64)), 2),
            (LossParts(("gate",), torch.tensor([0.3], dtype=torch.float64)), 4),
            (LossParts(), 1),
            (LossParts(("gate", "tap_state"), torch.tensor([0.2, 0.9], dtype=torch.float64)), 3),
        ]
        means = PartsMean()
        for parts, rows in batches:
            means.add(parts, rows=rows)

        assert means.means() == pytest.approx({
            "gate": (0.1 * 2 + 0.3 * 4 + 0.2 * 3) / 10, "tap_state": (0.7 * 2 + 0.9 * 3) / 10,
        })

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
        return make_entity_targets(
            gate=gate,
            fields=fields_everywhere({
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


def _verdict_batch(rows: int, values: dict, supervised: dict):
    """A verdict head's ``(target, mask)`` with ``values[name]`` per record
    (a list of member values for a member field) where ``supervised[name]``."""
    from effects.domain.effect_head_layout import VERDICT_LAYOUT

    target = torch.zeros(rows, VERDICT_LAYOUT.target_width)
    mask = torch.zeros(rows, len(VERDICT_LAYOUT.fields), dtype=torch.bool)
    for name, per_record in values.items():
        first, last = VERDICT_LAYOUT.target_slices[name]
        target[:, first:last] = torch.tensor(per_record, dtype=torch.float32).reshape(rows, -1)
        mask[:, VERDICT_LAYOUT.index[name]] = torch.tensor(supervised[name])
    return target, mask


def _sample(entity_records: int, heads: dict):
    from effects.domain.loss_floor import TargetSample

    sample = TargetSample()
    sample.add(make_entity_targets(torch.zeros(entity_records, 1)), heads)
    return sample


class TestEveryHead:
    """One function gives every shipped head's floor (T143): the verdict and
    global heads by field, the created-objects head by slot column."""

    def test_a_verdict_bit_s_floor_is_its_entropy_per_record_of_the_sample(self):
        """Two of four records set the bit, one of them positive: the floor
        is the two records' entropy spread over all four."""
        from effects.domain.loss_floor import constant_predictor_losses

        target, mask = _verdict_batch(
            4, {"can_play": [1.0, 0.0, 0.0, 0.0]},
            {"can_play": [True, True, False, False]},
        )
        losses = constant_predictor_losses(_sample(4, {"verdict": (target, mask)}), fields=())
        assert set(losses["verdict"]) == {"can_play"}
        assert losses["verdict"]["can_play"] == pytest.approx(2 * math.log(2) / 4, abs=1e-5)

    def test_a_count_member_field_fits_each_member_and_one_dispersion(self):
        """``mana_paid`` holds a count per colour: each colour keeps its own
        mean, all of them one fitted dispersion."""
        from effects.domain.effect_head_layout import VERDICT_LAYOUT
        from effects.domain.effect_losses import record_head_terms
        from effects.domain.loss_floor import constant_predictor_losses

        generator = torch.Generator().manual_seed(9)
        members = VERDICT_LAYOUT.by_name["mana_paid"].count
        paid = torch.poisson(
            torch.linspace(0.5, 4.0, members).expand(200, members), generator,
        ) * torch.randint(0, 3, (200, 1), generator=generator)
        target, mask = _verdict_batch(
            200, {"mana_paid": paid.tolist()}, {"mana_paid": [True] * 200},
        )
        floor = constant_predictor_losses(
            _sample(200, {"verdict": (target, mask)}), fields=(),
        )["verdict"]["mana_paid"]

        start, _ = VERDICT_LAYOUT.slices["mana_paid"]
        index = VERDICT_LAYOUT.index["mana_paid"]

        def scored(log_rates, log_dispersion):
            outputs = torch.zeros(200, VERDICT_LAYOUT.width)
            outputs[:, start:start + members] = log_rates
            dispersion = torch.zeros(len(VERDICT_LAYOUT.dispersed))
            dispersion[VERDICT_LAYOUT.dispersion_index["mana_paid"]] = log_dispersion
            return float(record_head_terms(
                outputs, target, mask, VERDICT_LAYOUT, log_dispersion=dispersion,
            )[index]) / 200

        means = torch.log(paid.mean(0))
        fitted = fit_log_dispersion(paid, paid.mean(0))
        assert floor == pytest.approx(scored(means, fitted), rel=1e-5)
        # The fit is the best constant: moving the means or the dispersion costs.
        assert floor < scored(means + 0.1, fitted)
        assert floor < scored(means, fitted + 0.2)
        assert floor < scored(means, fitted - 0.2)
        # Overdispersed by construction, so it fits well below the Poisson bound.
        assert float(fitted) < 2.0

    def test_a_global_count_and_categorical_each_get_a_floor(self):
        from effects.domain.effect_head_layout import GLOBAL_LAYOUT
        from effects.domain.loss_floor import constant_predictor_losses

        target = torch.zeros(3, GLOBAL_LAYOUT.target_width)
        first, _ = GLOBAL_LAYOUT.target_slices["turn_structure_change"]
        target[:, first] = torch.tensor([1.0, 0.0, 2.0])
        column, _ = GLOBAL_LAYOUT.target_slices["day_night_change"]
        target[:, column] = torch.tensor([0.0, 1.0, 2.0])
        mask = torch.ones(3, len(GLOBAL_LAYOUT.fields), dtype=torch.bool)
        losses = constant_predictor_losses(
            _sample(3, {"global": (target, mask)}), fields=(),
        )["global"]
        assert set(losses) == {"turn_structure_change", "day_night_change"}
        assert losses["day_night_change"] == pytest.approx(math.log(3), abs=1e-5)
        assert losses["turn_structure_change"] > 0.0

    def test_the_created_objects_head_reports_each_slot_column(self):
        from effects.domain.effect_head_layout import (
            CREATED_OBJECTS_WIDTH,
            CREATED_SLOT_WIDTH,
            CREATED_TERMS,
            CreatedSlot,
        )
        from effects.domain.loss_floor import constant_predictor_losses

        target = torch.zeros(4, CREATED_OBJECTS_WIDTH)
        target[:2, CreatedSlot.PRESENT] = 1.0
        target[:2, CreatedSlot.COUNT] = torch.tensor([1.0, 3.0])
        target[0, CREATED_SLOT_WIDTH + CreatedSlot.PRESENT] = 1.0
        target[0, CREATED_SLOT_WIDTH + CreatedSlot.COUNT] = 2.0
        known = torch.ones_like(target)
        known[:, CreatedSlot.OTHER_CONTROLLER] = 0.0  # said by no record
        mask = torch.tensor([True, True, True, False])
        losses = constant_predictor_losses(
            _sample(4, {"created_objects": (target, mask, known)}), fields=(),
        )["created_objects"]
        assert set(losses) <= set(CREATED_TERMS)
        assert losses["present"] > 0.0 and losses["count"] > 0.0
        # Unsaid in the first slot only: the later slots still give it a floor.
        assert "other_controller" in losses
        assert losses["scripted"] == 0.0  # never set: degenerate

    def test_a_head_no_record_supervises_is_absent(self):
        from effects.domain.loss_floor import PER_ENTITY, constant_predictor_losses

        losses = constant_predictor_losses(_sample(2, {}), fields=())
        assert set(losses) == {PER_ENTITY}

    def test_a_term_reads_per_record_over_every_batch(self):
        """A batch that never supervises a field still counts its records:
        the floor is in the unit of a step's loss term."""
        from effects.domain.loss_floor import TargetSample, constant_predictor_losses

        values = [[1.0, 0.0], [0.0, 0.0]]
        carrying = _tap_batch(values)
        silent = make_entity_targets(torch.ones(2, 2))
        alone = constant_predictor_losses(TargetSample.of([carrying]), fields=_TAP)
        both = constant_predictor_losses(TargetSample.of([carrying, silent]), fields=_TAP)
        assert both["per_entity"]["tap_state"] == pytest.approx(
            alone["per_entity"]["tap_state"] / 2, rel=1e-6,
        )


_TAP = (FIELDS_BY_NAME["tap_state"],)


def _tap_batch(values):
    gate = torch.ones(len(values), len(values[0]))
    return make_entity_targets(
        gate, fields_everywhere({"tap_state": torch.tensor(values)}),
    )


class TestNegativeBinomialFloor:
    def test_newton_steps_improve_on_the_moment_estimate(self):
        """The floor is the likelihood's best constant, not the moment fit's."""
        from effects.domain.field_kinds import negative_binomial

        generator = torch.Generator().manual_seed(2)
        counts = torch.poisson(torch.distributions.Gamma(0.7, 0.7 / 3.0).sample((300,)), generator)
        mean = counts.mean()
        moment = torch.log(mean ** 2 / (counts.var(unbiased=False) - mean))

        def nll(log_dispersion):
            return float(negative_binomial(
                torch.log(mean).expand_as(counts), counts, log_dispersion,
            ).sum())

        assert nll(fit_log_dispersion(counts, mean)) <= nll(moment)

    def test_a_count_fields_dispersion_is_fitted_per_field(self):
        from effects.domain.effect_head_layout import DISPERSION_INDEX

        wide = [[0.0, 0.0, 0.0, 12.0], [0.0, 1.0, 0.0, 9.0]]
        narrow = [[2.0, 2.0, 3.0, 2.0], [2.0, 1.0, 2.0, 2.0]]
        batch = make_entity_targets(
            torch.ones(2, 4),
            fields_everywhere({
                "damage_taken": torch.tensor(wide), "cards_drawn": torch.tensor(narrow),
            }),
        )
        fields = (FIELDS_BY_NAME["damage_taken"], FIELDS_BY_NAME["cards_drawn"])
        fitted = constant_predictor_dispersion([batch], fields=fields)
        assert fitted[DISPERSION_INDEX["damage_taken"]] < fitted[DISPERSION_INDEX["cards_drawn"]]
