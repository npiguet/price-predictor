"""The heads' losses (T069).

A conditional field contributes nothing where the gate's target is off, and
every head's term reduces one way.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from effects.domain.effect_head_layout import (
    CREATED_OBJECT_SLOTS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    VERDICT_WIDTH,
    ZONE_OUTCOMES,
    FieldType,
    per_entity_fields_of_type,
)
from effects.domain.effect_losses import (
    api_loss,
    created_objects_loss,
    field_loss,
    mlm_loss,
    per_entity_loss,
    verdict_loss,
)
from effects.domain.entity_field_targets import EntityFieldTargets
from effects.domain.loss_floor import EntityTargetBatch, constant_predictor_outputs


class TestLosses:
    def _outputs(self, batch=2, entities=3):
        torch.manual_seed(1)
        return torch.randn(batch, entities, PER_ENTITY_WIDTH)

    def test_a_conditional_field_contributes_nothing_where_the_gate_is_off(self):
        outputs = self._outputs()
        entity_mask = torch.ones(2, 3)
        targets = EntityFieldTargets.everywhere({"damage_taken": torch.zeros(2, 3)})
        fields = (FIELDS_BY_NAME["damage_taken"],)
        off = per_entity_loss(
            outputs, torch.zeros(2, 3), targets, entity_mask, fields=fields,
            report_parts=True,
        )[1]
        on = per_entity_loss(
            outputs, torch.ones(2, 3), targets, entity_mask, fields=fields,
            report_parts=True,
        )[1]
        assert "damage_taken" not in off.names
        assert "damage_taken" in on.names

    def test_the_gate_trains_on_every_entity_including_unaffected_ones(self):
        _, parts = per_entity_loss(
            self._outputs(), torch.zeros(2, 3), {}, torch.ones(2, 3), fields=(),
            report_parts=True,
        )
        assert parts.as_dict()["gate"] > 0.0

    def test_the_decomposition_is_off_by_default(self):
        """The training loop discards the terms on every step but the one that
        logs them — so the caller has to ask."""
        _, parts = per_entity_loss(
            self._outputs(), torch.zeros(2, 3), {}, torch.ones(2, 3), fields=(),
        )
        assert not parts
        assert parts.as_dict() == {}

    def test_the_parts_are_each_term_read_back_on_its_own(self):
        """Stacked on the device and read once, each term is still the float a
        read of that term alone gives, divided by the record count — exactly,
        since the validation means sum these values over batches."""
        torch.manual_seed(4)
        outputs = torch.randn(2, 3, PER_ENTITY_WIDTH)
        fields = tuple(
            FIELDS_BY_NAME[name]
            for name in ("tap_state", "damage_taken", "life_delta", "zone_outcome")
        ) + per_entity_fields_of_type(FieldType.MULTI_BINARY)[:1]
        values = {
            spec.name: (
                torch.randint(0, 2, (2, 3, spec.arity)).float()
                if spec.type is FieldType.MULTI_BINARY
                else torch.randint(0, spec.arity, (2, 3)).float()
                if spec.type is FieldType.CATEGORICAL
                else torch.randint(-3, 5, (2, 3)).float()
            )
            for spec in fields
        }
        _, parts = per_entity_loss(
            outputs, torch.ones(2, 3), EntityFieldTargets.everywhere(values),
            torch.ones(2, 3), fields=fields, report_parts=True,
        )

        flat = outputs.reshape(6, PER_ENTITY_WIDTH)
        expected = {"gate": float(functional.binary_cross_entropy_with_logits(
            flat[:, GATE_INDEX], torch.ones(6), reduction="sum",
        )) / 2}
        for spec in fields:
            start, end = FIELD_SLICES[spec.name]
            target = values[spec.name]
            expected[spec.name] = float(field_loss(
                spec, flat[:, start:end], target.reshape(6, *target.shape[2:]),
            )) / 2
        assert parts.names == ("gate", *(spec.name for spec in fields))
        assert parts.as_dict() == expected

    def test_the_total_is_the_same_either_way(self):
        """Reporting must not change what is optimized."""
        outputs = self._outputs()
        quiet, _ = per_entity_loss(
            outputs, torch.ones(2, 3), {}, torch.ones(2, 3), fields=(),
        )
        loud, _ = per_entity_loss(
            outputs, torch.ones(2, 3), {}, torch.ones(2, 3), fields=(),
            report_parts=True,
        )
        assert float(quiet) == pytest.approx(float(loud))

    def test_the_loss_is_normalized_per_record_not_per_entity(self):
        """A twelve-permanent board must not outweigh a two-permanent one."""
        torch.manual_seed(2)
        wide = torch.zeros(1, 12, PER_ENTITY_WIDTH)
        narrow = torch.zeros(1, 2, PER_ENTITY_WIDTH)
        wide_loss, _ = per_entity_loss(
            wide, torch.zeros(1, 12), {}, torch.ones(1, 12), fields=(),
        )
        narrow_loss, _ = per_entity_loss(
            narrow, torch.zeros(1, 2), {}, torch.ones(1, 2), fields=(),
        )
        # Both divide by one record; the wide board's larger sum is the point,
        # but doubling the batch must halve neither.
        two_records = torch.zeros(2, 2, PER_ENTITY_WIDTH)
        batched, _ = per_entity_loss(
            two_records, torch.zeros(2, 2), {}, torch.ones(2, 2), fields=(),
        )
        assert batched == pytest.approx(float(narrow_loss), rel=1e-5)
        assert wide_loss > narrow_loss

    def test_the_reported_parts_sum_to_the_printed_loss(self):
        """The breakdown is of the number beside it, not of an unscaled sum.

        Each term was reported raw while the total was divided by the record
        count, so a two-record batch printed fields summing to twice its loss.
        """
        outputs = self._outputs()
        targets = EntityFieldTargets.everywhere({"damage_taken": torch.ones(2, 3)})
        fields = (FIELDS_BY_NAME["damage_taken"],)
        total, reported = per_entity_loss(
            outputs, torch.ones(2, 3), targets, torch.ones(2, 3),
            fields=fields, report_parts=True,
        )
        parts = reported.as_dict()
        assert set(parts) == {"gate", "damage_taken"}
        assert sum(parts.values()) == pytest.approx(float(total), abs=1e-4)

    def test_padded_entities_contribute_nothing(self):
        outputs = self._outputs()
        full = per_entity_loss(
            outputs, torch.ones(2, 3), {}, torch.ones(2, 3), fields=(),
            report_parts=True,
        )[1].as_dict()["gate"]
        masked = per_entity_loss(
            outputs, torch.ones(2, 3), {},
            torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]), fields=(),
            report_parts=True,
        )[1].as_dict()["gate"]
        assert masked < full

    def test_a_count_field_uses_a_poisson_loss(self):
        """Poisson on a log-rate: the loss is lowest where exp(pred) == target.

        Compared across *predictions*, not across targets — Poisson NLL drops
        the constant log(target!) term, so two targets at one prediction are
        not comparable and only the prediction's fit is.
        """
        spec = FIELDS_BY_NAME["damage_taken"]
        target = torch.full((4,), 3.0)
        right = field_loss(spec, torch.full((4, 1), 1.0986), target)  # log 3
        too_low = field_loss(spec, torch.zeros(4, 1), target)
        too_high = field_loss(spec, torch.full((4, 1), 2.5), target)
        assert right < too_low
        assert right < too_high

    def test_the_count_loss_is_nonnegative_at_its_optimum(self):
        """`full=True` adds the Stirling term, so a perfect prediction scores ~0
        instead of k - k ln k, which is -306 for a target of 88 (FR-081)."""
        from effects.domain.effect_losses import _poisson
        target = torch.tensor([88.0, 3.0, 1.0])
        loss = _poisson(torch.log(target), target)
        assert loss.item() >= 0.0
        assert loss.item() < 6.0   # Stirling residual is ~0.5*log(2*pi*k) per element

    def test_a_signed_delta_separates_direction_from_magnitude(self):
        spec = FIELDS_BY_NAME["life_delta"]
        prediction = torch.zeros(4, 4)
        gain = field_loss(spec, prediction, torch.full((4,), 3.0))
        loss = field_loss(spec, prediction, torch.full((4,), -3.0))
        # Same magnitude, opposite direction: the magnitude term matches and
        # only the direction term can differ, so both are finite and positive.
        assert gain > 0 and loss > 0

    def test_a_signed_delta_is_cheapest_when_direction_and_size_are_right(self):
        spec = FIELDS_BY_NAME["life_delta"]
        confident_positive = torch.tensor([[-5.0, -5.0, 5.0, 1.0986]])
        right = field_loss(spec, confident_positive, torch.tensor([3.0]))
        wrong = field_loss(spec, confident_positive, torch.tensor([-3.0]))
        assert right < wrong

    def test_a_categorical_field_uses_cross_entropy(self):
        spec = FIELDS_BY_NAME["zone_outcome"]
        logits = torch.zeros(2, len(ZONE_OUTCOMES))
        logits[0, 1] = 10.0
        died = field_loss(spec, logits, torch.tensor([1, 1]))
        stayed = field_loss(spec, logits, torch.tensor([0, 0]))
        assert died < stayed

    def test_a_binary_field_uses_bce(self):
        spec = FIELDS_BY_NAME["tap_state"]
        assert field_loss(spec, torch.full((3, 1), 10.0), torch.ones(3)) < 0.01

    def test_a_multi_binary_field_scores_every_member(self):
        spec = FIELDS_BY_NAME["keywords_gained"]
        prediction = torch.full((2, spec.arity), -10.0)
        none_gained = field_loss(spec, prediction, torch.zeros(2, spec.arity))
        one_gained = field_loss(
            spec, prediction,
            torch.nn.functional.one_hot(torch.tensor([0, 0]), spec.arity),
        )
        assert one_gained > none_gained

    def test_created_objects_loss_scores_the_overflow_flag(self):
        target = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        target[:, -1] = 1.0  # more than K distinct groups
        predicts_overflow = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        predicts_overflow[:, -1] = 8.0
        denies_overflow = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        denies_overflow[:, -1] = -8.0
        assert created_objects_loss(predicts_overflow, target) < (
            created_objects_loss(denies_overflow, target)
        )

    def test_created_objects_loss_scores_every_slot(self):
        """A slot the head gets wrong costs, wherever in the K it sits."""
        target = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        baseline = created_objects_loss(torch.zeros(2, CREATED_OBJECTS_WIDTH), target)
        for slot in range(CREATED_OBJECT_SLOTS):
            wrong = torch.zeros(2, CREATED_OBJECTS_WIDTH)
            wrong[:, slot * CREATED_SLOT_WIDTH] = 8.0  # claims a group is present
            assert created_objects_loss(wrong, target) > baseline

    def test_verdict_loss_skips_records_with_no_verdict(self):
        prediction = torch.zeros(2, VERDICT_WIDTH)
        target = torch.zeros(2, VERDICT_WIDTH)
        empty = verdict_loss(prediction, target, torch.zeros(2, dtype=torch.bool))
        assert float(empty) == 0.0

    def test_verdict_loss_scores_the_records_that_have_one(self):
        prediction = torch.zeros(2, VERDICT_WIDTH)
        target = torch.ones(2, VERDICT_WIDTH)
        scored = verdict_loss(prediction, target, torch.ones(2, dtype=torch.bool))
        assert float(scored) > 0.0

    def test_mlm_loss_is_low_where_the_logits_name_the_target(self):
        targets = torch.tensor([3, 7])
        right = torch.full((2, 32), -10.0)
        right[0, 3] = right[1, 7] = 10.0
        assert float(mlm_loss(right, targets, rows=1)) < 1e-3
        assert float(mlm_loss(torch.zeros(2, 32), targets, rows=1)) > 1.0

    def test_api_loss_accepts_either_target_alone(self):
        types = api_loss(torch.randn(2, 5), torch.tensor([0, 1]), None, None)
        keys = api_loss(None, None, torch.randn(2, 7), torch.zeros(2, 7))
        assert float(types) > 0.0 and float(keys) > 0.0

    def test_api_loss_needs_at_least_one_target(self):
        with pytest.raises(ValueError, match="at least one"):
            api_loss(None, None, None, None)


class TestGen2Losses:
    """The verdict, created-objects, value and script-API losses."""

    def test_the_verdict_loss_scores_only_the_masked_columns(self):
        prediction = torch.zeros(2, VERDICT_WIDTH)
        target = torch.ones(2, VERDICT_WIDTH)
        mask = torch.zeros(2, VERDICT_WIDTH, dtype=torch.bool)
        mask[0, :3] = True
        bits_only = verdict_loss(prediction, target, mask)
        mask[1, -1] = True
        with_trigger = verdict_loss(prediction, target, mask)
        assert 0 < float(bits_only) < float(with_trigger)

    def test_the_created_objects_mask_drops_unsupervised_records(self):
        target = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        prediction = torch.zeros(2, CREATED_OBJECTS_WIDTH)
        prediction[1, 0] = 9.0  # the unsupervised row is badly wrong
        mask = torch.tensor([True, False])
        assert created_objects_loss(prediction, target, mask) < (
            created_objects_loss(prediction, target)
        )

    def test_the_value_loss_skips_masked_targets(self):
        from effects.domain.effect_head_layout import VALUE_WIDTH
        from effects.domain.effect_losses import value_loss
        from effects.domain.value_targets import VALUE_TARGETS

        prediction = torch.zeros(2, VALUE_WIDTH)
        target = torch.full((2, len(VALUE_TARGETS)), 3.0)
        none = torch.zeros(2, len(VALUE_TARGETS), dtype=torch.bool)
        assert float(value_loss(prediction, target, none)) == 0.0
        assert float(value_loss(prediction, target, ~none)) > 0.0

    def test_at_least_bits_read_the_magnitude(self):
        from effects.domain.effect_head_layout import AT_LEAST_STEPS
        from effects.domain.effect_losses import at_least_targets

        bits = at_least_targets(torch.tensor([0.0, 3.0, -2.0, 45.0]))
        assert bits.shape == (4, AT_LEAST_STEPS)
        assert bits[0].sum() == 0
        assert bits[1].tolist()[:4] == [1.0, 1.0, 1.0, 0.0]
        assert bits[2].sum() == 2  # a signed amount reads by its magnitude
        assert bits[3].sum() == AT_LEAST_STEPS

    def test_the_value_loss_scores_the_at_least_bits(self):
        from effects.domain.effect_head_layout import AT_LEAST_SLICES, VALUE_WIDTH
        from effects.domain.effect_losses import at_least_targets, value_loss
        from effects.domain.value_targets import VALUE_TARGET_NAMES

        damage = VALUE_TARGET_NAMES.index("damage")
        target = torch.zeros(1, len(VALUE_TARGET_NAMES))
        target[0, damage] = 3.0
        mask = torch.zeros(1, len(VALUE_TARGET_NAMES), dtype=torch.bool)
        mask[0, damage] = True
        start, end = AT_LEAST_SLICES["damage"]
        right = torch.zeros(1, VALUE_WIDTH)
        right[0, start:end] = 20.0 * (2 * at_least_targets(torch.tensor([3.0]))[0] - 1)
        wrong = right.clone()
        wrong[0, start:end] = -wrong[0, start:end]
        assert float(value_loss(right, target, mask)) < float(
            value_loss(wrong, target, mask)
        )

    def test_the_api_loss_skips_a_line_with_no_api_type(self):
        logits = torch.randn(2, 5)
        alone = api_loss(logits, torch.tensor([-1, -1]), None, None)
        assert float(alone) == 0.0
        assert float(api_loss(logits, torch.tensor([-1, 2]), None, None)) > 0.0


class TestHostPickedRowsMatchBooleanMasks:
    """The per-entity, verdict and created-objects losses pick their rows on the
    host and gather by index, so the step never waits on the device. That must
    be exactly the loss the boolean-mask selection gave."""

    def _targets(self, generator, batch, entities):
        from effects.domain.effect_head_layout import FIELD_SLICES, PER_ENTITY_FIELDS, FieldType

        out = {}
        for spec in PER_ENTITY_FIELDS:
            start, end = FIELD_SLICES[spec.name]
            width = end - start
            match spec.type:
                case FieldType.BINARY:
                    out[spec.name] = torch.randint(0, 2, (batch, entities), generator=generator)
                case FieldType.MULTI_BINARY:
                    out[spec.name] = torch.randint(
                        0, 2, (batch, entities, width), generator=generator)
                case FieldType.COUNT:
                    out[spec.name] = torch.randint(0, 5, (batch, entities), generator=generator)
                case FieldType.SIGNED_DELTA:
                    out[spec.name] = torch.randint(-3, 4, (batch, entities), generator=generator)
                case FieldType.CATEGORICAL:
                    out[spec.name] = torch.randint(
                        0, spec.arity, (batch, entities), generator=generator)
        return PER_ENTITY_FIELDS, EntityFieldTargets.everywhere(out)

    def _reference(self, outputs, gate_target, field_targets, entity_mask, fields):
        from effects.domain.effect_head_layout import FIELD_SLICES
        from effects.domain.effect_losses import field_loss

        mask = entity_mask.bool()
        total = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs[..., GATE_INDEX][mask], gate_target[mask].float(), reduction="sum",
        )
        affected = mask & gate_target.bool()
        for spec in fields:
            start, end = FIELD_SLICES[spec.name]
            if affected.any():
                total = total + field_loss(
                    spec, outputs[..., start:end][affected],
                    field_targets.values[spec.name][affected],
                )
        return total / outputs.shape[0]

    def test_the_per_entity_loss_is_unchanged(self):
        generator = torch.Generator().manual_seed(11)
        batch, entities = 6, 9
        outputs = torch.randn(batch, entities, PER_ENTITY_WIDTH, generator=generator)
        entity_mask = torch.randint(0, 2, (batch, entities), generator=generator)
        gate = torch.randint(0, 2, (batch, entities), generator=generator)
        fields, targets = self._targets(generator, batch, entities)
        expected = self._reference(outputs, gate, targets, entity_mask, fields)
        got, _ = per_entity_loss(outputs, gate, targets, entity_mask, fields=fields)
        assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_a_batch_with_nothing_affected_scores_the_gate_alone(self):
        generator = torch.Generator().manual_seed(12)
        outputs = torch.randn(3, 4, PER_ENTITY_WIDTH, generator=generator)
        fields, targets = self._targets(generator, 3, 4)
        gate = torch.zeros(3, 4)
        got, _ = per_entity_loss(outputs, gate, targets, torch.ones(3, 4), fields=fields)
        expected = self._reference(outputs, gate, targets, torch.ones(3, 4), fields)
        assert torch.allclose(got, expected)

    def test_created_objects_and_verdict_losses_are_unchanged(self):
        from effects.domain.effect_head_layout import CREATED_OBJECTS_WIDTH, VERDICT_WIDTH
        from effects.domain.effect_losses import created_objects_loss, verdict_loss

        generator = torch.Generator().manual_seed(13)
        prediction = torch.randn(5, CREATED_OBJECTS_WIDTH, generator=generator)
        target = torch.rand(5, CREATED_OBJECTS_WIDTH, generator=generator).round()
        mask = torch.tensor([True, False, True, True, False])
        picked = created_objects_loss(prediction, target, mask)
        # Selecting the rows first and scoring them unmasked is the same sum.
        whole = created_objects_loss(prediction[mask], target[mask], None)
        assert torch.allclose(picked, whole * mask.sum() / mask.numel())

        verdict = torch.randn(4, VERDICT_WIDTH, generator=generator)
        verdict_target = torch.rand(4, VERDICT_WIDTH, generator=generator).round()
        per_output = torch.rand(4, VERDICT_WIDTH, generator=generator) > 0.5
        assert torch.allclose(
            verdict_loss(verdict, verdict_target, per_output),
            verdict_loss(verdict, verdict_target, per_output.float()),
        )
        assert float(verdict_loss(verdict, verdict_target, torch.zeros(4, VERDICT_WIDTH))) == 0.0


class TestMagnitudeCap:
    """A magnitude above the cap scores as the cap.

    Above about forty, an amount is the game state compounding — a counter
    doubling run to two million — not what the ability's text says, and one
    such target under a Poisson loss pulls a step a million times harder than
    an ordinary one.
    """

    def test_a_count_above_the_cap_scores_as_the_cap(self):
        from effects.domain.effect_head_input import MAGNITUDE_CAP
        spec = FIELDS_BY_NAME["damage_taken"]
        prediction = torch.zeros(1, 1)
        assert field_loss(spec, prediction, torch.tensor([2_000_000.0])) == \
            field_loss(spec, prediction, torch.tensor([MAGNITUDE_CAP]))

    def test_a_signed_delta_keeps_its_direction_and_caps_its_size(self):
        from effects.domain.effect_head_input import MAGNITUDE_CAP
        spec = FIELDS_BY_NAME["power_delta"]
        prediction = torch.zeros(1, 4)
        for sign in (1.0, -1.0):
            assert field_loss(spec, prediction, torch.tensor([sign * 2_098_760.0])) == \
                field_loss(spec, prediction, torch.tensor([sign * MAGNITUDE_CAP]))

    def test_the_floor_is_computed_over_capped_targets(self):
        from effects.domain.effect_head_input import MAGNITUDE_CAP

        def floor(value):
            batch = EntityTargetBatch(
                gate=torch.ones(1, 2),
                fields=EntityFieldTargets.everywhere(
                    {"damage_taken": torch.tensor([[2.0, value]])},
                ),
                mask=torch.ones(1, 2),
            )
            return constant_predictor_outputs(
                [batch], fields=(FIELDS_BY_NAME["damage_taken"],),
            )

        assert torch.equal(floor(2_000_000.0), floor(MAGNITUDE_CAP))

    def test_the_evaluation_deviance_caps_the_observed_amount(self):
        import numpy as np

        from effects.domain.effect_head_input import MAGNITUDE_CAP
        from effects.domain.gate_metrics import poisson_deviance
        predicted = np.array([3.0, 5.0])
        assert poisson_deviance(predicted, np.array([1.0, 2_000_000.0])) == \
            poisson_deviance(predicted, np.array([1.0, MAGNITUDE_CAP]))


class TestCountTargetsAreNonnegative:
    """A count below zero has no Poisson likelihood (FR-080, FR-081).

    ``damage_taken`` is decremented by a healing event, so a heal with no
    damage before it is a negative count, and a negative target makes the
    Poisson NLL unbounded below: the head could lower the loss without limit by
    raising its rate.
    """

    def test_a_negative_count_scores_as_zero(self):
        spec = FIELDS_BY_NAME["damage_taken"]
        for log_rate in (-2.0, 0.0, 3.0):
            prediction = torch.full((1, 1), log_rate)
            assert field_loss(spec, prediction, torch.tensor([-3.0])) == \
                field_loss(spec, prediction, torch.tensor([0.0]))

    def test_the_floor_reads_a_negative_count_as_zero(self):
        def floor(value):
            batch = EntityTargetBatch(
                gate=torch.ones(1, 2),
                fields=EntityFieldTargets.everywhere(
                    {"damage_taken": torch.tensor([[2.0, value]])},
                ),
                mask=torch.ones(1, 2),
            )
            return constant_predictor_outputs(
                [batch], fields=(FIELDS_BY_NAME["damage_taken"],),
            )

        assert torch.equal(floor(-5.0), floor(0.0))


class TestOneReductionPolicy:
    """Every head's term sums each row's targets and averages over rows.

    A row is the head's unit — a record for the per-entity, created-objects and
    verdict heads, an encoded text for the value, script-API and MLM heads — so
    a head's weight flag scales the same kind of number whichever head it is.
    """

    def test_the_mlm_term_sums_a_texts_masked_tokens_and_averages_over_texts(self):
        logits = torch.zeros(6, 32)
        targets = torch.zeros(6, dtype=torch.long)
        per_token = math.log(32)
        # Six masked tokens over three texts: two tokens' loss per text.
        assert float(mlm_loss(logits, targets, rows=3)) == pytest.approx(2 * per_token)

    def test_the_api_keys_term_sums_a_texts_keys(self):
        keys = 1245
        loss = api_loss(None, None, torch.zeros(4, keys), torch.zeros(4, keys))
        assert float(loss) == pytest.approx(keys * math.log(2), rel=1e-5)

    def test_the_api_types_term_averages_over_every_text(self):
        """A text with no API type is a row with nothing to score, not one left out."""
        logits = torch.zeros(4, 5)
        loss = api_loss(logits, torch.tensor([0, -1, -1, -1]), None, None)
        assert float(loss) == pytest.approx(math.log(5) / 4, rel=1e-5)
