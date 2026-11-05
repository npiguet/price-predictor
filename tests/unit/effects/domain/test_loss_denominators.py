"""Each term's loss divided by its floored constant-prediction loss (FR-062, T144)."""

from __future__ import annotations

import pytest
import torch

from effects.domain.aux_heads import Head
from effects.domain.effect_head_layout import (
    CREATED_DISPERSED,
    CREATED_OBJECTS_WIDTH,
    CREATED_TERMS,
    DISPERSED_FIELDS,
    FIELDS_BY_NAME,
    PER_ENTITY_TERMS,
    PER_ENTITY_WIDTH,
    VERDICT_LAYOUT,
    VERDICT_WIDTH,
)
from effects.domain.effect_losses import (
    created_objects_loss,
    created_objects_terms,
    per_entity_loss,
    record_head_terms,
    verdict_loss,
)
from effects.domain.loss_denominators import HEAD_TERMS, HeadScales, LossDenominators
from effects.domain.loss_floor import PER_ENTITY
from tests.unit.effects.builders import make_entity_targets

FLOOR = 0.04


class TestDenominators:
    def test_a_field_at_a_hundredth_of_a_nat_is_divided_by_the_floor(self):
        """Story 7, acceptance 3."""
        denominators = LossDenominators.of(
            {PER_ENTITY: {"damage_taken": 0.01, "gate": 0.7}}, floor=FLOOR,
        )
        assert denominators.of_term(PER_ENTITY, "damage_taken") == FLOOR
        assert denominators.of_term(PER_ENTITY, "gate") == 0.7

    def test_a_field_at_a_hundredth_of_a_nat_trains_at_its_loss_over_the_floor(self):
        """Acceptance 3 where the loss is computed: the field's term is its
        nats divided by 0.04, the gate's by its own denominator."""
        torch.manual_seed(0)
        outputs = torch.randn(2, 3, PER_ENTITY_WIDTH)
        targets = make_entity_targets(
            torch.ones(2, 3), {"damage_taken": torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]])},
        )
        fields = (FIELDS_BY_NAME["damage_taken"],)
        dispersion = torch.zeros(len(DISPERSED_FIELDS))
        denominators = LossDenominators.of(
            {PER_ENTITY: {"damage_taken": 0.01, "gate": 0.7}}, floor=FLOOR,
        )
        scaled, _ = per_entity_loss(
            outputs, targets, fields=fields, log_dispersion=dispersion,
            scale=denominators.reciprocals(PER_ENTITY),
        )
        _, parts = per_entity_loss(
            outputs, targets, fields=fields, log_dispersion=dispersion, report_parts=True,
        )
        raw = parts.as_dict()
        assert float(scaled) == pytest.approx(
            raw["gate"] / 0.7 + raw["damage_taken"] / FLOOR, rel=1e-5,
        )

    def test_a_term_the_sample_never_supervised_divides_by_the_floor(self):
        denominators = LossDenominators.of({}, floor=FLOOR)
        assert denominators.of_term(Head.VERDICT.value, "mana_paid") == FLOOR
        assert torch.allclose(
            denominators.reciprocals(PER_ENTITY),
            torch.full((len(PER_ENTITY_TERMS),), 1 / FLOOR),
        )

    def test_a_floor_of_zero_or_less_is_refused(self):
        with pytest.raises(ValueError, match="deviance-floor"):
            LossDenominators.of({}, floor=0.0)

    def test_they_round_trip_through_a_plain_value(self):
        denominators = LossDenominators.of(
            {PER_ENTITY: {"gate": 0.5}, Head.VERDICT.value: {"can_play": 0.3}}, floor=FLOOR,
        )
        assert LossDenominators.from_dict(denominators.as_dict()) == denominators

    def test_each_head_s_reciprocals_follow_its_loss_s_term_order(self):
        denominators = LossDenominators.of(
            {Head.CREATED_OBJECTS.value: {"count": 0.5}}, floor=FLOOR,
        )
        reciprocals = denominators.reciprocals(Head.CREATED_OBJECTS.value)
        assert reciprocals.shape == (len(CREATED_TERMS),)
        assert float(reciprocals[CREATED_TERMS.index("count")]) == pytest.approx(2.0)
        assert set(HEAD_TERMS) == {PER_ENTITY, *(head.value for head in (
            Head.VERDICT, Head.CREATED_OBJECTS, Head.GLOBAL,
        ))}

    def test_the_scales_live_on_the_device_they_are_built_for(self):
        scales = HeadScales.of(LossDenominators.of({}, floor=FLOOR), torch.device("cpu"))
        assert scales.per_entity.shape == (len(PER_ENTITY_TERMS),)
        assert set(scales.heads) == {Head.VERDICT, Head.CREATED_OBJECTS, Head.GLOBAL}


class TestEveryShippedHeadScales:
    def test_the_verdict_head_divides_each_field_by_its_own(self):
        generator = torch.Generator().manual_seed(1)
        prediction = torch.randn(3, VERDICT_WIDTH, generator=generator)
        target = torch.rand(3, VERDICT_LAYOUT.target_width, generator=generator).round()
        mask = torch.ones(3, len(VERDICT_LAYOUT.fields), dtype=torch.bool)
        dispersion = torch.zeros(len(VERDICT_LAYOUT.dispersed))
        scale = torch.linspace(0.5, 3.0, len(VERDICT_LAYOUT.fields))
        terms = record_head_terms(
            prediction, target, mask, VERDICT_LAYOUT, log_dispersion=dispersion,
        )
        assert float(verdict_loss(
            prediction, target, mask, log_dispersion=dispersion, scale=scale,
        )) == pytest.approx(float((terms * scale).sum()) / 3, rel=1e-5)

    def test_the_created_objects_head_divides_each_column_by_its_own(self):
        generator = torch.Generator().manual_seed(2)
        prediction = torch.randn(2, CREATED_OBJECTS_WIDTH, generator=generator)
        target = torch.rand(2, CREATED_OBJECTS_WIDTH, generator=generator).round()
        dispersion = torch.zeros(len(CREATED_DISPERSED))
        scale = torch.rand(len(CREATED_TERMS), generator=generator)
        terms = created_objects_terms(prediction, target, log_dispersion=dispersion)
        assert float(created_objects_loss(
            prediction, target, log_dispersion=dispersion, scale=scale,
        )) == pytest.approx(float((terms * scale).sum()) / 2, rel=1e-5)
