"""The per-entity loss against the field-by-field reference.

The losses pick their rows on the host; that must equal the boolean-mask loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from effects.domain.effect_head_layout import (
    CREATED_DISPERSED,
    DISPERSED_FIELDS,
    DISPERSION_INDEX,
    FIELD_SLICES,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    VERDICT_LAYOUT,
    FieldSpec,
)
from effects.domain.effect_losses import EntityLossScoring, field_loss, per_entity_loss
from effects.domain.entity_field_targets import EntityFieldTargets
from effects.domain.record_head_losses import (
    CreatedTargets,
    HeadScoring,
    RecordHeadTargets,
)
from tests.unit.effects.batch_builders import fields_everywhere, make_entity_targets

CREATED = torch.zeros(len(CREATED_DISPERSED))
VERDICT = torch.zeros(len(VERDICT_LAYOUT.dispersed))


@dataclass(frozen=True)
class _Scene:
    """One batch the per-entity loss scores: outputs, targets and the entity mask."""

    outputs: torch.Tensor
    gate: torch.Tensor
    targets: EntityFieldTargets
    mask: torch.Tensor
    fields: tuple[FieldSpec, ...]


def _reference(scene: _Scene, dispersion: torch.Tensor) -> torch.Tensor:
    """The loss the field-by-field loop takes, one boolean mask per field."""
    outputs = scene.outputs
    mask = scene.mask.bool()
    total = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs[..., GATE_INDEX][mask], scene.gate[mask].float(), reduction="sum",
    )
    affected = mask & scene.gate.bool()
    for spec in scene.fields:
        start, end = FIELD_SLICES[spec.name]
        # A legality field trains gate or no gate (root spec § 11.4).
        where = affected if spec.gated else mask
        if where.any():
            total = total + field_loss(
                spec, outputs[..., start:end][where],
                scene.targets.values[spec.name][where],
                log_dispersion=(
                    dispersion[DISPERSION_INDEX[spec.name]] if spec.type.has_magnitude else None
                ),
            )
    return total / outputs.shape[0]


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
        return PER_ENTITY_FIELDS, fields_everywhere(out)

    #: A different log-dispersion per field, so a gather that read the wrong
    #: one would show.
    DISPERSION = torch.linspace(-2.0, 4.0, len(DISPERSED_FIELDS))

    def test_the_per_entity_loss_is_the_field_by_field_sum(self):
        """Same-kind fields are scored together; the loss is still exactly
        the sum the field-by-field loop takes."""
        generator = torch.Generator().manual_seed(11)
        batch, entities = 6, 9
        outputs = torch.randn(batch, entities, PER_ENTITY_WIDTH, generator=generator)
        entity_mask = torch.randint(0, 2, (batch, entities), generator=generator)
        gate = torch.randint(0, 2, (batch, entities), generator=generator)
        fields, targets = self._targets(generator, batch, entities)
        scene = _Scene(outputs, gate, targets, entity_mask, fields)
        expected = _reference(scene, self.DISPERSION)
        got, _ = per_entity_loss(
            outputs, make_entity_targets(gate, targets, entity_mask),
            EntityLossScoring(fields=fields, log_dispersion=self.DISPERSION),
        )
        assert torch.allclose(got, expected, rtol=1e-6, atol=1e-6)

    def test_each_field_s_part_is_its_own_loss_with_presence_varying(self):
        """Every field present at its own entities, so each group scores the
        union and weighs each field only where it is present."""
        generator = torch.Generator().manual_seed(21)
        batch, entities = 5, 7
        outputs = torch.randn(batch, entities, PER_ENTITY_WIDTH, generator=generator)
        gate = torch.randint(0, 2, (batch, entities), generator=generator)
        fields, everywhere = self._targets(generator, batch, entities)
        present = {
            name: torch.rand(batch, entities, generator=generator) < 0.4
            for name in everywhere.values
        }
        targets = make_entity_targets(gate, EntityFieldTargets(everywhere.values, present))
        _, parts = per_entity_loss(
            outputs, targets,
            EntityLossScoring(fields=fields, log_dispersion=self.DISPERSION, report_parts=True),
        )
        expected = {}
        for spec in fields:
            where = targets.scored(spec.name)
            if not where.any():
                continue
            start, end = FIELD_SLICES[spec.name]
            expected[spec.name] = float(field_loss(
                spec, outputs[..., start:end][where], everywhere.values[spec.name][where],
                log_dispersion=(
                    self.DISPERSION[DISPERSION_INDEX[spec.name]]
                    if spec.type.has_magnitude else None
                ),
            )) / batch
        got = parts.as_dict()
        got.pop("gate")
        assert list(got) == list(expected)
        assert got == pytest.approx(expected, rel=1e-5, abs=1e-6)

    def test_the_gradient_is_the_field_by_field_gradient(self):
        generator = torch.Generator().manual_seed(17)
        batch, entities = 4, 6
        outputs = torch.randn(batch, entities, PER_ENTITY_WIDTH, generator=generator)
        gate = torch.randint(0, 2, (batch, entities), generator=generator)
        fields, targets = self._targets(generator, batch, entities)
        mask = torch.ones(batch, entities)

        def gradients(loss_of):
            leaf = outputs.clone().requires_grad_(True)
            dispersion = self.DISPERSION.clone().requires_grad_(True)
            loss_of(leaf, dispersion).backward()
            return leaf.grad, dispersion.grad

        grouped = gradients(lambda leaf, dispersion: per_entity_loss(
            leaf, make_entity_targets(gate, targets, mask),
            EntityLossScoring(fields=fields, log_dispersion=dispersion),
        )[0])

        def looped(leaf, dispersion):
            return _reference(_Scene(leaf, gate, targets, mask, fields), dispersion)

        reference = gradients(looped)
        for got, expected in zip(grouped, reference):
            assert torch.allclose(got, expected, rtol=1e-5, atol=1e-7)

    def test_a_batch_with_nothing_affected_scores_the_gate_alone(self):
        generator = torch.Generator().manual_seed(12)
        outputs = torch.randn(3, 4, PER_ENTITY_WIDTH, generator=generator)
        fields, targets = self._targets(generator, 3, 4)
        gate = torch.zeros(3, 4)
        got, _ = per_entity_loss(
            outputs, make_entity_targets(gate, targets),
            EntityLossScoring(fields=fields, log_dispersion=self.DISPERSION),
        )
        scene = _Scene(outputs, gate, targets, torch.ones(3, 4), fields)
        expected = _reference(scene, self.DISPERSION)
        assert torch.allclose(got, expected)

    def test_created_objects_and_verdict_losses_are_unchanged(self):
        from effects.domain.effect_head_layout import CREATED_OBJECTS_WIDTH, VERDICT_WIDTH
        from effects.domain.record_head_losses import created_objects_loss, verdict_loss

        generator = torch.Generator().manual_seed(13)
        prediction = torch.randn(5, CREATED_OBJECTS_WIDTH, generator=generator)
        target = torch.rand(5, CREATED_OBJECTS_WIDTH, generator=generator).round()
        mask = torch.tensor([True, False, True, True, False])
        picked = created_objects_loss(
            prediction, CreatedTargets(target, mask), HeadScoring(log_dispersion=CREATED),
        )
        # Selecting the rows first and scoring them unmasked is the same sum.
        whole = created_objects_loss(
            prediction[mask],
            CreatedTargets(target[mask], None),
            HeadScoring(log_dispersion=CREATED),
        )
        assert torch.allclose(picked, whole * mask.sum() / mask.numel())

        from effects.domain.effect_head_layout import VERDICT_LAYOUT

        fields = len(VERDICT_LAYOUT.fields)
        verdict = torch.randn(4, VERDICT_WIDTH, generator=generator)
        verdict_target = torch.rand(4, VERDICT_LAYOUT.target_width, generator=generator).round()
        per_field = torch.rand(4, fields, generator=generator) > 0.5
        assert torch.allclose(
            verdict_loss(
                verdict,
                RecordHeadTargets(verdict_target, per_field),
                HeadScoring(log_dispersion=VERDICT),
            ),
            verdict_loss(
                verdict,
                RecordHeadTargets(verdict_target, per_field.float()),
                HeadScoring(log_dispersion=VERDICT),
            ),
        )
        assert float(verdict_loss(
            verdict,
            RecordHeadTargets(verdict_target, torch.zeros(4, fields)),
            HeadScoring(log_dispersion=VERDICT),
        )) == 0.0
