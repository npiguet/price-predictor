"""The breakdowns' per-record losses (T104).

Each record is scored on its own, by the trainer's own loss over a batch of
one, so the breakdowns cannot drift from what training minimized.
"""

from __future__ import annotations

import pytest


class TestPerRecordLosses:
    """``score_records``' per-record rows sum the trainer's own loss terms (T104)."""

    def test_a_record_s_total_is_the_per_entity_loss_of_that_record_alone(self):
        import torch

        from effects.application.breakdowns import TOTAL
        from effects.application.record_scoring import _record_losses
        from effects.domain.effect_head_layout import (
            PER_ENTITY_FIELDS,
            PER_ENTITY_WIDTH,
            EntityFieldTargets,
            FieldType,
        )
        from effects.domain.effect_losses import per_entity_loss

        torch.manual_seed(0)
        fields = PER_ENTITY_FIELDS
        outputs = torch.randn(2, 3, PER_ENTITY_WIDTH)
        gate = torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        targets = EntityFieldTargets.everywhere({
            spec.name: torch.zeros(
                (2, 3, spec.arity) if spec.type is FieldType.MULTI_BINARY else (2, 3)
            )
            for spec in fields
        })
        for row in range(2):
            expected, _ = per_entity_loss(
                outputs[row : row + 1], gate[row : row + 1],
                targets.select(slice(row, row + 1)),
                mask[row : row + 1], fields=fields,
            )
            losses = _record_losses(row, outputs, gate, targets, mask, fields)
            assert losses[TOTAL] == pytest.approx(float(expected), rel=1e-5)

    def test_a_record_with_nothing_affected_supervises_only_the_gate(self):
        import torch

        from effects.application.breakdowns import TOTAL
        from effects.application.record_scoring import _record_losses
        from effects.domain.effect_head_layout import (
            PER_ENTITY_FIELDS,
            PER_ENTITY_WIDTH,
            EntityFieldTargets,
        )

        outputs = torch.zeros(1, 2, PER_ENTITY_WIDTH)
        losses = _record_losses(
            0, outputs, torch.zeros(1, 2), EntityFieldTargets({}, {}), torch.ones(1, 2),
            PER_ENTITY_FIELDS,
        )
        assert set(losses) == {"gate", TOTAL}

    def test_every_term_equals_the_masked_sum_over_that_record(self):
        """Each reported term, not only the total, is the gate BCE over the real
        entities plus each field's loss over the target-affected ones — the
        same numbers a hand-masked sum gives, on targets that are not all zero."""
        import torch
        from torch.nn import functional

        from effects.application.breakdowns import TOTAL
        from effects.application.record_scoring import _record_losses
        from effects.domain.effect_head_layout import (
            FIELD_SLICES,
            GATE_INDEX,
            PER_ENTITY_FIELDS,
            PER_ENTITY_WIDTH,
            EntityFieldTargets,
            FieldType,
        )
        from effects.domain.effect_losses import field_loss

        generator = torch.Generator().manual_seed(3)
        fields = PER_ENTITY_FIELDS
        outputs = torch.randn(3, 4, PER_ENTITY_WIDTH, generator=generator)
        gate = torch.tensor([
            [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
        ])
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],
        ])

        def target(spec):
            match spec.type:
                case FieldType.MULTI_BINARY:
                    return torch.randint(0, 2, (3, 4, spec.arity), generator=generator)
                case FieldType.CATEGORICAL:
                    return torch.randint(0, spec.arity, (3, 4), generator=generator)
                case FieldType.SIGNED_DELTA:
                    return torch.randint(-3, 4, (3, 4), generator=generator)
                case FieldType.COUNT:
                    return torch.randint(0, 5, (3, 4), generator=generator)
            return torch.randint(0, 2, (3, 4), generator=generator)

        values = {spec.name: target(spec).float() for spec in fields}
        targets = EntityFieldTargets.everywhere(values)
        for row in range(3):
            real = mask[row].bool()
            affected = real & gate[row].bool()
            expected = {"gate": float(functional.binary_cross_entropy_with_logits(
                outputs[row, :, GATE_INDEX][real], gate[row][real], reduction="sum",
            ))}
            if bool(affected.any()):
                for spec in fields:
                    start, end = FIELD_SLICES[spec.name]
                    expected[spec.name] = float(field_loss(
                        spec, outputs[row, :, start:end][affected],
                        values[spec.name][row][affected],
                    ))
            losses = _record_losses(row, outputs, gate, targets, mask, fields)
            assert set(losses) == {*expected, TOTAL}
            for name, value in expected.items():
                assert losses[name] == pytest.approx(value, rel=1e-6, abs=1e-6), name
            assert losses[TOTAL] == pytest.approx(sum(expected.values()), rel=1e-6)

