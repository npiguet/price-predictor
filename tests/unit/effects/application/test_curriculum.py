"""The sparse-field curriculum starts at an epoch boundary (FR-082)."""

from __future__ import annotations

from effects.application.training_schedule import Curriculum, EarlyStopper
from effects.domain.effect_head_layout import PER_ENTITY_FIELDS, FieldGroup
from effects.domain.sampling import SamplingClass

PRESENT = frozenset({SamplingClass.RESOLUTION_EFFECT})


def _sparse(fields):
    return [f for f in fields if f.group is not FieldGroup.DENSE]


def test_sparse_fields_are_off_before_the_curriculum_epoch_and_on_from_it():
    curriculum = Curriculum(3)
    assert _sparse(curriculum.fields_for(2, PRESENT)) == []
    on = _sparse(curriculum.fields_for(3, PRESENT))
    assert on and all(f in PER_ENTITY_FIELDS for f in on)
    assert _sparse(curriculum.fields_for(4, PRESENT)) == on


def test_a_first_epoch_curriculum_enables_the_sparse_group_from_the_first_step():
    for epoch in (0, 1):
        assert _sparse(Curriculum(epoch).fields_for(1, PRESENT))


def test_the_stopper_resets():
    stopper = EarlyStopper(patience=2)
    assert stopper.update(1.0)
    assert not stopper.update(2.0)
    stopper.reset()
    assert stopper.best == float("inf") and stopper.since_best == 0
    assert stopper.update(5.0)          # a worse number is a new best after the reset


def test_the_loop_resets_the_stopper_when_the_curriculum_switches():
    """Losses before and after the sparse group enables do not compare (FR-082).

    The reset fires once, at the curriculum epoch, and never at epoch 1 — a
    ``--curriculum-epoch 1`` run enables the sparse group from its first step,
    so there is no earlier best measuring a different objective to forget.
    """
    curriculum = Curriculum(2)
    assert curriculum.resets_at(1) is False
    assert curriculum.resets_at(2) is True
    assert curriculum.resets_at(3) is False
    assert Curriculum(1).resets_at(1) is False
