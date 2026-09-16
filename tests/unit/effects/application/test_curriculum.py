"""The sparse-field curriculum starts at an epoch boundary (FR-082)."""

from __future__ import annotations

from effects.application.train_effect_model import (
    EarlyStopper,
    TrainEffectModelConfig,
    fields_for_epoch,
)
from effects.domain.effect_model import (
    CLASS_RESOLUTION_EFFECT,
    FieldGroup,
    PER_ENTITY_FIELDS,
)

PRESENT = frozenset({CLASS_RESOLUTION_EFFECT})


def _sparse(fields):
    return [f for f in fields if f.group is not FieldGroup.DENSE]


def test_the_curriculum_step_is_the_first_step_of_the_curriculum_epoch():
    config = TrainEffectModelConfig(steps_per_epoch=5000, curriculum_epoch=3)
    assert config.curriculum_step == 10_000
    assert TrainEffectModelConfig(steps_per_epoch=5000, curriculum_epoch=1).curriculum_step == 0


def test_sparse_fields_are_off_before_the_curriculum_epoch_and_on_from_it():
    config = TrainEffectModelConfig(steps_per_epoch=5000, curriculum_epoch=3)
    assert _sparse(fields_for_epoch(config, present=PRESENT, epoch=2)) == []
    on = _sparse(fields_for_epoch(config, present=PRESENT, epoch=3))
    assert on and all(f in PER_ENTITY_FIELDS for f in on)


def test_the_stopper_resets():
    stopper = EarlyStopper(patience=2)
    assert stopper.update(1.0)
    assert not stopper.update(2.0)
    stopper.reset()
    assert stopper.best == float("inf") and stopper.since_best == 0
    assert stopper.update(5.0)          # a worse number is a new best after the reset
