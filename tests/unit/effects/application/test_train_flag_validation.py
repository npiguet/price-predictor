"""The flags a run cannot start with are refused when the config is built,
before any start-up work."""

from __future__ import annotations

import pytest

from effects.application.train_effect_model import TrainEffectModelConfig


@pytest.mark.parametrize("flag, value", [
    ("grad_accum", 0), ("batch_size", 0), ("steps_per_epoch", 0), ("epochs", 0),
    ("grad_accum", -1), ("lr_scale", -0.5), ("lr_scale", float("nan")),
    ("curriculum_epoch", 41), ("curriculum_rewarm_steps", -1),
    ("shards_per_epoch", 0), ("shards_per_epoch", -3), ("e_dim", 0),
    ("lethality_copies", -1), ("patience", -1),
])
def test_a_flag_the_run_cannot_train_with_is_refused_by_name(flag, value):
    with pytest.raises(ValueError, match=flag.replace("_", "-")):
        TrainEffectModelConfig.unpathed(**{flag: value})


def test_a_curriculum_epoch_inside_the_run_is_accepted():
    TrainEffectModelConfig.unpathed(epochs=5, curriculum_epoch=5, lr_scale=0.0)


def test_the_re_warm_defaults_to_500_steps_and_0_turns_it_off():
    assert TrainEffectModelConfig.unpathed().curriculum_rewarm_steps == 500
    TrainEffectModelConfig.unpathed(curriculum_rewarm_steps=0)


def test_a_step_count_the_accumulation_does_not_divide_is_refused_naming_both_flags():
    with pytest.raises(ValueError, match=r"--steps-per-epoch 7.*--grad-accum 2"):
        TrainEffectModelConfig.unpathed(grad_accum=2, steps_per_epoch=7, epochs=3)


def test_a_step_count_the_accumulation_divides_is_accepted():
    TrainEffectModelConfig.unpathed(grad_accum=2, steps_per_epoch=8)
    TrainEffectModelConfig.unpathed(grad_accum=1, steps_per_epoch=7)
