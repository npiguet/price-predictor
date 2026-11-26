"""The flags a run cannot start with are refused when the config is built,
before any start-up work (finding 11.11)."""

from __future__ import annotations

import pytest

from effects.application.train_effect_model import TrainEffectModelConfig


@pytest.mark.parametrize("flag, value", [
    ("grad_accum", 0), ("batch_size", 0), ("steps_per_epoch", 0), ("epochs", 0),
    ("grad_accum", -1), ("lr_scale", -0.5), ("lr_scale", float("nan")),
    ("curriculum_epoch", 41),
])
def test_a_flag_the_run_cannot_train_with_is_refused_by_name(flag, value):
    with pytest.raises(ValueError, match=flag.replace("_", "-")):
        TrainEffectModelConfig.unpathed(**{flag: value})


def test_a_curriculum_epoch_inside_the_run_is_accepted():
    TrainEffectModelConfig.unpathed(epochs=5, curriculum_epoch=5, lr_scale=0.0)
