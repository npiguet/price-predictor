"""Which settings a resume compares: every flag unless it is named as free."""

from __future__ import annotations

from dataclasses import fields, replace

import torch

from effects.application.resume_state import (
    RECORDED_IN_TRAINING_SETTINGS,
    RESUME_MAY_CHANGE,
    ResumeState,
    compared_flags,
)
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.infrastructure.effect_model_store import resume_mismatches

FLAGS = {spec.name for spec in fields(TrainEffectModelConfig)}


def _settings(config: TrainEffectModelConfig, **training_settings) -> dict:
    return ResumeState(
        config, seed=7, corpus_digest="d", training_settings=lambda: training_settings,
        noise=None, identity_table=None, device=torch.device("cpu"),
    ).settings()


def test_every_flag_is_compared_unless_it_is_named_as_free():
    """A flag added later is checked by default: it lands in none of the sets."""
    named = RESUME_MAY_CHANGE | RECORDED_IN_TRAINING_SETTINGS
    assert named <= FLAGS
    assert not RESUME_MAY_CHANGE & RECORDED_IN_TRAINING_SETTINGS
    assert set(compared_flags()) == FLAGS - named


def test_the_compared_flags_are_the_ones_that_place_a_step():
    assert set(compared_flags()) == {
        "variant", "e_dim", "encoder_layers", "encoder_d_model",
        "keyword_expand_p", "context_dropout", "curriculum_epoch", "batch_size",
        "grad_accum", "steps_per_epoch", "shards_per_epoch", "withhold_keyword",
    }


def test_a_compared_flag_that_moved_is_a_mismatch_and_a_free_one_is_not():
    config = TrainEffectModelConfig()
    saved = _settings(config)
    assert resume_mismatches(saved, _settings(replace(config, batch_size=16)))
    assert not resume_mismatches(
        saved, _settings(replace(config, epochs=80, patience=0, lr_scale=0.5)),
    )


def test_the_training_settings_are_compared_as_recorded():
    """``lethality_copies`` is recorded only when set, so a run without copies
    resumes against exactly the settings it always wrote."""
    config = TrainEffectModelConfig()
    settings = _settings(config, seed=7)
    assert "lethality_copies" not in settings
    assert settings["seed"] == 7
