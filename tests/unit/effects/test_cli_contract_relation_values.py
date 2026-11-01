"""``train-effect-model --relation-values on|off`` (T103, story 4 acceptance 5)."""

from __future__ import annotations

import pytest

from effects.application.train_effect_model import TrainEffectModelConfig, on_off
from effects.infrastructure.cli import build_parser


def parse(*argv: str):
    return build_parser().parse_args(["train-effect-model", "--corpus", "c", *argv])


def test_the_default_is_on():
    assert parse().relation_values == "on"
    assert TrainEffectModelConfig.__dataclass_fields__["relation_values"].default == "on"


def test_off_is_accepted_and_anything_else_refused():
    assert parse("--relation-values", "off").relation_values == "off"
    with pytest.raises(SystemExit):
        parse("--relation-values", "maybe")
    with pytest.raises(ValueError):
        on_off("true")


def test_a_resume_refuses_a_change_of_relation_values():
    """Every config field not listed as changeable is compared on resume."""
    from effects.application.resume_state import RESUME_MAY_CHANGE

    assert "relation_values" not in RESUME_MAY_CHANGE


def test_off_builds_a_model_without_value_parameters():
    from effects.domain.effect_model import EffectModel, EffectModelConfig

    config = EffectModelConfig(global_features=2, act_features=2, player_features=2,
                               card_features=2, d_model=8, n_heads=2, n_layers=1, ff_dim=16,
                               relation_values=False)
    names = dict(EffectModel(config).named_parameters())
    assert "trunk.layers.0.rel_k" in names and "trunk.layers.0.rel_v" not in names
