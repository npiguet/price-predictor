"""A checkpoint loads only on the input layout it trained on, and its weights
load strictly, less only the training-only heads (FR-045, M1)."""

from __future__ import annotations

from types import SimpleNamespace


def _checkpoint(drop=(), *, encoder_state=None):
    import torch

    from effects.domain.ability_encoder import AbilityEncoder, AbilityEncoderConfig
    from effects.domain.effect_model import EffectModel, EffectModelConfig
    from effects.infrastructure.effect_model_store import filter_training_only

    config = EffectModelConfig(
        global_features=3, act_features=3, player_features=3, card_features=3,
        e_dim=4, vocab_size=10, n_api_types=2, n_param_keys=2,
        encoder_d_model=8, d_model=8, n_layers=1, n_heads=2, ff_dim=16,
    )
    torch.manual_seed(0)
    state = filter_training_only(EffectModel(config).state_dict())
    for key in drop:
        del state[key]
    encoder_config = AbilityEncoderConfig(vocab_size=10, e_dim=4, d_model=8, n_layers=1)
    if encoder_state is None:
        encoder_state = AbilityEncoder(encoder_config).state_dict()
    return SimpleNamespace(
        encoder_config=encoder_config,
        encoder_state=encoder_state, model_config=config, model_state=state,
    )


def test_a_shipped_checkpoint_loads_without_its_training_only_heads():
    import torch

    from effects.infrastructure.effect_model_store import load_weights

    encoder, model = load_weights(_checkpoint(), torch.device("cpu"))
    assert not encoder.training and not model.training


def test_a_missing_shipped_key_is_refused():
    """``strict=False`` would leave the head at its random initial weights."""
    import pytest
    import torch

    from effects.infrastructure.effect_model_store import load_weights

    with pytest.raises(RuntimeError, match="verdict_head"):
        load_weights(
            _checkpoint(drop=("verdict_head.0.weight",)), torch.device("cpu"),
        )


def test_a_checkpoint_missing_the_option_kind_embedding_is_refused():
    """Gen-1 checkpoints lacked it; no checkpoint this code loads may."""
    import pytest
    import torch

    from effects.infrastructure.effect_model_store import load_weights

    with pytest.raises(RuntimeError, match="option_kind_embedding"):
        load_weights(
            _checkpoint(drop=("option_kind_embedding.weight",)), torch.device("cpu"),
        )


def test_only_the_current_layout_is_accepted():
    import pytest

    from effects.domain.checkpoint_layout import (
        INPUT_LAYOUT,
        INPUT_LAYOUT_KEY,
        CheckpointLayoutError,
        layout_settings,
        require_input_layout,
    )

    require_input_layout(layout_settings())
    with pytest.raises(CheckpointLayoutError, match="another relation_types order"):
        require_input_layout({**layout_settings(), "relation_types_digest": "0" * 64})
    with pytest.raises(CheckpointLayoutError, match="another subtypes_changed order"):
        # Another closed output list moves the subtype outputs' columns.
        require_input_layout({**layout_settings(), "subtypes_changed_digest": "0" * 64})
    with pytest.raises(CheckpointLayoutError, match="another relation_target_indices"):
        require_input_layout({**layout_settings(), "relation_target_indices": 3})
    for settings in ({}, None, {INPUT_LAYOUT_KEY: "gen-1"}, {INPUT_LAYOUT_KEY: "gen-2"},
                     {**layout_settings(), INPUT_LAYOUT_KEY: "gen-2"}):
        with pytest.raises(CheckpointLayoutError, match=INPUT_LAYOUT):
            require_input_layout(settings)


def test_a_refusal_names_the_generation_the_checkpoint_belongs_to():
    """No recorded layout is a gen-1 checkpoint; ``"gen-2"`` is a gen-2 one."""
    import pytest

    from effects.domain.checkpoint_layout import (
        INPUT_LAYOUT_KEY,
        CheckpointLayoutError,
        layout_settings,
        require_input_layout,
    )

    gen3 = layout_settings()
    without_layout = {k: v for k, v in gen3.items() if k != INPUT_LAYOUT_KEY}
    for settings, named in (
        (without_layout, "gen-1 checkpoint"), (None, "gen-1 checkpoint"),
        ({**gen3, INPUT_LAYOUT_KEY: "gen-2"}, "gen-2 checkpoint"),
    ):
        with pytest.raises(CheckpointLayoutError, match=named):
            require_input_layout(settings)


def test_a_fresh_run_records_the_gen3_layout():
    """FR-045: what a new checkpoint records is what every loader accepts."""
    from effects.application.run_context import training_settings
    from effects.application.train_effect_model import (
        CorpusFacts,
        RunSpec,
        TrainEffectModelConfig,
    )
    from effects.domain.checkpoint_layout import INPUT_LAYOUT_KEY, require_input_layout

    settings = training_settings(
        RunSpec(TrainEffectModelConfig.unpathed(corpus="unused"), CorpusFacts(), seed=1),
        api_types=[], param_keys=[],
    )
    assert settings[INPUT_LAYOUT_KEY] == "gen-3"
    require_input_layout(settings)


def test_a_checkpoint_with_an_empty_encoder_state_is_refused():
    """An encoder loaded from nothing would run at random initial weights."""
    import pytest
    import torch

    from effects.infrastructure.effect_model_store import load_weights

    with pytest.raises(RuntimeError, match="encoder"):
        load_weights(_checkpoint(encoder_state={}), torch.device("cpu"))
