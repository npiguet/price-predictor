"""A checkpoint's weights load strictly, less only the training-only heads.

Which input layout a loaded checkpoint is fed is ``InputLayout.of``'s, tested
with the layout in ``domain/test_effect_head_input.py``.
"""

from __future__ import annotations

from types import SimpleNamespace


def _checkpoint(drop=()):
    import torch

    from effects.domain.ability_encoder import AbilityEncoderConfig
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
    return SimpleNamespace(
        encoder_config=AbilityEncoderConfig(vocab_size=10, e_dim=4, d_model=8, n_layers=1),
        encoder_state={}, model_config=config, model_state=state,
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


def test_a_checkpoint_from_before_option_rows_still_loads():
    """The option-kind embedding starts at zero, so its absence changes nothing."""
    import torch

    from effects.infrastructure.effect_model_store import load_weights

    load_weights(
        _checkpoint(drop=("option_kind_embedding.weight",)), torch.device("cpu"),
    )
