"""A loaded checkpoint is run on the input layout it trained on (FR-063c).

The layout changes what the model's input vectors mean and how wide they are,
so a checkpoint has to be fed the one it learned: gen-1, and gen-2 arms trained
before the layout was recorded, record none and read the earlier layout.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from effects.domain.effect_head_input import (
    EARLIER_LAYOUT,
    INPUT_LAYOUT,
    THERMOMETER_STEPS,
    InputLayout,
    SlotKind,
    feature_widths,
)
from effects.domain.recorded_version import UnknownRecordedVersion
from tests.unit.effects.domain.test_effect_head_input import _record


def test_a_checkpoint_recording_no_layout_reads_the_earlier_one():
    assert InputLayout.of({}) is EARLIER_LAYOUT
    assert InputLayout.of(None) is EARLIER_LAYOUT


def test_a_checkpoint_recording_the_layout_reads_it():
    assert InputLayout.of({"input_layout": "gen-2"}) is INPUT_LAYOUT


def test_a_layout_this_code_does_not_know_is_refused():
    with pytest.raises(UnknownRecordedVersion, match="gen-3"):
        InputLayout.of({"input_layout": "gen-3"})


def test_a_run_records_its_layout_under_the_value_checkpoints_on_disk_hold():
    assert INPUT_LAYOUT.name == "gen-2"
    assert EARLIER_LAYOUT.name is None


def test_the_widths_follow_the_layout():
    records = [_record()]
    earlier = feature_widths(records, layout=EARLIER_LAYOUT)
    gen2 = feature_widths(records, layout=INPUT_LAYOUT)
    assert gen2[SlotKind.GLOBAL] == earlier[SlotKind.GLOBAL]
    assert gen2[SlotKind.PLAYER] == earlier[SlotKind.PLAYER] + THERMOMETER_STEPS
    assert gen2[SlotKind.CARD] == earlier[SlotKind.CARD] + 3 * THERMOMETER_STEPS + 1


# ── loading the weights ──────────────────────────────────────────────────


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
