"""A checkpoint's payload is one table of members that save and load both read,
and a loader refuses what it cannot run rather than filling a default
(task H04, finding 11.8)."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from effects.domain.baselines import Baseline
from effects.infrastructure.effect_model_store import PAYLOAD_MEMBERS, EffectModelStore
from tests.unit.effects.infrastructure.test_effect_model_store import _checkpoint

_KEYS = [member.key for member in PAYLOAD_MEMBERS]


def _saved_path(tmp_path, files, **overrides):
    return EffectModelStore(tmp_path).save(_checkpoint(files, **overrides))


def _without(path, key):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload[key]
    torch.save(payload, path)


def test_a_saved_payload_holds_the_tables_keys_and_the_config(tmp_path, files):
    path = _saved_path(tmp_path, files)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload) == {*_KEYS, "config"}


@pytest.mark.parametrize("key", _KEYS)
def test_a_payload_missing_a_member_raises_a_key_error_naming_it(tmp_path, files, key):
    path = _saved_path(tmp_path, files)
    _without(path, key)
    with pytest.raises(KeyError) as raised:
        EffectModelStore(tmp_path).load(path)
    assert raised.value.args == (key,)


def test_settings_without_rule_mask_raise_key_error_rule_mask(tmp_path, files):
    """The flags are settings, not payload members, and none is read with a
    default (task H09)."""
    assert "rule_mask" not in _KEYS
    loaded = EffectModelStore(tmp_path).load(_saved_path(tmp_path, files))
    with pytest.raises(KeyError, match="rule_mask"):
        loaded.scored_under()


def test_an_empty_encoder_state_is_refused(tmp_path, files):
    path = _saved_path(tmp_path, files, encoder_state={})
    with pytest.raises(ValueError, match="encoder_state_dict"):
        EffectModelStore(tmp_path).load(path)


def test_an_encoder_config_lacking_a_size_is_refused_by_its_name(tmp_path, files):
    path = _saved_path(tmp_path, files)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["encoder_config"]["n_layers"]
    torch.save(payload, path)
    with pytest.raises(KeyError, match="n_layers"):
        EffectModelStore(tmp_path).load(path)


def test_extra_sits_under_its_own_key_so_no_name_can_collide_with_a_member(
    tmp_path, files,
):
    path = _saved_path(tmp_path, files, extra={"epoch": 99, "identity_table": {"a": 1}})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["extra"] == {"epoch": 99, "identity_table": {"a": 1}}
    loaded = EffectModelStore(tmp_path).load(path)
    assert (loaded.epoch, loaded.extra) == (0, {"epoch": 99, "identity_table": {"a": 1}})


def test_every_member_survives_a_round_trip(tmp_path, files):
    checkpoint = replace(
        _checkpoint(files), variant=Baseline.IDENTITY, epoch=3,
        cards_folders=("a", "b"), control_score=0.5, averaged=False,
    )
    path = EffectModelStore(tmp_path).save(checkpoint)
    loaded = EffectModelStore(tmp_path).load(path)
    for member in PAYLOAD_MEMBERS:
        if member.attribute not in ("encoder_state", "model_state"):
            assert getattr(loaded, member.attribute) == getattr(checkpoint, member.attribute)


def test_a_model_config_lacking_a_size_is_refused_by_its_name(tmp_path, files):
    path = _saved_path(tmp_path, files)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["config"]["d_model"]
    torch.save(payload, path)
    with pytest.raises(KeyError) as raised:
        EffectModelStore(tmp_path).load(path)
    assert raised.value.args == ("d_model",)
