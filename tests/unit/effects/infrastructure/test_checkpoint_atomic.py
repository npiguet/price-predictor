"""Every checkpoint file is written to a temporary name and moved into place,
and a file under ``epochs/`` is named for its epoch and says whose score and
which weights it holds (task H03, findings 15.1, 13.7)."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
import torch

from effects.infrastructure.effect_model_store import EffectModelStore
from price_predictor.infrastructure import torch_checkpoint
from price_predictor.infrastructure.torch_checkpoint import (
    copy_atomically,
    write_atomically,
)
from tests.unit.effects.infrastructure.test_effect_model_store import (
    _checkpoint,
    files,  # noqa: F401  (fixture)
)

_real_replace = os.replace


def _failing_replace(error):
    def replace_(source, target):
        raise error
    return replace_


def test_an_interrupted_save_leaves_the_previous_latest_loadable(
    tmp_path, files, monkeypatch,  # noqa: F811
):
    store = EffectModelStore(tmp_path / "run")
    store.save(replace(_checkpoint(files), epoch=1))
    before = store.latest_path().read_bytes()
    stamped = sorted(p.name for p in store.directory.glob("2*.pt"))

    seen = {}

    def interrupted(source, target):
        seen["temporary_written"] = os.path.exists(source)
        raise KeyboardInterrupt

    monkeypatch.setattr(torch_checkpoint.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        store.save(replace(_checkpoint(files), epoch=2))
    monkeypatch.undo()

    assert seen["temporary_written"]
    assert store.latest_path().read_bytes() == before
    assert store.load().epoch == 1
    assert sorted(p.name for p in store.directory.glob("2*.pt")) == stamped
    assert not list(store.directory.glob("*.writing*"))


def test_a_replace_onto_a_locked_file_raises_naming_the_path(tmp_path, monkeypatch):
    target = tmp_path / "latest.pt"
    target.write_bytes(b"old")
    monkeypatch.setattr(
        torch_checkpoint.os, "replace", _failing_replace(PermissionError("locked")),
    )
    with pytest.raises(OSError, match=r"latest\.pt"):
        write_atomically(target, lambda temporary: temporary.write_bytes(b"new"))
    assert target.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["latest.pt"]


def test_a_copy_goes_through_a_temporary_name_too(tmp_path, monkeypatch):
    source = tmp_path / "resume.pt"
    source.write_bytes(b"new")
    target = tmp_path / "resume-best.pt"
    target.write_bytes(b"old")
    monkeypatch.setattr(
        torch_checkpoint.os, "replace", _failing_replace(PermissionError("locked")),
    )
    with pytest.raises(OSError, match=r"resume-best\.pt"):
        copy_atomically(source, target)
    monkeypatch.undo()
    assert target.read_bytes() == b"old"
    copy_atomically(source, target)
    assert target.read_bytes() == b"new"


def test_a_non_best_epochs_file_holds_its_own_score(tmp_path, files):  # noqa: F811
    store = EffectModelStore(tmp_path / "epochs")
    path = store.save_epoch(replace(
        _checkpoint(files), epoch=3, control_score=0.4, best_control_score=0.9,
        averaged=False,
    ))
    assert path.name == "epoch-03.pt"
    assert not (store.directory / "latest.pt").exists()
    loaded = store.load(path)
    assert (loaded.control_score, loaded.best_control_score) == (0.4, 0.9)
    assert loaded.averaged is False
    assert torch.is_tensor(next(iter(loaded.model_state.values())))
