"""T018 — Unit tests for PoolModelStore."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.application.train_stage1 import TrainingState
from sealed.infrastructure.pool_model_store import CheckpointData, PoolModelStore

MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=16,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)


def _make_artifacts():
    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    state = TrainingState(best_run=3, episode_count=50)
    return model, optimizer, state


# ── save + load round-trip ─────────────────────────────────────────────────────

def test_save_load_round_trip_state_dict(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    path = tmp_path / "latest.pt"

    store.save(path, model, optimizer, state)
    ckpt = store.load(path)

    assert isinstance(ckpt, CheckpointData)
    assert set(ckpt.pool_transformer_state_dict.keys()) == set(model.state_dict().keys())


def test_save_load_round_trip_training_state(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    path = tmp_path / "latest.pt"

    store.save(path, model, optimizer, state)
    ckpt = store.load(path)

    loaded_state = ckpt.training_state
    assert loaded_state.best_run == state.best_run
    assert loaded_state.episode_count == state.episode_count


def test_save_creates_parent_dirs(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    path = tmp_path / "deep" / "nested" / "model.pt"

    store.save(path, model, optimizer, state)
    assert path.exists()


def test_no_tmp_file_after_save(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    path = tmp_path / "latest.pt"

    store.save(path, model, optimizer, state)
    tmp_files = list(tmp_path.glob("*.tmp.pt"))
    assert tmp_files == [], f"Temp files should be cleaned up: {tmp_files}"


def test_load_raises_file_not_found(tmp_path):
    store = PoolModelStore()
    with pytest.raises(FileNotFoundError):
        store.load(tmp_path / "nonexistent.pt")


# ── save_timestamped ──────────────────────────────────────────────────────────

def test_save_timestamped_creates_file_under_checkpoints(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    base_path = tmp_path / "latest.pt"

    dest = store.save_timestamped(base_path, model, optimizer, state)
    assert dest.exists()
    assert dest.parent.name == "checkpoints"


def test_save_timestamped_iso8601_filename(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    base_path = tmp_path / "latest.pt"

    dest = store.save_timestamped(base_path, model, optimizer, state)
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.pt$"
    assert re.match(pattern, dest.name), f"Unexpected filename: {dest.name}"


def test_save_timestamped_returns_path(tmp_path):
    model, optimizer, state = _make_artifacts()
    store = PoolModelStore()
    base_path = tmp_path / "latest.pt"

    result = store.save_timestamped(base_path, model, optimizer, state)
    assert isinstance(result, Path)
