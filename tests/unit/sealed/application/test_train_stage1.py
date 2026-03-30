"""T010 + T019 — Unit tests for TrainStage1UseCase."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.application.train_stage1 import TrainStage1UseCase, TrainingState
from sealed.infrastructure.pool_loader import card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore

MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=12,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)

EMBED_DIM = 8


def _write_npz(path: Path, embed_dim: int = EMBED_DIM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=np.ones(embed_dim, dtype=np.float32))


BASIC_LANDS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
BOOSTER_CARDS = ["Card1", "Card2", "Card3", "Card4"]


def _setup_cards(cards_path: Path) -> None:
    for name in BOOSTER_CARDS + BASIC_LANDS:
        _write_npz(card_npz_path(cards_path, name))


def _setup_pools(pools_path: Path) -> None:
    pools_path.mkdir(parents=True, exist_ok=True)
    (pools_path / "pools.txt").write_text(
        ";".join(BOOSTER_CARDS) + "\n", encoding="utf-8"
    )


def _run_one_batch(tmp_path: Path) -> Path:
    """Helper: run TrainStage1UseCase for 1 batch using mini config. Returns model_path."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    use_case = TrainStage1UseCase()

    # Patch PoolTransformerConfig to return mini config, and stop after 1 batch
    call_count = [0]
    original_execute = use_case.execute

    with patch("sealed.application.train_stage1.PoolTransformerConfig", return_value=MINI):
        # Stop after 1 batch via a side-effect on model_store.save
        saved = []

        original_save = PoolModelStore.save

        def _save_and_stop(self, path, model, optimizer, training_state):
            original_save(self, path, model, optimizer, training_state)
            saved.append(1)
            if len(saved) >= 1:
                raise _StopAfterBatch()

        with patch.object(PoolModelStore, "save", _save_and_stop):
            with pytest.raises(_StopAfterBatch):
                use_case.execute(
                    pools_path=pools_path,
                    cards_path=cards_path,
                    model_path=model_path,
                    batch_size=2,
                )

    return model_path


class _StopAfterBatch(Exception):
    pass


# ── Validation errors ─────────────────────────────────────────────────────────

def test_empty_pools_raises_value_error(tmp_path):
    pools_path = tmp_path / "pools"
    pools_path.mkdir()
    (pools_path / "pools.txt").write_text("", encoding="utf-8")
    cards_path = tmp_path / "cards"
    cards_path.mkdir()
    model_path = tmp_path / "model.pt"

    use_case = TrainStage1UseCase()
    with pytest.raises(ValueError, match="pools.txt not found or empty"):
        with patch("sealed.application.train_stage1.PoolTransformerConfig", return_value=MINI):
            use_case.execute(pools_path, cards_path, model_path, batch_size=1)


def test_missing_npz_raises_file_not_found(tmp_path):
    pools_path = tmp_path / "pools"
    _setup_pools(pools_path)
    cards_path = tmp_path / "cards"
    cards_path.mkdir()
    # No .npz files created
    model_path = tmp_path / "model.pt"

    use_case = TrainStage1UseCase()
    with pytest.raises(FileNotFoundError):
        with patch("sealed.application.train_stage1.PoolTransformerConfig", return_value=MINI):
            use_case.execute(pools_path, cards_path, model_path, batch_size=1)


def test_model_path_dir_created(tmp_path):
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    _setup_pools(pools_path)
    _setup_cards(cards_path)
    model_path = tmp_path / "deep" / "nested" / "model.pt"

    use_case = TrainStage1UseCase()
    with patch("sealed.application.train_stage1.PoolTransformerConfig", return_value=MINI):
        with patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch):
            with pytest.raises(_StopAfterBatch):
                use_case.execute(pools_path, cards_path, model_path, batch_size=1)

    assert model_path.parent.exists()


# ── T019 Checkpoint resume ─────────────────────────────────────────────────────

def test_checkpoint_resume_restores_state(tmp_path):
    """Loading an existing checkpoint should restore model weights, optimizer, and training state."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    _setup_pools(pools_path)
    _setup_cards(cards_path)
    model_path = tmp_path / "models" / "latest.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a checkpoint with known training state
    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=5, episode_count=100)
    store = PoolModelStore()
    store.save(model_path, model, optimizer, state)

    # Now run use case and check it picks up the state
    loaded_states = []
    original_save = PoolModelStore.save

    def _capture_save(self, path, mdl, opt, training_state):
        loaded_states.append(training_state)
        original_save(self, path, mdl, opt, training_state)
        raise _StopAfterBatch()

    use_case = TrainStage1UseCase()
    with patch("sealed.application.train_stage1.PoolTransformerConfig", return_value=MINI):
        with patch.object(PoolModelStore, "save", _capture_save):
            with pytest.raises(_StopAfterBatch):
                use_case.execute(pools_path, cards_path, model_path, batch_size=1)

    assert len(loaded_states) >= 1
    saved_state = loaded_states[0]
    # episode_count should have incremented from 100
    assert saved_state.episode_count > 100
    # best_run should be at least 5 (may increase)
    assert saved_state.best_run >= 5
