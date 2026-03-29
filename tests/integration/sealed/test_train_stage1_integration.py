"""T012 — Integration test for TrainStage1UseCase."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.application.train_stage1 import TrainStage1UseCase
from sealed.infrastructure.pool_loader import card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore

# Miniaturized config: 4 slots, 8-dim card embeddings, d_model=12
MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=12,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)

# 4 booster cards + 6 basic lands = 10 .npz files needed
BOOSTER_CARDS = ["Card1", "Card2", "Card3", "Card4"]
BASIC_LANDS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
EMBED_DIM = 8  # must match MINI.card_embed_dim


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.stem)) % (2**31))
    embedding = rng.random(EMBED_DIM).astype(np.float32)
    np.savez_compressed(path, embedding=embedding)


@pytest.fixture()
def env(tmp_path):
    """Set up real filesystem environment: cards, pools, model path."""
    cards_path = tmp_path / "cards"
    # Write 4 booster + 6 basic land .npz files in the encode-cards layout
    for name in BOOSTER_CARDS + BASIC_LANDS:
        _write_npz(card_npz_path(cards_path, name))

    pools_path = tmp_path / "pools"
    pools_path.mkdir(parents=True)
    pool_line = ";".join(BOOSTER_CARDS)
    (pools_path / "pools.txt").write_text(pool_line + "\n", encoding="utf-8")

    model_path = tmp_path / "models" / "stage1" / "latest.pt"

    return {
        "pools_path": pools_path,
        "cards_path": cards_path,
        "model_path": model_path,
        "tmp_path": tmp_path,
    }


def _run_n_batches(env, n_batches: int, batch_size: int = 2) -> None:
    """Run the use case for exactly n_batches, then stop via side-effect."""
    call_count = [0]
    original_save = PoolModelStore.save

    def _save_and_maybe_stop(self, path, model, optimizer, training_state):
        original_save(self, path, model, optimizer, training_state)
        call_count[0] += 1
        if call_count[0] >= n_batches:
            raise _Stop()

    use_case = TrainStage1UseCase()
    with patch("sealed.application.train_stage1.PoolTransformerConfig", return_value=MINI):
        with patch.object(PoolModelStore, "save", _save_and_maybe_stop):
            with pytest.raises(_Stop):
                use_case.execute(
                    pools_path=env["pools_path"],
                    cards_path=env["cards_path"],
                    model_path=env["model_path"],
                    batch_size=batch_size,
                    set_code="TEST",
                )


class _Stop(Exception):
    pass


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_latest_pt_written_after_one_batch(env):
    _run_n_batches(env, n_batches=1, batch_size=2)
    assert env["model_path"].exists(), "latest.pt should exist after one batch"


def test_latest_pt_written_after_two_batches(env):
    _run_n_batches(env, n_batches=2, batch_size=2)
    assert env["model_path"].exists()


def test_checkpoint_loadable(env):
    """Checkpoint must be loadable after training."""
    _run_n_batches(env, n_batches=1, batch_size=2)

    store = PoolModelStore()
    ckpt = store.load(env["model_path"])
    assert ckpt.pool_transformer_state_dict is not None
    assert ckpt.optimizer_state_dict is not None
    assert ckpt.training_state is not None


def test_best_run_ge_1(env):
    """After at least one episode, best_run should be >= 1."""
    _run_n_batches(env, n_batches=1, batch_size=2)

    store = PoolModelStore()
    ckpt = store.load(env["model_path"])
    assert ckpt.training_state.best_run >= 1


def test_episode_count_incremented(env):
    """After two batches of batch_size=2, episode_count should be >= 4."""
    _run_n_batches(env, n_batches=2, batch_size=2)

    store = PoolModelStore()
    ckpt = store.load(env["model_path"])
    assert ckpt.training_state.episode_count >= 4


def test_no_exception_raised(env):
    """Running 2 batches should not raise any unexpected exceptions."""
    # _run_n_batches handles the controlled stop — no other exceptions expected
    _run_n_batches(env, n_batches=2, batch_size=2)
