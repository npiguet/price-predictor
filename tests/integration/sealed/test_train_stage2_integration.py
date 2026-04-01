"""T021 — Integration test for TrainStage2UseCase."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.application.train_stage1 import TrainingState
from sealed.application.train_stage2 import TrainStage2UseCase
from sealed.infrastructure.pool_loader import card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore
from sealed.domain.episode_runner import MAX_PICKS

# Miniaturized config: 4 slots, 8-dim card embeddings
# d_model = card_embed_dim(8) + 8 flag bits = 16 (must match _build_base_tensor)
MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=16,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)

BOOSTER_CARDS = ["Card1", "Card2", "Card3", "Card4"]
BASIC_LANDS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
EMBED_DIM = 8


def _write_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.stem)) % (2**31))
    embedding = rng.random(EMBED_DIM).astype(np.float32)
    np.savez_compressed(path, embedding=embedding)


def _write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def env(tmp_path):
    """Set up real filesystem environment with Stage 1 checkpoint."""
    cards_path = tmp_path / "cards"

    for name in BOOSTER_CARDS + BASIC_LANDS:
        _write_npz(card_npz_path(cards_path, name))
        txt_path = card_npz_path(cards_path, name).with_suffix(".txt")
        if name in BASIC_LANDS:
            color = {"Plains": "W", "Island": "U", "Swamp": "B",
                     "Mountain": "R", "Forest": "G", "Wastes": "C"}[name]
            _write_txt(txt_path, f"name: {name.lower()}\ntypes: basic land\nactivated[1]: {{T}}: add {{{color}}}\n")
        else:
            _write_txt(txt_path, f"name: {name.lower()}\nmana cost: {{1}}{{W}}\ntypes: creature\n")

    pools_path = tmp_path / "pools"
    pools_path.mkdir(parents=True)
    (pools_path / "pools.txt").write_text(";".join(BOOSTER_CARDS) + "\n", encoding="utf-8")

    # Create a Stage 1 checkpoint (init_from)
    import torch.optim as optim
    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    stage1_state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from = tmp_path / "models" / "stage1" / "latest.pt"
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, stage1_state)

    model_path = tmp_path / "models" / "stage2" / "latest.pt"

    return {
        "pools_path": pools_path,
        "cards_path": cards_path,
        "model_path": model_path,
        "init_from": init_from,
        "tmp_path": tmp_path,
    }


class _Stop(Exception):
    pass


def _run_n_batches(env: dict, n_batches: int, batch_size: int = 2) -> None:
    call_count = [0]
    original_save = PoolModelStore.save

    def _save_and_stop(self, path, model, optimizer, training_state):
        original_save(self, path, model, optimizer, training_state)
        call_count[0] += 1
        if call_count[0] >= n_batches:
            raise _Stop()

    use_case = TrainStage2UseCase()
    with patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI):
        with patch.object(PoolModelStore, "save", _save_and_stop):
            with pytest.raises(_Stop):
                use_case.execute(
                    pools_path=env["pools_path"],
                    cards_path=env["cards_path"],
                    model_path=env["model_path"],
                    init_from=env["init_from"],
                    batch_size=batch_size,
                    set_code="TEST",
                )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_stage2_checkpoint_written_after_one_batch(env):
    _run_n_batches(env, n_batches=1, batch_size=2)
    assert env["model_path"].exists(), "stage2 latest.pt should exist after one batch"


def test_stage2_checkpoint_loadable(env):
    _run_n_batches(env, n_batches=1, batch_size=2)
    ckpt = PoolModelStore().load(env["model_path"])
    assert ckpt.pool_transformer_state_dict is not None
    assert ckpt.optimizer_state_dict is not None
    assert ckpt.training_state is not None


def test_stage2_episode_count_incremented(env):
    _run_n_batches(env, n_batches=2, batch_size=2)
    ckpt = PoolModelStore().load(env["model_path"])
    assert ckpt.training_state.episode_count >= 4


def test_pool_reshuffled_before_each_pick(env):
    """EpisodeRunner uses different seeds per step — pool is reshuffled per pick."""
    # This is verified by the episode runner's shuffle_seeds: each step has a unique seed.
    # We verify this at the integration level by checking episode.shuffle_seeds are unique.
    from sealed.application.train_stage2 import TrainStage2UseCase
    from sealed.domain.episode_runner import EpisodeRunner

    episodes_seen: list = []
    original_run = EpisodeRunner.run

    def capture_run(self, **kwargs):
        ep = original_run(self, **kwargs)
        episodes_seen.append(ep)
        return ep

    call_count = [0]
    original_save = PoolModelStore.save

    def _save_and_stop(self, path, model, optimizer, training_state):
        original_save(self, path, model, optimizer, training_state)
        call_count[0] += 1
        if call_count[0] >= 1:
            raise _Stop()

    use_case = TrainStage2UseCase()
    with patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI):
        with patch.object(EpisodeRunner, "run", capture_run):
            with patch.object(PoolModelStore, "save", _save_and_stop):
                with pytest.raises(_Stop):
                    use_case.execute(
                        pools_path=env["pools_path"],
                        cards_path=env["cards_path"],
                        model_path=env["model_path"],
                        init_from=env["init_from"],
                        batch_size=2,
                        set_code="TEST",
                    )

    assert len(episodes_seen) > 0
    for ep in episodes_seen:
        if len(ep.shuffle_seeds) > 1:
            # shuffle_seeds should not all be the same (each step uses a different seed)
            assert len(set(ep.shuffle_seeds.tolist())) > 1, (
                "Shuffle seeds should vary per step (pool reshuffled before each pick)"
            )


def test_mana_scores_computed_for_completed_episodes(env):
    """Completed episodes should have their rewards replaced with mana scores."""
    from sealed.application.train_stage2 import TrainStage2UseCase
    from sealed.domain.ppo_trainer import PPOTrainer

    captured_episodes: list = []
    original_update = PPOTrainer.update

    def capture_update(self, episodes, card_port):
        captured_episodes.extend(episodes)
        return original_update(self, episodes, card_port)

    call_count = [0]
    original_save = PoolModelStore.save

    def _save_and_stop(self, path, model, optimizer, training_state):
        original_save(self, path, model, optimizer, training_state)
        call_count[0] += 1
        if call_count[0] >= 1:
            raise _Stop()

    use_case = TrainStage2UseCase()
    with patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI):
        with patch.object(PPOTrainer, "update", capture_update):
            with patch.object(PoolModelStore, "save", _save_and_stop):
                with pytest.raises(_Stop):
                    use_case.execute(
                        pools_path=env["pools_path"],
                        cards_path=env["cards_path"],
                        model_path=env["model_path"],
                        init_from=env["init_from"],
                        batch_size=2,
                        set_code="TEST",
                    )

    # Check that completed episodes have uniform step_rewards (mana score)
    completed = [ep for ep in captured_episodes if ep.termination == "success"]
    for ep in completed:
        if len(ep.step_rewards) > 1:
            rewards_set = set(ep.step_rewards.tolist())
            assert len(rewards_set) == 1, (
                f"Completed episode should have uniform step_rewards, got {rewards_set}"
            )
