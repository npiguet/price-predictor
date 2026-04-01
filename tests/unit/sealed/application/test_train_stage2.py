"""T017 + T028-T029 — Unit tests for TrainStage2UseCase."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock, call
from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.replay_buffer import Episode
from sealed.application.train_stage1 import TrainingState
from sealed.application.train_stage2 import TrainStage2UseCase
from sealed.infrastructure.pool_loader import card_npz_path
from sealed.infrastructure.pool_model_store import PoolModelStore
from sealed.domain.episode_runner import MAX_PICKS


MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=16,
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


def _write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


BASIC_LANDS = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
BOOSTER_CARDS = ["Card1", "Card2", "Card3", "Card4"]


def _setup_cards(cards_path: Path) -> None:
    for name in BOOSTER_CARDS + BASIC_LANDS:
        _write_npz(card_npz_path(cards_path, name))
        txt_path = card_npz_path(cards_path, name).with_suffix(".txt")
        if name in BASIC_LANDS:
            color = {"Plains": "W", "Island": "U", "Swamp": "B",
                     "Mountain": "R", "Forest": "G", "Wastes": "C"}[name]
            _write_txt(txt_path, f"name: {name.lower()}\ntypes: basic land\nactivated[1]: {{T}}: add {{{color}}}\n")
        else:
            _write_txt(txt_path, f"name: {name.lower()}\nmana cost: {{1}}{{W}}\ntypes: creature\n")


def _setup_pools(pools_path: Path) -> None:
    pools_path.mkdir(parents=True, exist_ok=True)
    (pools_path / "pools.txt").write_text(
        ";".join(BOOSTER_CARDS) + "\n", encoding="utf-8"
    )


def _make_success_episode(pool_names: str = ";".join(BOOSTER_CARDS)) -> Episode:
    """Minimal completed episode (MAX_PICKS=40 picks, no duplicates)."""
    n_picks = MAX_PICKS
    return Episode(
        pool_names=pool_names,
        shuffle_seeds=np.zeros(n_picks + 1, dtype=np.int64),
        actions=np.array([0] * n_picks, dtype=np.int32),  # always pick slot 0 (basic land)
        log_probs=np.zeros(n_picks, dtype=np.float32),
        step_rewards=np.ones(n_picks, dtype=np.float32),
        reward=1.0,
        effective_run=n_picks,
        termination="success",
        term_action=-1,
        term_log_prob=0.0,
    )


def _make_duplicate_episode(pool_names: str = ";".join(BOOSTER_CARDS)) -> Episode:
    """Episode that terminated due to duplicate pick."""
    n_picks = 5
    return Episode(
        pool_names=pool_names,
        shuffle_seeds=np.zeros(n_picks + 2, dtype=np.int64),
        actions=np.array([0, 1, 2, 3, 0], dtype=np.int32)[:n_picks - 1],
        log_probs=np.zeros(n_picks - 1, dtype=np.float32),
        step_rewards=np.array([1.0, 1.0, 1.0, -1.0], dtype=np.float32),
        reward=-0.5,
        effective_run=4,
        termination="duplicate",
        term_action=0,
        term_log_prob=-0.5,
    )


class _StopAfterBatch(Exception):
    pass


# ─── Reward override ──────────────────────────────────────────────────────────

def test_completed_episode_step_rewards_overridden_with_mana_score(tmp_path):
    """Completed episodes should have step_rewards replaced with uniform mana score reward."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    # Create stage1 checkpoint for init-from
    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, state)

    ep = _make_success_episode()
    original_step_rewards = ep.step_rewards.copy()

    captured_episodes: list[list[Episode]] = []

    mock_runner = MagicMock()
    mock_runner.return_value.run.return_value = ep

    mock_ppo = MagicMock()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo.return_value.update.side_effect = lambda episodes, _port: (
        captured_episodes.append([e for e in episodes]) or mock_ppo_result
    )

    # Make runner return the episode we control
    episodes_seen: list[list] = []
    mock_runner_instance = MagicMock()
    mock_runner_instance.run.return_value = ep

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner_instance),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo.return_value),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    # The episode passed to update should have uniform step_rewards (mana score reward)
    # not the original per-step rewards
    assert len(captured_episodes) == 1
    updated_ep = captured_episodes[0][0]
    # All step_rewards should be identical (uniform mana score)
    assert len(set(updated_ep.step_rewards.tolist())) == 1, (
        "Completed episode step_rewards should all be equal (uniform mana score)"
    )
    # The reward should match the mana score (not be the Stage 1 effective_run-based reward)
    assert updated_ep.reward == updated_ep.step_rewards[0]


def test_duplicate_episode_step_rewards_unchanged(tmp_path):
    """Duplicate episodes should keep their original Stage 1 step_rewards."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, state)

    dup_ep = _make_duplicate_episode()
    original_rewards = dup_ep.step_rewards.copy()
    original_reward = dup_ep.reward

    captured_episodes: list[list[Episode]] = []

    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = -0.5
    mock_ppo_result.episode_runs = [4]
    mock_ppo = MagicMock()
    mock_ppo.update.side_effect = lambda episodes, _port: (
        captured_episodes.append(list(episodes)) or mock_ppo_result
    )

    mock_runner = MagicMock()
    mock_runner.run.return_value = dup_ep

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    assert len(captured_episodes) == 1
    updated_ep = captured_episodes[0][0]
    np.testing.assert_array_equal(updated_ep.step_rewards, original_rewards)
    assert updated_ep.reward == original_reward


# ─── Completion criterion ─────────────────────────────────────────────────────

def test_training_halts_when_all_scores_above_threshold(tmp_path):
    """Training should halt when all episodes in a batch score > 0.90."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, state)

    # Episode where mana score will be high (all lands are islands = pure blue)
    # We override mana scorer to return a perfect score
    ep = _make_success_episode()

    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.95
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo = MagicMock()
    mock_ppo.update.return_value = mock_ppo_result

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    # Patch mana scorer to return score=0.95 (reward=0.9) for any deck
    from sealed.domain.mana_scorer import ManaScore
    perfect_score = ManaScore(score=0.95, reward=0.9, l1_error=0.0, n_lands=17)

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=perfect_score),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
    ):
        # Should terminate without _StopAfterBatch since completion triggers a return
        use_case = TrainStage2UseCase()
        use_case.execute(
            pools_path=pools_path,
            cards_path=cards_path,
            model_path=model_path,
            init_from=init_from,
            batch_size=1,
        )
    # If we reach here, training halted normally (no infinite loop)


def test_training_continues_when_duplicate_in_batch(tmp_path):
    """Training should NOT halt if any episode in the batch is a duplicate."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, state)

    dup_ep = _make_duplicate_episode()
    batch_count = [0]

    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.95
    mock_ppo_result.episode_runs = [4]
    mock_ppo = MagicMock()
    mock_ppo.update.return_value = mock_ppo_result

    mock_runner = MagicMock()
    mock_runner.run.return_value = dup_ep

    from sealed.domain.mana_scorer import ManaScore
    perfect_score = ManaScore(score=0.95, reward=0.9, l1_error=0.0, n_lands=17)

    def save_and_count(*args, **kwargs):
        batch_count[0] += 1
        if batch_count[0] >= 2:
            raise _StopAfterBatch()

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=perfect_score),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", save_and_count),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    # Should have run at least 2 batches (didn't halt after first duplicate)
    assert batch_count[0] >= 2


# ─── Batch logging format ─────────────────────────────────────────────────────

def test_batch_log_contains_expected_fields(tmp_path, capsys):
    """Stdout should contain batch scores, mean_score, dup count, and timing."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, state)

    ep = _make_success_episode()

    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo = MagicMock()
    mock_ppo.update.return_value = mock_ppo_result

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.8, reward=0.6, l1_error=1.0, n_lands=17)

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=score_val),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    out = capsys.readouterr().out
    # Should contain batch scores, mean_score, dup count, timing
    assert "batch scores:" in out
    assert "mean_score=" in out
    assert "dup=" in out


# ─── Card encoder frozen (FR-002) ────────────────────────────────────────────

def test_card_encoder_parameters_not_in_optimizer(tmp_path):
    """Optimizer should only contain PoolTransformerModel parameters (card encoder not trained)."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, state)

    ep = _make_success_episode()
    captured_optimizers: list = []

    mock_ppo_class = MagicMock()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo_instance = MagicMock()
    mock_ppo_instance.update.return_value = mock_ppo_result

    def capture_ppo(model_arg, optimizer_arg, **kwargs):
        captured_optimizers.append(optimizer_arg)
        return mock_ppo_instance

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.5, reward=0.0, l1_error=4.25, n_lands=17)

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", side_effect=capture_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=score_val),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    assert len(captured_optimizers) == 1
    optimizer_used = captured_optimizers[0]
    # The optimizer should only contain pool transformer model parameters
    # (i.e. no extra params from a card encoder)
    param_count = sum(
        len(pg["params"]) for pg in optimizer_used.param_groups
    )
    model_param_count = sum(1 for _ in PoolTransformerModel(MINI).parameters())
    assert param_count == model_param_count, (
        f"Optimizer has {param_count} params but PoolTransformerModel has {model_param_count}. "
        "Card encoder should not be in the optimizer."
    )


# ─── T028: Checkpoint resume behavior ────────────────────────────────────────

def test_model_path_takes_priority_over_init_from(tmp_path):
    """If model_path exists, load from it (ignore init_from)."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    # Create BOTH checkpoints
    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    s1_state = TrainingState(best_run=MAX_PICKS, episode_count=0)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, s1_state)

    s2_state = TrainingState(best_run=MAX_PICKS, episode_count=999)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(model_path, model, optimizer, s2_state)

    saved_states: list[TrainingState] = []
    original_save = PoolModelStore.save

    def capture_save(self, path, mdl, opt, ts):
        saved_states.append(ts)
        original_save(self, path, mdl, opt, ts)
        raise _StopAfterBatch()

    ep = _make_success_episode()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo = MagicMock()
    mock_ppo.update.return_value = mock_ppo_result
    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.5, reward=0.0, l1_error=0.0, n_lands=17)

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=score_val),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", capture_save),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    # Should have resumed from episode_count=999 (model_path), not 0 (init_from)
    assert len(saved_states) >= 1
    assert saved_states[0].episode_count > 999


def test_init_from_used_when_model_path_absent(tmp_path):
    """If model_path doesn't exist but init_from does, use init_from."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    # Only create init_from
    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    s1_state = TrainingState(best_run=MAX_PICKS, episode_count=500)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer, s1_state)

    saved_states: list[TrainingState] = []
    original_save = PoolModelStore.save

    def capture_save(self, path, mdl, opt, ts):
        saved_states.append(ts)
        original_save(self, path, mdl, opt, ts)
        raise _StopAfterBatch()

    ep = _make_success_episode()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo = MagicMock()
    mock_ppo.update.return_value = mock_ppo_result
    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.5, reward=0.0, l1_error=0.0, n_lands=17)

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=score_val),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", capture_save),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    # episode_count should start at 0 (fresh from init_from, not resuming from 500)
    assert len(saved_states) >= 1
    assert saved_states[0].episode_count > 0  # at least 1 episode ran
    assert saved_states[0].episode_count <= 5  # but nowhere near 500


def test_error_when_neither_checkpoint_exists(tmp_path):
    """Should raise ValueError when neither model_path nor init_from exists."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    use_case = TrainStage2UseCase()
    with pytest.raises((ValueError, FileNotFoundError)):
        with patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI):
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )


def test_init_from_uses_model_weights_only_fresh_optimizer(tmp_path):
    """init_from should load model weights only; optimizer should be fresh."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer_with_history = optim.Adam(model.parameters(), lr=1e-3)
    # Simulate optimizer having state (momentum etc.) from Stage 1 training
    s1_state = TrainingState(best_run=MAX_PICKS, episode_count=100)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(init_from, model, optimizer_with_history, s1_state)

    captured_optimizers: list = []

    def capture_ppo(model_arg, optimizer_arg, **kwargs):
        captured_optimizers.append(optimizer_arg)
        mock_inst = MagicMock()
        result = MagicMock()
        result.mean_reward = 0.0
        result.episode_runs = [MAX_PICKS]
        mock_inst.update.return_value = result
        return mock_inst

    ep = _make_success_episode()
    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.5, reward=0.0, l1_error=0.0, n_lands=17)

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", side_effect=capture_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=score_val),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    assert len(captured_optimizers) == 1
    opt = captured_optimizers[0]
    # Fresh optimizer should have no state (no momentum/adam state built up)
    assert len(opt.state) == 0, "Optimizer should be fresh (no state from init_from)"


# ─── T029: Timestamped checkpoint every 1000 episodes ─────────────────────────

def test_timestamped_checkpoint_saved_every_1000_episodes(tmp_path):
    """save_timestamped should be called when episode_count is a multiple of 1000."""
    pools_path = tmp_path / "pools"
    cards_path = tmp_path / "cards"
    model_path = tmp_path / "models" / "stage2" / "latest.pt"
    init_from = tmp_path / "models" / "stage1" / "latest.pt"

    _setup_pools(pools_path)
    _setup_cards(cards_path)

    model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters())
    # Start near the 1000-episode boundary
    state = TrainingState(best_run=MAX_PICKS, episode_count=999)
    init_from.parent.mkdir(parents=True, exist_ok=True)
    # Save as model_path (resume from it)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    PoolModelStore().save(model_path, model, optimizer, state)

    timestamped_calls = [0]
    original_save_ts = PoolModelStore.save_timestamped

    def count_timestamped(self, *args, **kwargs):
        timestamped_calls[0] += 1
        return original_save_ts(self, *args, **kwargs)

    ep = _make_success_episode()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo = MagicMock()
    mock_ppo.update.return_value = mock_ppo_result
    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.5, reward=0.0, l1_error=0.0, n_lands=17)

    batch_count = [0]
    original_save = PoolModelStore.save

    def save_and_stop(self, path, mdl, opt, ts):
        original_save(self, path, mdl, opt, ts)
        batch_count[0] += 1
        if batch_count[0] >= 2:
            raise _StopAfterBatch()

    with (
        patch("sealed.application.train_stage2.PoolTransformerConfig", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2.compute_mana_score", return_value=score_val),
        patch("sealed.application.train_stage2.count_pips", return_value=MagicMock(counts={})),
        patch("sealed.application.train_stage2.compute_ideal_distribution", return_value=MagicMock(ideal={})),
        patch("sealed.application.train_stage2.count_actual_sources", return_value=MagicMock(sources={})),
        patch.object(PoolModelStore, "save", save_and_stop),
        patch.object(PoolModelStore, "save_timestamped", count_timestamped),
    ):
        with pytest.raises(_StopAfterBatch):
            use_case = TrainStage2UseCase()
            use_case.execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    # With episode_count starting at 999 and batch_size=1, first batch → episode_count=1000
    # That should trigger a timestamped save
    assert timestamped_calls[0] >= 1, "Expected at least one timestamped checkpoint save"
