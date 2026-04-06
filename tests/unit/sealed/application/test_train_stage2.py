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
    """Minimal completed episode (MAX_PICKS=40 picks).

    Picks: slot 0 (spell), then alternates slot 0 (spell) and slot 4
    (Plains basic land, first slot after booster) so that both pip demand
    and mana supply exist — required for non-zero shaping.
    """
    n_picks = MAX_PICKS
    n_booster = len(pool_names.split(";"))
    # spell, land, spell, land, ... so shaping can activate
    actions = [(0 if t % 2 == 0 else n_booster) for t in range(n_picks)]
    return Episode(
        pool_names=pool_names,
        shuffle_seeds=np.zeros(n_picks, dtype=np.int64),
        actions=np.array(actions, dtype=np.int32),
        log_probs=np.zeros(n_picks, dtype=np.float32),
        step_rewards=np.ones(n_picks, dtype=np.float32),
        reward=1.0,
        effective_run=n_picks,
    )


class _StopAfterBatch(Exception):
    pass


# ─── Reward override ──────────────────────────────────────────────────────────

def test_completed_episode_step_rewards_overridden_with_per_step_rewards(tmp_path):
    """Completed episodes should have step_rewards replaced with per-step reward values."""
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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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

    # The episode passed to update should have per-step rewards from compute_per_step_rewards()
    assert len(captured_episodes) == 1
    updated_ep = captured_episodes[0][0]
    # step_rewards should have been replaced (differ from the initial all-ones array)
    assert not np.all(updated_ep.step_rewards == 1.0), (
        "step_rewards should have been overridden (not left as all-ones)"
    )
    # ep.reward should hold the mana score (for convergence checking and logging)
    assert isinstance(updated_ep.reward, float)


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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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
    # Should contain batch scores, mean_score, and timing
    assert "batch scores:" in out
    assert "mean_score=" in out


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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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
        with patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI):
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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
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


# ─── T005: Per-step reward integration ───────────────────────────────────────

def _make_train_env(tmp_path):
    """Reusable setup: pools, cards, stage1 checkpoint."""
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

    return pools_path, cards_path, model_path, init_from


def test_step_rewards_are_per_step_not_uniform(tmp_path):
    """T005: step_rewards on episodes must be per-step (not all identical after T007)."""
    pools_path, cards_path, model_path, init_from = _make_train_env(tmp_path)

    ep = _make_success_episode()
    captured_episodes: list[list[Episode]] = []

    mock_ppo = MagicMock()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo.update.side_effect = lambda episodes, _port: (
        captured_episodes.append(list(episodes)) or mock_ppo_result
    )

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    with (
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            TrainStage2UseCase().execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    assert len(captured_episodes) == 1
    updated_ep = captured_episodes[0][0]
    # Per-step rewards: should have more than one distinct value (NOT uniform)
    unique_vals = set(round(float(v), 6) for v in updated_ep.step_rewards)
    assert len(unique_vals) > 1, (
        "step_rewards should be per-step (distinct values), not uniform. "
        f"Got {len(unique_vals)} unique values: {sorted(unique_vals)[:5]}"
    )


def test_ep_reward_holds_mana_score_for_convergence_logging(tmp_path):
    """T005: ep.reward must still be the end-of-episode mana score (not step reward)."""
    pools_path, cards_path, model_path, init_from = _make_train_env(tmp_path)

    ep = _make_success_episode()
    captured_episodes: list[list[Episode]] = []

    mock_ppo = MagicMock()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo.update.side_effect = lambda episodes, _port: (
        captured_episodes.append(list(episodes)) or mock_ppo_result
    )

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    fixed_score = ManaScore(score=0.75, reward=0.5, l1_error=1.0, n_lands=17)

    with (
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2._compute_episode_mana_score",
              return_value=fixed_score),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            TrainStage2UseCase().execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    assert len(captured_episodes) == 1
    updated_ep = captured_episodes[0][0]
    # ep.reward must be the mana score reward (for convergence checking)
    assert updated_ep.reward == pytest.approx(fixed_score.reward), (
        f"ep.reward={updated_ep.reward} should equal mana score reward={fixed_score.reward}"
    )


# ─── T013: Batch log format with shaping and imbalance fields ─────────────────

def test_batch_log_includes_shaping_field(tmp_path, capsys):
    """T013: batch print line must include 'shaping=' field."""
    pools_path, cards_path, model_path, init_from = _make_train_env(tmp_path)

    ep = _make_success_episode()

    mock_ppo = MagicMock()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo.update.return_value = mock_ppo_result

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.8, reward=0.6, l1_error=1.0, n_lands=17)

    with (
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2._compute_episode_mana_score",
              return_value=score_val),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            TrainStage2UseCase().execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    out = capsys.readouterr().out
    assert "shaping=" in out, f"Expected 'shaping=' in batch log. Got:\n{out}"


def test_batch_log_includes_imbalance_field(tmp_path, capsys):
    """T013: batch print line must include 'imbalance=' field."""
    pools_path, cards_path, model_path, init_from = _make_train_env(tmp_path)

    ep = _make_success_episode()

    mock_ppo = MagicMock()
    mock_ppo_result = MagicMock()
    mock_ppo_result.mean_reward = 0.5
    mock_ppo_result.episode_runs = [MAX_PICKS]
    mock_ppo.update.return_value = mock_ppo_result

    mock_runner = MagicMock()
    mock_runner.run.return_value = ep

    from sealed.domain.mana_scorer import ManaScore
    score_val = ManaScore(score=0.8, reward=0.6, l1_error=1.0, n_lands=17)

    with (
        patch.object(PoolTransformerConfig, "from_embed_dim", return_value=MINI),
        patch("sealed.application.train_stage2.EpisodeRunner", return_value=mock_runner),
        patch("sealed.application.train_stage2.PPOTrainer", return_value=mock_ppo),
        patch("sealed.application.train_stage2._compute_episode_mana_score",
              return_value=score_val),
        patch.object(PoolModelStore, "save", side_effect=_StopAfterBatch),
    ):
        with pytest.raises(_StopAfterBatch):
            TrainStage2UseCase().execute(
                pools_path=pools_path,
                cards_path=cards_path,
                model_path=model_path,
                init_from=init_from,
                batch_size=1,
            )

    out = capsys.readouterr().out
    assert "imbalance=" in out, f"Expected 'imbalance=' in batch log. Got:\n{out}"
