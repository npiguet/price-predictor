"""T009 — Unit tests for PPOTrainer."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.optim as optim

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.replay_buffer import Episode
from sealed.domain.ppo_trainer import PPOTrainer, TrainBatchResult
from sealed.domain.episode_runner import EpisodeRunner, MAX_PICKS

MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=12,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)

POOL_NAMES = "Card1;Card2;Card3;Card4"


class _MockCardPort:
    def get_embedding(self, card_name: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(card_name)) % (2**31))
        return rng.random(8).astype(np.float32)

    def is_land(self, card_name: str) -> bool:
        return False


def _make_real_episode(model=None, seed=42) -> Episode:
    """Generate a real episode using EpisodeRunner."""
    if model is None:
        model = PoolTransformerModel(MINI)
    runner = EpisodeRunner()
    port = _MockCardPort()
    return runner.run(POOL_NAMES, port, model, rng_seed=seed)


def _make_trainer(model=None):
    if model is None:
        model = PoolTransformerModel(MINI)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer, PPOTrainer(model, optimizer)


# ── TrainBatchResult fields ────────────────────────────────────────────────────

def test_train_batch_result_fields_populated():
    model, optimizer, trainer = _make_trainer()
    port = _MockCardPort()
    episodes = [_make_real_episode(model, seed=i) for i in range(3)]
    result = trainer.update(episodes, port)

    assert isinstance(result, TrainBatchResult)
    assert isinstance(result.mean_reward, float)
    assert isinstance(result.episode_runs, list)
    assert len(result.episode_runs) == 3


def test_episode_runs_lengths_match_actions():
    model, optimizer, trainer = _make_trainer()
    port = _MockCardPort()
    episodes = [_make_real_episode(model, seed=i) for i in range(4)]
    result = trainer.update(episodes, port)
    for i, ep in enumerate(episodes):
        assert result.episode_runs[i] == len(ep.actions)


# ── Gradient update ────────────────────────────────────────────────────────────

def test_ppo_loss_backward_produces_nonzero_gradients():
    model, optimizer, trainer = _make_trainer()
    port = _MockCardPort()
    episodes = [_make_real_episode(model, seed=i) for i in range(2)]
    episodes = [ep for ep in episodes if len(ep.actions) > 0]
    assert len(episodes) > 0

    trainer.update(episodes, port)
    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad, "PPO backward pass should produce non-zero gradients"


# ── Empty episodes ─────────────────────────────────────────────────────────────

def test_update_with_empty_episodes_list():
    model, optimizer, trainer = _make_trainer()
    port = _MockCardPort()
    result = trainer.update([], port)
    assert result.mean_reward == 0.0
    assert result.episode_runs == []
