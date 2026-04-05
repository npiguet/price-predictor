"""T008 — Unit tests for EpisodeRunner."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel
from sealed.domain.episode_runner import EpisodeRunner, _build_base_tensor, MAX_PICKS

# Miniaturized config: 4 booster slots, no basic lands (n_basic=0)
MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=16,  # 8 embed + 8 padding
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)

POOL_NAMES = "Card1;Card2;Card3;Card4"


class _MockCardPort:
    """Returns a unique embedding per card name (deterministic)."""

    def get_embedding(self, card_name: str) -> np.ndarray:
        # Use hash to produce repeatable but distinct vectors
        rng = np.random.default_rng(abs(hash(card_name)) % (2**31))
        return rng.random(8).astype(np.float32)

    def is_land(self, card_name: str) -> bool:
        return False


def _make_model():
    model = PoolTransformerModel(MINI)
    return model


# ── _build_base_tensor ────────────────────────────────────────────────────────

def test_build_base_tensor_shape():
    port = _MockCardPort()
    t = _build_base_tensor(POOL_NAMES, port, n_slots=4)
    assert t.shape == (4, 16), f"Expected (4,16), got {t.shape}"


def test_build_base_tensor_booster_available_flag():
    port = _MockCardPort()
    t = _build_base_tensor(POOL_NAMES, port, n_slots=4)
    embed_dim = 8
    for i in range(4):
        assert t[i, embed_dim + 1] == 1.0, "available_flag should be 1 for booster slots"


def test_build_base_tensor_booster_flags_picked_and_land_zero():
    port = _MockCardPort()
    t = _build_base_tensor(POOL_NAMES, port, n_slots=4)
    embed_dim = 8
    for i in range(4):
        assert t[i, embed_dim] == 0.0     # pick_count
        assert t[i, embed_dim + 2] == 0.0  # reserved


# ── EpisodeRunner ─────────────────────────────────────────────────────────────

def test_episode_always_completes_all_picks():
    """With action masking, every episode completes all best_run steps."""
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()

    for seed in range(10):
        ep = runner.run(POOL_NAMES, port, model, rng_seed=seed, best_run=4)
        assert len(ep.actions) == 4, f"Expected 4 picks, got {len(ep.actions)} (seed={seed})"


def test_masking_prevents_duplicate_picks():
    """Even with a model strongly biased toward slot 0, all picks must be unique."""
    model = _make_model()
    # Override head weights so slot 0 always has the highest logit — without masking
    # this would immediately cause a duplicate on step 2.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[0] = 100.0  # slot 0 always wins
    runner = EpisodeRunner()
    port = _MockCardPort()

    for seed in range(5):
        ep = runner.run(POOL_NAMES, port, model, rng_seed=seed, best_run=4)
        assert len(ep.actions) == 4, "Masking should ensure all 4 picks complete"
        assert len(set(ep.actions)) == 4, "All 4 picks must be unique pool indices"


def test_reward_formula():
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()
    ep = runner.run(POOL_NAMES, port, model, rng_seed=7, best_run=2)
    expected_reward = len(ep.actions) / 2.0 * 2.0 - 1.0
    assert abs(ep.reward - expected_reward) < 1e-6


def test_actions_are_valid_pool_indices():
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()
    ep = runner.run(POOL_NAMES, port, model, rng_seed=1)
    n_booster = 4
    for action in ep.actions:
        assert 0 <= action < n_booster, f"Pool index {action} out of range [0,{n_booster})"


def test_log_probs_have_same_length_as_actions():
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()
    ep = runner.run(POOL_NAMES, port, model, rng_seed=5)
    assert len(ep.log_probs) == len(ep.actions)


def test_actions_are_always_unique():
    """With masking, actions are always a permutation of pool indices."""
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()
    for seed in range(20):
        ep = runner.run(POOL_NAMES, port, model, rng_seed=seed, best_run=4)
        assert len(ep.actions) == 4
        assert sorted(ep.actions) == [0, 1, 2, 3], f"Expected all 4 unique indices (seed={seed})"


def test_shuffle_seeds_shape():
    """shuffle_seeds always has shape (best_run,)."""
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()
    # Use best_run=4 (== n_booster) so we don't exhaust all available slots
    ep = runner.run(POOL_NAMES, port, model, rng_seed=3, best_run=4)
    assert ep.shuffle_seeds.shape == (4,)


def test_different_pool_indices_at_same_shuffled_position_are_both_legal():
    """
    Two picks that happen to occupy shuffled position 0 at different steps
    are legal as long as they resolve to different pool indices.
    """
    model = _make_model()
    runner = EpisodeRunner()
    port = _MockCardPort()
    # With masking all episodes complete with 4 unique picks
    for seed in range(20):
        ep = runner.run(POOL_NAMES, port, model, rng_seed=seed, best_run=4)
        assert len(set(ep.actions)) == 4
