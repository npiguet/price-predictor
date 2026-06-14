"""GAE(λ) over a terminal-reward trajectory (spec 020 FR-011; research D2)."""

from __future__ import annotations

import numpy as np

from draft.application.rl_advantage import gae_advantages


def test_lambda_one_is_monte_carlo_advantage() -> None:
    # λ=1, γ=1 ⇒ A_t = R − V(s_t) for every pick (telescoping).
    values = np.array([0.2, -0.1, 0.5, 0.0], dtype=np.float64)
    reward = 1.0
    adv = gae_advantages(values, reward, gae_lambda=1.0)
    np.testing.assert_allclose(adv, reward - values, rtol=0, atol=1e-9)


def test_lambda_zero_is_one_step_td() -> None:
    # λ=0 ⇒ A_t = δ_t = γ·V(s_{t+1}) − V(s_t) (non-terminal), R − V(s_T) (terminal).
    values = np.array([0.2, -0.1, 0.5], dtype=np.float64)
    reward = 2.0
    adv = gae_advantages(values, reward, gae_lambda=0.0)
    expected = np.array([
        values[1] - values[0],
        values[2] - values[1],
        reward - values[2],
    ])
    np.testing.assert_allclose(adv, expected, atol=1e-9)


def test_intermediate_lambda_between_extremes() -> None:
    values = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    reward = 1.0
    a_td = gae_advantages(values, reward, gae_lambda=0.0)
    a_mc = gae_advantages(values, reward, gae_lambda=1.0)
    a_mid = gae_advantages(values, reward, gae_lambda=0.5)
    # With all-zero values, earlier picks' advantage grows toward MC as λ rises.
    assert a_td[0] <= a_mid[0] <= a_mc[0]
    assert a_mid[-1] == a_td[-1] == a_mc[-1]  # terminal pick identical for any λ


def test_terminal_bootstrap_has_no_next_value() -> None:
    # Terminal δ uses R − V(s_T) only (no V beyond the trajectory).
    values = np.array([5.0], dtype=np.float64)
    adv = gae_advantages(values, reward=3.0, gae_lambda=0.95)
    np.testing.assert_allclose(adv, np.array([3.0 - 5.0]), atol=1e-9)


def test_empty_trajectory_returns_empty() -> None:
    adv = gae_advantages(np.zeros(0), reward=1.0, gae_lambda=0.95)
    assert adv.shape == (0,)
