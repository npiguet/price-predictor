"""Noise on ``e`` (FR-056, T089): the covariance, the sigma schedule and the ramp."""

from __future__ import annotations

import pytest
import torch

from effects.application.surface_batching import NoiseState

# ── noise on e (FR-056, T089) ────────────────────────────────────────────

def _spread_matrix(rows: int = 4000, dim: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    scale = torch.tensor([3.0, 1.0, 0.25])[:dim]
    return torch.randn(rows, dim, generator=generator) * scale


def test_the_added_noise_has_covariance_r_squared_sigma():
    torch.manual_seed(0)
    matrix = _spread_matrix()
    noise = NoiseState(ratio=0.5, ramp_steps=10)
    added = noise.apply(matrix, step=10) - matrix
    # Sigma is held in float64 so the factorization stays exact.
    expected = 0.25 * noise.sigma.float()
    observed = torch.cov(added.T)
    assert torch.allclose(observed, expected, atol=0.05 * expected.abs().max())


def test_sigma_starts_from_the_first_batch_and_decays_at_0_99():
    noise = NoiseState(ratio=0.1, ramp_steps=1)
    first = _spread_matrix()
    noise.update(first)
    assert torch.allclose(noise.sigma.float(), torch.cov(first.T), atol=1e-4)
    second = first * 2.0
    noise.update(second)
    expected = 0.99 * torch.cov(first.T) + 0.01 * torch.cov(second.T)
    assert torch.allclose(noise.sigma.float(), expected, atol=1e-4)


def test_no_gradient_flows_through_sigma():
    matrix = _spread_matrix(rows=50).requires_grad_(True)
    noise = NoiseState(ratio=0.2, ramp_steps=1)
    noise.apply(matrix, step=1).sum().backward()
    assert not noise.sigma.requires_grad
    # The noise is additive and detached: the gradient is the identity's.
    assert torch.equal(matrix.grad, torch.ones_like(matrix))


def test_the_ratio_ramps_linearly_over_the_first_epoch():
    noise = NoiseState(ratio=0.1, ramp_steps=5000)
    assert noise.scale(0) == 0.0
    assert noise.scale(2500) == pytest.approx(0.05)
    assert noise.scale(9000) == pytest.approx(0.1)


def test_at_step_zero_nothing_is_added():
    matrix = _spread_matrix(rows=20)
    noise = NoiseState(ratio=0.1, ramp_steps=100)
    assert torch.equal(noise.apply(matrix, step=0), matrix)


def test_a_resume_file_holding_the_old_step_key_still_loads():
    """Resume files written before the state dropped its unread ``step`` carry
    it; an interrupted run must continue across the change."""
    noise = NoiseState(ratio=0.1, ramp_steps=1)
    noise.apply(_spread_matrix(rows=20), step=1)
    saved = noise.state_dict()
    restored = NoiseState(ratio=0.1, ramp_steps=1)
    restored.load_state_dict({**saved, "step": 7}, torch.device("cpu"))
    assert torch.equal(restored.sigma, noise.sigma)
    assert torch.equal(restored._factor, noise._factor)
    assert restored._draws_since_factor == noise._draws_since_factor


def test_a_scoring_batcher_carries_no_noise(tmp_path):
    """Validation, evaluation and encoding never perturb e (scenario 4): the
    materializer a scoring batch is built with has none unless handed one."""
    from effects.application.surface_batching import Materializer
    from effects.domain.baselines import baseline_masks

    assert Materializer(torch.device("cpu"), baseline_masks("full")).noise is None


def test_a_rank_deficient_batch_under_bf16_autocast_still_draws_noise():
    """The training condition that broke the Cholesky factor.

    Training runs under bf16 autocast, and a batch often carries fewer unique
    texts than ``e`` has dimensions, so the batch covariance is singular. The
    covariance and the draw must leave autocast and stay exact, or the factor
    sees a matrix that is not positive semi-definite.
    """
    generator = torch.Generator().manual_seed(5)
    dim = 64
    # Twelve texts in a 64-wide space, at the scale a trained e reaches.
    matrix = torch.randn(12, dim, generator=generator) * 40.0
    noise = NoiseState(ratio=0.1, ramp_steps=1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        for _ in range(20):
            out = noise.apply(matrix, step=1)
    assert torch.isfinite(out).all()
    assert out.dtype == matrix.dtype
    assert torch.allclose(noise.sigma, noise.sigma.T)
