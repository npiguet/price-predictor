"""Unit tests for the scorer's pairwise Bradley-Terry loss and its margin weights."""

from __future__ import annotations

import pytest
import torch

from sealed.application.scorer_pairwise_loss import _compute_margin_weights, _pairwise_bce


class TestBradleyTerryLoss:
    def test_bce_on_score_difference(self):
        """Loss = BCE(score_winner - score_loser, target=1)."""
        score_winner = torch.tensor([2.0])
        score_loser = torch.tensor([0.5])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            score_winner - score_loser,
            torch.ones_like(score_winner),
        )
        assert loss.item() > 0
        assert loss.item() < 1.0

    def test_pairwise_bce_unweighted_matches_reduction_mean(self):
        """weights=None preserves the pre-flag .mean() path bit-for-bit."""
        winner = torch.tensor([2.0, 1.5, -0.3])
        loser = torch.tensor([0.5, 1.0, 0.1])
        ours = _pairwise_bce(winner, loser)
        reference = torch.nn.functional.binary_cross_entropy_with_logits(
            winner - loser, torch.ones_like(winner),
        )
        assert torch.allclose(ours, reference)

    def test_pairwise_bce_linear_weights_scale_loss(self):
        """A single pair weighted by 4 produces 4x the loss of the same pair weighted by 1."""
        winner = torch.tensor([0.8])
        loser = torch.tensor([0.2])
        w1 = _pairwise_bce(winner, loser, torch.tensor([1.0]))
        w4 = _pairwise_bce(winner, loser, torch.tensor([4.0]))
        assert torch.allclose(w4, w1 * 4.0)

    def test_compute_margin_weights_modes(self):
        margins = torch.tensor([1, 2, 3, 4])
        assert _compute_margin_weights(margins, "none") is None
        linear = _compute_margin_weights(margins, "linear")
        assert torch.allclose(linear, torch.tensor([1.0, 2.0, 3.0, 4.0]))
        log = _compute_margin_weights(margins, "log")
        assert torch.allclose(log, torch.log(torch.tensor([2.0, 3.0, 4.0, 5.0])))

    def test_compute_margin_weights_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="margin_weighting"):
            _compute_margin_weights(torch.tensor([1]), "sqrt")
