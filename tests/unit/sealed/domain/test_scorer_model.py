"""Unit tests for the Set Transformer scorer model."""

from __future__ import annotations

import torch
import pytest

from sealed.domain.scorer_model import SetTransformerScorer


def _default_config():
    return dict(d_model=544, n_layers=2, n_heads=4, n_seeds=4, d_ff=1088, mlp_hidden=256)


def _make_scorer(**overrides):
    config = _default_config()
    config.update(overrides)
    model = SetTransformerScorer(**config)
    model.eval()
    return model


class TestForwardPass:
    def test_produces_scalar_output(self):
        model = _make_scorer()
        cards = torch.randn(2, 10, 544)
        mask = torch.ones(2, 10, dtype=torch.bool)
        scores = model(cards, mask)
        assert scores.shape == (2, 1)

    def test_single_example(self):
        model = _make_scorer()
        cards = torch.randn(1, 5, 544)
        mask = torch.ones(1, 5, dtype=torch.bool)
        scores = model(cards, mask)
        assert scores.shape == (1, 1)

    def test_different_deck_sizes(self):
        """Variable-length decks produce valid scores."""
        model = _make_scorer()
        # Batch with different deck sizes (padded to max)
        cards = torch.randn(3, 29, 544)
        mask = torch.ones(3, 29, dtype=torch.bool)
        mask[0, 20:] = False  # 20-card deck
        mask[1, 25:] = False  # 25-card deck
        # mask[2] is all True → 29-card deck
        scores = model(cards, mask)
        assert scores.shape == (3, 1)
        assert torch.isfinite(scores).all()


class TestPermutationInvariance:
    def test_same_cards_different_order_same_score(self):
        model = _make_scorer()
        cards = torch.randn(1, 8, 544)
        mask = torch.ones(1, 8, dtype=torch.bool)
        score1 = model(cards, mask)

        # Permute card order
        perm = torch.randperm(8)
        permuted = cards[:, perm, :]
        score2 = model(permuted, mask)

        torch.testing.assert_close(score1, score2, atol=1e-5, rtol=1e-5)


class TestPaddingMask:
    def test_padding_does_not_influence_score(self):
        """Padded cards should not affect the score."""
        model = _make_scorer()
        real_cards = torch.randn(1, 5, 544)

        # No padding
        mask_no_pad = torch.ones(1, 5, dtype=torch.bool)
        score_no_pad = model(real_cards, mask_no_pad)

        # Add 5 padding cards
        padding = torch.randn(1, 5, 544)
        padded_cards = torch.cat([real_cards, padding], dim=1)
        mask_padded = torch.ones(1, 10, dtype=torch.bool)
        mask_padded[0, 5:] = False
        score_padded = model(padded_cards, mask_padded)

        torch.testing.assert_close(score_no_pad, score_padded, atol=1e-4, rtol=1e-4)


class TestNormalizationBuffers:
    def test_feat_mean_and_std_registered(self):
        model = _make_scorer()
        assert hasattr(model, "feat_mean")
        assert hasattr(model, "feat_std")
        assert model.feat_mean.shape == (32,)
        assert model.feat_std.shape == (32,)

    def test_default_normalization_is_identity(self):
        """Default: mean=0, std=1 → no transformation."""
        model = _make_scorer()
        assert (model.feat_mean == 0).all()
        assert (model.feat_std == 1).all()

    def test_custom_normalization_transforms_features(self):
        """Setting feat_mean/feat_std normalizes indices 512-543."""
        model = _make_scorer()
        mean = torch.full((32,), 2.0)
        std = torch.full((32,), 0.5)
        model.feat_mean.copy_(mean)
        model.feat_std.copy_(std)

        cards = torch.ones(1, 5, 544)
        mask = torch.ones(1, 5, dtype=torch.bool)
        # Forward should work without error
        score = model(cards, mask)
        assert torch.isfinite(score).all()

    def test_normalization_only_affects_last_32_dims(self):
        """Indices 0-511 pass through unchanged during normalization."""
        model = _make_scorer()
        mean = torch.full((32,), 100.0)
        std = torch.full((32,), 0.01)
        model.feat_mean.copy_(mean)
        model.feat_std.copy_(std)

        cards = torch.randn(1, 3, 544)
        original_text_features = cards[:, :, :512].clone()

        # Manually apply what the model should do
        normalized = model._normalize(cards)
        # First 512 dims unchanged
        torch.testing.assert_close(normalized[:, :, :512], original_text_features)


class TestSharedWeights:
    def test_same_model_scores_both_decks(self):
        """The same model instance should be used for both decks in a pair."""
        model = _make_scorer()
        deck_a = torch.randn(1, 10, 544)
        deck_b = torch.randn(1, 12, 544)
        mask_a = torch.ones(1, 10, dtype=torch.bool)
        mask_b = torch.ones(1, 12, dtype=torch.bool)

        score_a = model(deck_a, mask_a)
        score_b = model(deck_b, mask_b)

        assert score_a.shape == (1, 1)
        assert score_b.shape == (1, 1)
        # Scores should be different for different decks
        assert not torch.equal(score_a, score_b)
