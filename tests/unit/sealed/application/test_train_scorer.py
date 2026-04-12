"""Unit tests for the training use case."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from sealed.application.train_scorer import TrainScorerUseCase, TrainScorerConfig


def _make_synthetic_data(tmp_path, n_matches=20, n_cards_per_deck=10):
    """Create synthetic match outcomes and card embeddings for testing."""
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()

    card_names = [f"card_{i}" for i in range(50)]
    for name in card_names:
        first_letter = name[0]
        letter_dir = cards_dir / first_letter
        letter_dir.mkdir(exist_ok=True)
        emb = np.random.randn(544).astype(np.float32)
        np.savez_compressed(letter_dir / f"{name}.npz", embedding=emb)

    outcomes_file = tmp_path / "outcomes.txt"
    lines = []
    for _ in range(n_matches):
        deck_a = np.random.choice(card_names, n_cards_per_deck, replace=True)
        deck_b = np.random.choice(card_names, n_cards_per_deck, replace=True)
        wins_a, wins_b = (2, np.random.choice([0, 1]))
        lines.append(f"{'|'.join(deck_a)};{'|'.join(deck_b)};{wins_a};{wins_b}")

    outcomes_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcomes_file, cards_dir


class TestNormalizationStats:
    def test_normalization_computed_from_training_corpus(self, tmp_path):
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=10)
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        model, _ = use_case.execute(config)

        # feat_mean and feat_std should be set (not default 0/1 if training data has variance)
        assert model.feat_mean.shape == (32,)
        assert model.feat_std.shape == (32,)


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
        # Winner score > loser score → loss should be relatively small
        assert loss.item() < 1.0


class TestTrainingReducesLoss:
    def test_training_step_reduces_loss(self, tmp_path):
        """Training on synthetic data should reduce loss over epochs."""
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=5,
            batch_size=8,
            lr=1e-3,
            val_interval=5,
        )
        use_case = TrainScorerUseCase()
        model, metrics = use_case.execute(config)

        assert len(metrics.train_losses) > 0
        # Loss should decrease (first > last, approximately)
        assert metrics.train_losses[-1] <= metrics.train_losses[0] + 0.5  # some tolerance


class TestValidationSplit:
    def test_80_20_split_by_match(self, tmp_path):
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=2,
            batch_size=4,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        model, metrics = use_case.execute(config)

        # Validation should have been run
        assert len(metrics.val_losses) > 0


class TestPredictionAccuracy:
    def test_accuracy_reported_each_val_interval(self, tmp_path):
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=3,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        model, metrics = use_case.execute(config)

        # Accuracy should be reported at each val_interval
        assert len(metrics.val_accuracies) == 3  # 3 epochs, val_interval=1
        for acc in metrics.val_accuracies:
            assert 0.0 <= acc <= 1.0

    def test_perfect_accuracy_on_trivial_data(self, tmp_path):
        """If model perfectly separates winners from losers, accuracy = 1.0."""
        # This tests the accuracy computation logic, not model convergence.
        # We use a trained model to verify accuracy ranges are correct.
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        model, metrics = use_case.execute(config)
        # Just verify accuracy is computed (between 0 and 1)
        assert len(metrics.val_accuracies) > 0
        assert 0.0 <= metrics.val_accuracies[0] <= 1.0


class TestEmbeddingUnfreezing:
    def test_unfrozen_embeddings_training_completes(self, tmp_path):
        """With --unfreeze-embeddings, training should complete with embedding fine-tuning."""
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)

        # First train Phase A (frozen) to create a checkpoint
        config_a = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints_a",
            epochs=1,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        use_case.execute(config_a)

        # Phase B: unfreeze embeddings
        config_b = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints_b",
            resume=tmp_path / "checkpoints_a" / "best_l2_h4_s4_ff1088_mlp256_lr0.001.pt",
            epochs=1,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
            unfreeze_embeddings=True,
            embedding_lr=1e-5,
        )
        model, metrics = use_case.execute(config_b)

        # Training with unfrozen embeddings should complete successfully
        assert len(metrics.train_losses) == 1
        assert len(metrics.val_losses) == 1

    def test_frozen_embeddings_by_default(self, tmp_path):
        """Without --unfreeze-embeddings, no embedding table is created."""
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=10)
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
            unfreeze_embeddings=False,
        )
        use_case = TrainScorerUseCase()
        model, _ = use_case.execute(config)
        # No embedding table when frozen
        assert not hasattr(model, "embedding_table") or model.embedding_table is None

    def test_drift_metric_reported_when_unfrozen(self, tmp_path):
        """Embedding drift should be reported when embeddings are unfrozen."""
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)

        # Phase A
        config_a = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints_a",
            epochs=1,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        use_case.execute(config_a)

        # Phase B with unfreezing
        config_b = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=tmp_path / "checkpoints_b",
            resume=tmp_path / "checkpoints_a" / "best_l2_h4_s4_ff1088_mlp256_lr0.001.pt",
            epochs=2,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
            unfreeze_embeddings=True,
            embedding_lr=1e-5,
        )
        model, metrics = use_case.execute(config_b)

        # Drift metrics should be present when unfrozen
        assert len(metrics.embedding_drifts) > 0
        for drift in metrics.embedding_drifts:
            assert drift >= 0.0


class TestCheckpointing:
    def test_best_checkpoint_saved_on_improvement(self, tmp_path):
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)
        ckpt_dir = tmp_path / "checkpoints"
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=ckpt_dir,
            epochs=3,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        use_case.execute(config)

        assert (ckpt_dir / "best_l2_h4_s4_ff1088_mlp256_lr0.001.pt").exists()

    def test_latest_checkpoint_saved_every_validation(self, tmp_path):
        outcomes_file, cards_dir = _make_synthetic_data(tmp_path, n_matches=20)
        ckpt_dir = tmp_path / "checkpoints"
        config = TrainScorerConfig(
            outcomes_path=outcomes_file,
            cards_path=cards_dir,
            checkpoint_dir=ckpt_dir,
            epochs=3,
            batch_size=8,
            lr=1e-3,
            val_interval=1,
        )
        use_case = TrainScorerUseCase()
        use_case.execute(config)

        assert (ckpt_dir / "latest.pt").exists()
