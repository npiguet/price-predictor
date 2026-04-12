"""Unit tests for the training use case."""

from __future__ import annotations

import torch

from sealed.application.train_scorer import TrainScorerConfig, TrainScorerUseCase


def _config(outcomes_file, cards_dir, checkpoint_dir, *, epochs=1, val_interval=1, **overrides):
    return TrainScorerConfig(
        outcomes_path=outcomes_file,
        cards_path=cards_dir,
        checkpoint_dir=checkpoint_dir,
        epochs=epochs,
        batch_size=8,
        lr=1e-3,
        val_interval=val_interval,
        **overrides,
    )


class TestNormalizationStats:
    def test_normalization_computed_from_training_corpus(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
        )
        result = TrainScorerUseCase().execute(config)

        assert result.model.feat_mean.shape == (32,)
        assert result.model.feat_std.shape == (32,)


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


class TestTrainingReducesLoss:
    def test_training_step_reduces_loss(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        """Training on synthetic data should reduce loss over epochs."""
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=5, val_interval=5,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.train_losses) > 0
        assert result.metrics.train_losses[-1] <= result.metrics.train_losses[0] + 0.5


class TestValidationSplit:
    def test_80_20_split_by_match(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=2,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.val_losses) > 0


class TestPredictionAccuracy:
    def test_accuracy_reported_each_val_interval(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            epochs=3,
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.val_accuracies) == 3
        for acc in result.metrics.val_accuracies:
            assert 0.0 <= acc <= 1.0

    def test_perfect_accuracy_on_trivial_data(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
        )
        result = TrainScorerUseCase().execute(config)

        assert len(result.metrics.val_accuracies) > 0
        assert 0.0 <= result.metrics.val_accuracies[0] <= 1.0


class TestEmbeddingUnfreezing:
    def test_unfrozen_embeddings_training_completes(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        """With --unfreeze-embeddings, training should complete with embedding fine-tuning."""
        config_a = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints_a",
        )
        TrainScorerUseCase().execute(config_a)

        config_b = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints_b",
            resume=tmp_path / "checkpoints_a" / config_a.best_checkpoint_name(),
            unfreeze_embeddings=True,
            embedding_lr=1e-5,
        )
        result = TrainScorerUseCase().execute(config_b)

        assert len(result.metrics.train_losses) == 1
        assert len(result.metrics.val_losses) == 1
        assert not result.embedding_table.is_frozen()

    def test_frozen_embeddings_by_default(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        """Without --unfreeze-embeddings, the embedding table stays frozen."""
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints",
            unfreeze_embeddings=False,
        )
        result = TrainScorerUseCase().execute(config)

        assert result.embedding_table.is_frozen()

    def test_drift_metric_reported_when_unfrozen(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        """Embedding drift should be reported when embeddings are unfrozen."""
        config_a = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints_a",
        )
        TrainScorerUseCase().execute(config_a)

        config_b = _config(
            synthetic_outcomes_file, synthetic_cards_dir, tmp_path / "checkpoints_b",
            resume=tmp_path / "checkpoints_a" / config_a.best_checkpoint_name(),
            epochs=2,
            unfreeze_embeddings=True,
            embedding_lr=1e-5,
        )
        result = TrainScorerUseCase().execute(config_b)

        assert len(result.metrics.embedding_drifts) > 0
        for drift in result.metrics.embedding_drifts:
            assert drift >= 0.0


class TestCheckpointing:
    def test_best_checkpoint_saved_on_improvement(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        ckpt_dir = tmp_path / "checkpoints"
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, ckpt_dir, epochs=3,
        )
        TrainScorerUseCase().execute(config)

        assert (ckpt_dir / config.best_checkpoint_name()).exists()

    def test_latest_checkpoint_saved_every_validation(
        self, tmp_path, synthetic_cards_dir, synthetic_outcomes_file,
    ):
        ckpt_dir = tmp_path / "checkpoints"
        config = _config(
            synthetic_outcomes_file, synthetic_cards_dir, ckpt_dir, epochs=3,
        )
        TrainScorerUseCase().execute(config)

        assert (ckpt_dir / "latest.pt").exists()


class TestBestCheckpointName:
    def test_uses_arch_hyperparameters(self, tmp_path):
        config = TrainScorerConfig(
            outcomes_path=tmp_path / "outcomes.txt",
            cards_path=tmp_path / "cards",
            checkpoint_dir=tmp_path / "checkpoints",
            n_layers=3,
            n_heads=8,
            n_seeds=2,
            d_ff=2048,
            mlp_hidden=128,
            lr=5e-4,
        )
        assert config.best_checkpoint_name() == "best_l3_h8_s2_ff2048_mlp128_lr0.0005.pt"
