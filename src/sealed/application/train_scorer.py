"""Training use case for the deck scorer model.

The run's parts live in modules of their own: the dataset
(``scorer_dataset``), what a run starts from (``scorer_resume``), the
Phase B encoder (``scorer_encoder_loading``, ``scorer_phase_b``), the
pairwise loss (``scorer_pairwise_loss``), one epoch
(``scorer_training_loop``), the checkpoints (``scorer_checkpoints``) and the
setup that gathers them (``scorer_training_setup``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sealed.application.scorer_checkpoints import EpochAccuracy, _persist_checkpoint
from sealed.application.scorer_log import _log
from sealed.application.scorer_training_loop import (
    TrainingMetrics,
    _PeakPatience,
    _print_epoch_report,
    _train_one_epoch,
    _validate_epoch,
)
from sealed.application.scorer_training_setup import _prepare
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.match_data_loader import EmbeddingTable
from sealed.infrastructure.scorer_store import ScorerStore


@dataclass
class TrainScorerConfig:
    outcomes_path: Path
    cards_path: Path
    checkpoint_dir: Path
    #: The ``.npz`` embeddings' tree, laid out like ``cards_path``; None
    #: reads them beside the cards.
    embeddings_path: Path | None = None
    resume: Path | None = None
    scorer_checkpoint: Path | None = None
    #: Where a fresh Phase B run takes its encoder weights; None on Phase A.
    encoder_checkpoint: Path | None = None
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-5
    n_layers: int = 6
    n_heads: int = 4
    n_seeds: int = 4
    d_ff: int = 1088
    mlp_hidden: int = 256
    dropout: float = 0.2
    embedding_lr: float = 0.0
    patience: int = 5
    val_fraction: float = 0.2
    random_seed: int = 42
    encoder_chunk_size: int = 128
    max_grad_norm: float = 100.0
    margin_weighting: str = "none"  # "none" | "linear" | "log"

    @property
    def phase(self) -> Literal["A", "B"]:
        return "A" if self.embedding_lr == 0 else "B"

    def _margin_suffix(self) -> str:
        if self.margin_weighting == "linear":
            return "_mwlin"
        if self.margin_weighting == "log":
            return "_mwlog"
        return ""

    def best_checkpoint_name(self) -> str:
        suffix = self._margin_suffix()
        if self.phase == "B":
            return (
                f"best_phaseB_l{self.n_layers}_h{self.n_heads}_s{self.n_seeds}"
                f"_ff{self.d_ff}_mlp{self.mlp_hidden}"
                f"_lr{self.lr}_emblr{self.embedding_lr}{suffix}.pt"
            )
        return (
            f"best_l{self.n_layers}_h{self.n_heads}_s{self.n_seeds}"
            f"_ff{self.d_ff}_mlp{self.mlp_hidden}_lr{self.lr}{suffix}.pt"
        )

    def latest_checkpoint_name(self) -> str:
        suffix = self._margin_suffix()
        return (
            f"latest_phaseB{suffix}.pt" if self.phase == "B" else f"latest{suffix}.pt"
        )

    def scorer_config(self, d_model: int) -> ScorerConfig:
        """Build the architecture config for a fresh scorer.

        ``d_model`` is the width of the cached card embeddings (encoder
        text-vector width + ``FEATURE_COUNT``), read off the loaded ``.npz``
        cache by the caller — it is not a user-chosen knob.
        """
        return ScorerConfig(
            d_model=d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_seeds=self.n_seeds,
            d_ff=self.d_ff,
            mlp_hidden=self.mlp_hidden,
            dropout=self.dropout,
        )


@dataclass
class TrainScorerResult:
    model: SetTransformerScorer
    metrics: TrainingMetrics
    embedding_table: EmbeddingTable


class TrainScorerUseCase:
    """Train the deck scorer on match outcome data."""

    def execute(self, config: TrainScorerConfig) -> TrainScorerResult:
        store = ScorerStore()
        ctx = _prepare(config, store)

        metrics = TrainingMetrics()
        best_val_accuracy = ctx.best_val_accuracy
        patience = _PeakPatience(config.patience)

        _log(f"Starting training loop ({config.epochs} epochs)")
        for epoch in range(ctx.start_epoch, ctx.start_epoch + config.epochs):
            train_stats = _train_one_epoch(ctx)
            metrics.train_losses.append(train_stats.loss)

            val = _validate_epoch(ctx, metrics)
            _print_epoch_report(epoch, train_stats, val, metrics)
            new_best, best_val_accuracy = _persist_checkpoint(
                ctx, store, EpochAccuracy(epoch, val.accuracy, best_val_accuracy),
            )
            if patience.exhausted(new_best):
                break

        return TrainScorerResult(
            model=ctx.model,
            metrics=metrics,
            embedding_table=ctx.embedding_table,
        )
