"""One ``train-scorer`` epoch: the training pass with per-group gradient
clipping, the validation pass, and the report each epoch logs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from price_predictor.infrastructure.torch_training import clip_per_group
from sealed.application.scorer_log import _log
from sealed.application.scorer_pairwise_loss import (
    _batch_to_device,
    _BatchStats,
    _compute_margin_weights,
    _pairwise_bce,
    _score_batch,
)
from sealed.application.scorer_phase_b import _phase_b_inject_encoder
from sealed.application.scorer_training_context import _TrainingContext
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.match_data_loader import EmbeddingTable


@dataclass
class TrainingMetrics:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    embedding_drifts: list[float] = field(default_factory=list)


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    grad_norms_mean: dict[str, float]
    grad_norms_max: dict[str, float]


@dataclass
class ValidationResult:
    loss: float
    accuracy: float
    score_winner_mean: float
    score_winner_std: float
    score_loser_mean: float
    score_loser_std: float


def _train_one_epoch(ctx: _TrainingContext) -> EpochStats:
    ctx.model.train()
    if ctx.encoder is not None:
        # Phase B runs the encoder in eval mode so dropout doesn't add
        # stochasticity. The cached `.npz` files the Phase A scorer was tuned
        # against were produced under `encoder.eval()`; turning dropout on
        # here would feed the scorer noisier vectors than it learned to
        # score against, dragging early Phase B val_acc down independently
        # of any encoder weight movement. `eval()` no-ops dropout/BatchNorm
        # only — it does not disable autograd, so the encoder still trains.
        ctx.encoder.eval()
    stats = _BatchStats()
    norms_per_batch: dict[str, list[float]] = {}

    for batch in ctx.train_loader:
        batch = _batch_to_device(batch, ctx.device)
        ctx.optimizer.zero_grad()
        if ctx.encoder is not None:
            _phase_b_inject_encoder(ctx, batch)
        score_winner, score_loser = _score_batch(ctx.model, ctx.embedding_table, batch)
        weights = _compute_margin_weights(batch.margins, ctx.margin_weighting)
        loss = _pairwise_bce(score_winner, score_loser, weights)
        loss.backward()
        per_batch = clip_per_group(ctx.optimizer, max_norm=ctx.max_grad_norm)
        for name, value in per_batch.items():
            norms_per_batch.setdefault(name, []).append(value)
        ctx.optimizer.step()
        ctx.embedding_table.clear_live_weight()
        stats.add(loss, score_winner, score_loser)

    grad_norms_mean = {
        name: sum(values) / len(values) for name, values in norms_per_batch.items()
    }
    grad_norms_max = {
        name: max(values) for name, values in norms_per_batch.items()
    }

    return EpochStats(
        loss=stats.mean_loss,
        accuracy=stats.accuracy,
        grad_norms_mean=grad_norms_mean,
        grad_norms_max=grad_norms_max,
    )


def _validate(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    val_loader: DataLoader,
    device: torch.device,
    margin_weighting: str = "none",
) -> ValidationResult:
    """Compute validation loss, accuracy, and score statistics."""
    model.eval()
    stats = _BatchStats()
    with torch.no_grad():
        for batch in val_loader:
            batch = _batch_to_device(batch, device)
            score_winner, score_loser = _score_batch(model, embedding_table, batch)
            weights = _compute_margin_weights(batch.margins, margin_weighting)
            loss = _pairwise_bce(score_winner, score_loser, weights)
            stats.add(loss, score_winner, score_loser)

    w_mean, w_std, l_mean, l_std = stats.score_summary()
    return ValidationResult(
        loss=stats.mean_loss,
        accuracy=stats.accuracy,
        score_winner_mean=w_mean,
        score_winner_std=w_std,
        score_loser_mean=l_mean,
        score_loser_std=l_std,
    )


def _print_epoch_report(
    epoch: int,
    train_stats: EpochStats,
    val: ValidationResult,
    metrics: TrainingMetrics,
) -> None:
    drift_str = ""
    if metrics.embedding_drifts:
        drift_str = f"  embedding_drift={metrics.embedding_drifts[-1]:.6f}"
    grad_str = "  ".join(
        f"{k}=mean({train_stats.grad_norms_mean[k]:.4f})/max({train_stats.grad_norms_max[k]:.4f})"
        for k in train_stats.grad_norms_mean
    )
    _log(
        f"Epoch {epoch + 1}: "
        f"train_loss={train_stats.loss:.4f}  "
        f"train_acc={train_stats.accuracy:.4f}  "
        f"val_loss={val.loss:.4f}  "
        f"val_acc={val.accuracy:.4f}"
        f"{drift_str}"
    )
    _log(
        f"  scores: "
        f"winner={val.score_winner_mean:.4f}±{val.score_winner_std:.4f}  "
        f"loser={val.score_loser_mean:.4f}±{val.score_loser_std:.4f}"
    )
    _log(f"  grad_norms: {grad_str}")
