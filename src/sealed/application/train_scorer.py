"""Training use case for the deck scorer model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sealed.domain.card_embedding_layout import DET_FEATURE_DIM
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.match_data_loader import (
    EmbeddingTable,
    TrainingBatch,
    TrainingExample,
    build_training_examples,
    collate_training_examples,
    load_match_outcomes,
)
from sealed.infrastructure.scorer_store import ScorerStore


@dataclass
class TrainScorerConfig:
    outcomes_path: Path
    cards_path: Path
    checkpoint_dir: Path
    resume: Path | None = None
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    n_layers: int = 2
    n_heads: int = 4
    n_seeds: int = 4
    d_ff: int = 1088
    mlp_hidden: int = 256
    val_interval: int = 1
    unfreeze_embeddings: bool = False
    embedding_lr: float = 1e-5

    def best_checkpoint_name(self) -> str:
        return (
            f"best_l{self.n_layers}_h{self.n_heads}_s{self.n_seeds}"
            f"_ff{self.d_ff}_mlp{self.mlp_hidden}_lr{self.lr}.pt"
        )

    def scorer_config(self) -> ScorerConfig:
        return ScorerConfig(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_seeds=self.n_seeds,
            d_ff=self.d_ff,
            mlp_hidden=self.mlp_hidden,
        )


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
    grad_norms: dict[str, float]


@dataclass
class ValidationResult:
    loss: float
    accuracy: float
    score_winner_mean: float
    score_winner_std: float
    score_loser_mean: float
    score_loser_std: float


@dataclass
class ResumeState:
    model: SetTransformerScorer
    start_epoch: int
    best_val_accuracy: float
    optimizer_state: dict | None


@dataclass
class TrainScorerResult:
    model: SetTransformerScorer
    metrics: TrainingMetrics
    embedding_table: EmbeddingTable


class TrainScorerUseCase:
    """Train the deck scorer on match outcome data."""

    def execute(self, config: TrainScorerConfig) -> TrainScorerResult:
        store = ScorerStore()

        train_examples, val_examples, embedding_table = _load_dataset(config)
        if config.unfreeze_embeddings:
            embedding_table.unfreeze()

        resume = _resume_or_build_model(config, store)
        _set_normalization_stats(resume.model, train_examples, embedding_table)

        optimizer = _build_optimizer(resume.model, embedding_table, config)
        if resume.optimizer_state and not config.unfreeze_embeddings:
            optimizer.load_state_dict(resume.optimizer_state)

        train_loader, val_loader = _make_loaders(
            train_examples, val_examples, config.batch_size,
        )

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        latest_path = config.checkpoint_dir / "latest.pt"
        best_path = config.checkpoint_dir / config.best_checkpoint_name()

        metrics = TrainingMetrics()
        initial_embeddings = (
            embedding_table.embedding.weight.detach().clone()
            if not embedding_table.is_frozen()
            else None
        )
        best_val_accuracy = resume.best_val_accuracy

        for epoch in range(resume.start_epoch, resume.start_epoch + config.epochs):
            train_stats = _train_one_epoch(
                resume.model, embedding_table, train_loader, optimizer,
            )
            metrics.train_losses.append(train_stats.loss)

            if (epoch - resume.start_epoch + 1) % config.val_interval == 0:
                val = _validate(resume.model, embedding_table, val_loader)
                metrics.val_losses.append(val.loss)
                metrics.val_accuracies.append(val.accuracy)
                if initial_embeddings is not None:
                    metrics.embedding_drifts.append(
                        _embedding_drift(embedding_table, initial_embeddings),
                    )

                _print_epoch_report(epoch, train_stats, val, metrics)

                store.save_checkpoint(
                    resume.model, optimizer, epoch, best_val_accuracy,
                    resume.model.config, latest_path,
                )
                if val.accuracy > best_val_accuracy:
                    best_val_accuracy = val.accuracy
                    store.save_checkpoint(
                        resume.model, optimizer, epoch, best_val_accuracy,
                        resume.model.config, best_path,
                    )

        return TrainScorerResult(
            model=resume.model,
            metrics=metrics,
            embedding_table=embedding_table,
        )


def _load_dataset(
    config: TrainScorerConfig,
) -> tuple[list[TrainingExample], list[TrainingExample], EmbeddingTable]:
    """Load all outcomes, build a shared EmbeddingTable, and 80/20-split examples."""
    outcomes = load_match_outcomes(config.outcomes_path)
    examples, embedding_table = build_training_examples(outcomes, config.cards_path)
    split_idx = int(len(examples) * 0.8)
    return examples[:split_idx], examples[split_idx:], embedding_table


def _resume_or_build_model(
    config: TrainScorerConfig, store: ScorerStore,
) -> ResumeState:
    if config.resume is None:
        return ResumeState(
            model=SetTransformerScorer(config.scorer_config()),
            start_epoch=0,
            best_val_accuracy=-1.0,
            optimizer_state=None,
        )
    checkpoint = store.load_checkpoint(config.resume)
    model = SetTransformerScorer(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return ResumeState(
        model=model,
        start_epoch=checkpoint["epoch"] + 1,
        best_val_accuracy=checkpoint.get("best_val_accuracy", -1.0),
        optimizer_state=checkpoint.get("optimizer_state_dict"),
    )


def _build_optimizer(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    config: TrainScorerConfig,
) -> torch.optim.Optimizer:
    if embedding_table.is_frozen():
        return torch.optim.Adam(model.parameters(), lr=config.lr)
    return torch.optim.Adam([
        {"params": list(model.parameters()), "lr": config.lr},
        {"params": list(embedding_table.parameters()), "lr": config.embedding_lr},
    ])


def _make_loaders(
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample],
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_training_examples,
    )
    val_loader = DataLoader(
        val_examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_training_examples,
    )
    return train_loader, val_loader


def _train_one_epoch(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    grad_norms: dict[str, float] = {}

    for batch in loader:
        optimizer.zero_grad()
        score_winner, score_loser = _score_batch(model, embedding_table, batch)
        loss = _pairwise_bce(score_winner, score_loser)
        loss.backward()
        grad_norms = _gradient_norms(model)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        with torch.no_grad():
            correct += (score_winner > score_loser).sum().item()
            total += score_winner.size(0)

    return EpochStats(
        loss=total_loss / max(n_batches, 1),
        accuracy=correct / max(total, 1),
        grad_norms=grad_norms,
    )


def _validate(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    val_loader: DataLoader,
) -> ValidationResult:
    """Compute validation loss, accuracy, and score statistics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0
    all_winner_scores: list[torch.Tensor] = []
    all_loser_scores: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            score_winner, score_loser = _score_batch(model, embedding_table, batch)
            loss = _pairwise_bce(score_winner, score_loser)
            total_loss += loss.item()
            n_batches += 1
            correct += (score_winner > score_loser).sum().item()
            total += score_winner.size(0)
            all_winner_scores.append(score_winner.squeeze(1))
            all_loser_scores.append(score_loser.squeeze(1))

    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    all_w = torch.cat(all_winner_scores) if all_winner_scores else torch.zeros(1)
    all_l = torch.cat(all_loser_scores) if all_loser_scores else torch.zeros(1)
    return ValidationResult(
        loss=avg_loss,
        accuracy=accuracy,
        score_winner_mean=all_w.mean().item(),
        score_winner_std=all_w.std().item(),
        score_loser_mean=all_l.mean().item(),
        score_loser_std=all_l.std().item(),
    )


def _score_batch(
    model: SetTransformerScorer,
    embedding_table: EmbeddingTable,
    batch: TrainingBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    winner_cards = embedding_table(batch.winner_indices)
    loser_cards = embedding_table(batch.loser_indices)
    return (
        model(winner_cards, batch.winner_mask),
        model(loser_cards, batch.loser_mask),
    )


def _pairwise_bce(
    score_winner: torch.Tensor, score_loser: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry pairwise loss: BCE on (winner − loser) with target=1."""
    return F.binary_cross_entropy_with_logits(
        score_winner - score_loser,
        torch.ones_like(score_winner),
    )


def _gradient_norms(model: SetTransformerScorer) -> dict[str, float]:
    """Compute L2 gradient norm for each named component of the model."""
    norms: dict[str, float] = {}
    for i, sab in enumerate(model.sab_layers):
        norms[f"sab{i}"] = _component_grad_norm(sab)
    norms["pma"] = _component_grad_norm(model.pma)
    norms["mlp"] = _component_grad_norm(model.mlp)
    return norms


def _component_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _set_normalization_stats(
    model: SetTransformerScorer,
    examples: list[TrainingExample],
    embedding_table: EmbeddingTable,
) -> None:
    """Compute per-feature mean and std for the deterministic-feature slice."""
    indices: list[torch.Tensor] = []
    for ex in examples:
        indices.append(ex.winner_indices)
        indices.append(ex.loser_indices)
    if not indices:
        return
    flat = torch.cat(indices)
    offset = embedding_table.embedding.embedding_dim - DET_FEATURE_DIM
    feats = embedding_table.embedding.weight.detach()[flat, offset:]
    model.feat_mean.copy_(feats.mean(dim=0))
    std = feats.std(dim=0)
    std[std == 0] = 1.0
    model.feat_std.copy_(std)


def _embedding_drift(
    embedding_table: EmbeddingTable, initial: torch.Tensor,
) -> float:
    return (embedding_table.embedding.weight.data - initial).norm(dim=1).mean().item()


def _print_epoch_report(
    epoch: int,
    train_stats: EpochStats,
    val: ValidationResult,
    metrics: TrainingMetrics,
) -> None:
    drift_str = ""
    if metrics.embedding_drifts:
        drift_str = f"  embedding_drift={metrics.embedding_drifts[-1]:.6f}"
    grad_str = "  ".join(f"{k}={v:.4f}" for k, v in train_stats.grad_norms.items())
    print(
        f"Epoch {epoch + 1}: "
        f"train_loss={train_stats.loss:.4f}  "
        f"train_acc={train_stats.accuracy:.4f}  "
        f"val_loss={val.loss:.4f}  "
        f"val_acc={val.accuracy:.4f}"
        f"{drift_str}"
    )
    print(
        f"  scores: "
        f"winner={val.score_winner_mean:.4f}±{val.score_winner_std:.4f}  "
        f"loser={val.score_loser_mean:.4f}±{val.score_loser_std:.4f}"
    )
    print(f"  grad_norms: {grad_str}")
