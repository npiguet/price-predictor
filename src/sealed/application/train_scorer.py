"""Training use case for the deck scorer model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.match_data_loader import (
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


@dataclass
class TrainingMetrics:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    embedding_drifts: list[float] = field(default_factory=list)


class _ExampleDataset(Dataset):
    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class TrainScorerUseCase:
    """Train the deck scorer on match outcome data."""

    def execute(self, config: TrainScorerConfig) -> tuple[SetTransformerScorer, TrainingMetrics]:
        store = ScorerStore()

        # Load data
        outcomes = load_match_outcomes(config.outcomes_path)

        # 80/20 split by match
        split_idx = int(len(outcomes) * 0.8)
        train_outcomes = outcomes[:split_idx]
        val_outcomes = outcomes[split_idx:]

        train_examples = build_training_examples(train_outcomes, config.cards_path)
        val_examples = build_training_examples(val_outcomes, config.cards_path)

        # Build model
        model_config = dict(
            d_model=544,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            n_seeds=config.n_seeds,
            d_ff=config.d_ff,
            mlp_hidden=config.mlp_hidden,
        )

        if config.resume:
            checkpoint = store.load_checkpoint(config.resume)
            model = SetTransformerScorer(**checkpoint["config"])
            model.load_state_dict(checkpoint["model_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_accuracy = checkpoint.get("best_val_accuracy", -1.0)
        else:
            model = SetTransformerScorer(**model_config)
            start_epoch = 0
            best_val_accuracy = -1.0

        # Compute normalization statistics from training data
        _set_normalization_stats(model, train_examples)

        # Embedding unfreezing setup
        embedding_table = None
        initial_embeddings = None

        if config.unfreeze_embeddings:
            # Build embedding lookup table from all unique card vectors
            all_examples = train_examples + val_examples
            embedding_table, card_to_idx = _build_embedding_table(all_examples)
            initial_embeddings = embedding_table.data.clone()

            # Replace card vectors with indices in training examples
            train_examples = _replace_with_indices(train_examples, card_to_idx)
            val_examples = _replace_with_indices(val_examples, card_to_idx)

            # Store on the use case, not on the model (to avoid double-counting in parameters)
            model.embedding_table = None  # marker for test inspection

            # Differential learning rates: model params at lr, embeddings at embedding_lr
            scorer_params = list(model.parameters())
            optimizer = torch.optim.Adam([
                {"params": scorer_params, "lr": config.lr},
                {"params": [embedding_table], "lr": config.embedding_lr},
            ])
        else:
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        can_restore_optim = (
            config.resume and not config.unfreeze_embeddings
            and "optimizer_state_dict" in checkpoint
        )
        if can_restore_optim:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Data loaders
        train_loader = DataLoader(
            _ExampleDataset(train_examples),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_training_examples,
        )
        val_loader = DataLoader(
            _ExampleDataset(val_examples),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_training_examples,
        )

        metrics = TrainingMetrics()
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint_path = config.checkpoint_dir / config.best_checkpoint_name()

        for epoch in range(start_epoch, start_epoch + config.epochs):
            # Training
            model.train()
            epoch_loss = 0.0
            train_correct = 0
            train_total = 0
            n_batches = 0
            grad_norms: dict[str, float] = {}

            for batch in train_loader:
                optimizer.zero_grad()

                winner_cards = batch.winner_cards
                loser_cards = batch.loser_cards
                if embedding_table is not None:
                    winner_cards = _lookup_embeddings(winner_cards, embedding_table)
                    loser_cards = _lookup_embeddings(loser_cards, embedding_table)

                score_winner = model(winner_cards, batch.winner_mask)
                score_loser = model(loser_cards, batch.loser_mask)
                loss = F.binary_cross_entropy_with_logits(
                    score_winner - score_loser,
                    torch.ones_like(score_winner),
                )
                loss.backward()
                grad_norms = _gradient_norms(model)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
                with torch.no_grad():
                    train_correct += (score_winner > score_loser).sum().item()
                    train_total += score_winner.size(0)

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_accuracy = train_correct / max(train_total, 1)
            metrics.train_losses.append(avg_train_loss)

            # Validation
            if (epoch - start_epoch + 1) % config.val_interval == 0:
                val = _validate(model, val_loader, embedding_table)
                metrics.val_losses.append(val.loss)
                metrics.val_accuracies.append(val.accuracy)

                drift_str = ""
                if embedding_table is not None and initial_embeddings is not None:
                    drift = (embedding_table.data - initial_embeddings).norm(dim=1).mean().item()
                    metrics.embedding_drifts.append(drift)
                    drift_str = f"  embedding_drift={drift:.6f}"

                grad_str = "  ".join(f"{k}={v:.4f}" for k, v in grad_norms.items())

                print(
                    f"Epoch {epoch + 1}: "
                    f"train_loss={avg_train_loss:.4f}  "
                    f"train_acc={train_accuracy:.4f}  "
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

                # Save latest
                store.save_checkpoint(
                    model, optimizer, epoch, best_val_accuracy,
                    model_config, config.checkpoint_dir / "latest.pt",
                )

                # Save best
                if val.accuracy > best_val_accuracy:
                    best_val_accuracy = val.accuracy
                    store.save_checkpoint(
                        model, optimizer, epoch, best_val_accuracy,
                        model_config, best_checkpoint_path,
                    )

        return model, metrics


def _gradient_norms(model: SetTransformerScorer) -> dict[str, float]:
    """Compute L2 gradient norm for each named component of the model."""
    norms: dict[str, float] = {}
    for i, sab in enumerate(model.sab_layers):
        total = 0.0
        for p in sab.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        norms[f"sab{i}"] = total ** 0.5
    pma_total = 0.0
    for p in model.pma.parameters():
        if p.grad is not None:
            pma_total += p.grad.data.norm(2).item() ** 2
    norms["pma"] = pma_total ** 0.5
    mlp_total = 0.0
    for p in model.mlp.parameters():
        if p.grad is not None:
            mlp_total += p.grad.data.norm(2).item() ** 2
    norms["mlp"] = mlp_total ** 0.5
    return norms


def _set_normalization_stats(model: SetTransformerScorer, examples: list[TrainingExample]) -> None:
    """Compute per-feature mean and std for deterministic features (indices 512-543)."""
    all_feats = []
    for ex in examples:
        all_feats.append(ex.winner_cards[:, 512:])
        all_feats.append(ex.loser_cards[:, 512:])

    if not all_feats:
        return

    combined = torch.cat(all_feats, dim=0)  # (total_cards, 32)
    model.feat_mean.copy_(combined.mean(dim=0))
    std = combined.std(dim=0)
    std[std == 0] = 1.0  # avoid division by zero
    model.feat_std.copy_(std)


@dataclass
class ValidationResult:
    loss: float
    accuracy: float
    score_winner_mean: float
    score_winner_std: float
    score_loser_mean: float
    score_loser_std: float


def _validate(
    model: SetTransformerScorer,
    val_loader: DataLoader,
    embedding_table: torch.nn.Parameter | None = None,
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
            winner_cards = batch.winner_cards
            loser_cards = batch.loser_cards
            if embedding_table is not None:
                winner_cards = _lookup_embeddings(winner_cards, embedding_table)
                loser_cards = _lookup_embeddings(loser_cards, embedding_table)

            score_winner = model(winner_cards, batch.winner_mask)
            score_loser = model(loser_cards, batch.loser_mask)
            loss = F.binary_cross_entropy_with_logits(
                score_winner - score_loser,
                torch.ones_like(score_winner),
            )
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


def _build_embedding_table(
    examples: list[TrainingExample],
) -> tuple[torch.nn.Parameter, dict[int, int]]:
    """Build an nn.Parameter embedding table from all unique card vectors.

    Returns the table and a mapping from vector hash to row index.
    """
    unique_vecs = {}
    vec_list = []

    for ex in examples:
        for cards in [ex.winner_cards, ex.loser_cards]:
            for i in range(cards.size(0)):
                vec = cards[i]
                key = hash(vec.numpy().tobytes())
                if key not in unique_vecs:
                    unique_vecs[key] = len(vec_list)
                    vec_list.append(vec)

    table = torch.stack(vec_list)
    return torch.nn.Parameter(table), unique_vecs


def _replace_with_indices(
    examples: list[TrainingExample],
    card_to_idx: dict[int, int],
) -> list[TrainingExample]:
    """Replace card vectors with index tensors for embedding table lookup."""
    new_examples = []
    for ex in examples:
        winner_idx = torch.tensor([
            card_to_idx[hash(ex.winner_cards[i].numpy().tobytes())]
            for i in range(ex.winner_cards.size(0))
        ], dtype=torch.long)
        loser_idx = torch.tensor([
            card_to_idx[hash(ex.loser_cards[i].numpy().tobytes())]
            for i in range(ex.loser_cards.size(0))
        ], dtype=torch.long)
        new_examples.append(TrainingExample(
            winner_cards=winner_idx.unsqueeze(1).float(),  # shape (N, 1) - stores indices as float
            loser_cards=loser_idx.unsqueeze(1).float(),
        ))
    return new_examples


def _lookup_embeddings(
    index_tensor: torch.Tensor,
    embedding_table: torch.nn.Parameter,
) -> torch.Tensor:
    """Look up card vectors from the embedding table using stored indices."""
    # index_tensor has shape (batch, max_cards, 1) with float indices from collate
    indices = index_tensor[:, :, 0].long()
    return embedding_table[indices]
