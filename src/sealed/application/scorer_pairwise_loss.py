"""The scorer's pairwise loss: each batch's winner and loser decks scored,
the Bradley-Terry binary cross-entropy on the score difference, optionally
weighted by the match margin, and the statistics a training or validation
epoch accumulates over its batches."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.match_data_loader import EmbeddingTable, TrainingBatch


@dataclass(frozen=True)
class PairScoring:
    """How a batch's decks are scored and their loss weighed: the embedding
    table the card indices read, the device the batch moves to, and the
    margin weighting of the pairwise loss."""

    embedding_table: EmbeddingTable
    device: torch.device
    margin_weighting: str = "none"


class _BatchStats:
    """Accumulate pairwise-BCE batch statistics across a training or validation epoch."""

    def __init__(self) -> None:
        self._total_loss = 0.0
        self._n_batches = 0
        self._correct = 0
        self._total = 0
        self._winner_scores: list[torch.Tensor] = []
        self._loser_scores: list[torch.Tensor] = []

    def add(
        self,
        loss: torch.Tensor,
        score_winner: torch.Tensor,
        score_loser: torch.Tensor,
    ) -> None:
        self._total_loss += loss.item()
        self._n_batches += 1
        self._correct += (score_winner > score_loser).sum().item()
        self._total += score_winner.size(0)
        self._winner_scores.append(score_winner.detach().squeeze(1))
        self._loser_scores.append(score_loser.detach().squeeze(1))

    @property
    def mean_loss(self) -> float:
        return self._total_loss / max(self._n_batches, 1)

    @property
    def accuracy(self) -> float:
        return self._correct / max(self._total, 1)

    def score_summary(self) -> tuple[float, float, float, float]:
        winners = torch.cat(self._winner_scores) if self._winner_scores else torch.zeros(1)
        losers = torch.cat(self._loser_scores) if self._loser_scores else torch.zeros(1)
        return (
            winners.mean().item(), winners.std().item(),
            losers.mean().item(), losers.std().item(),
        )


def _batch_to_device(batch: TrainingBatch, device: torch.device) -> TrainingBatch:
    return TrainingBatch(
        winner_indices=batch.winner_indices.to(device),
        loser_indices=batch.loser_indices.to(device),
        winner_mask=batch.winner_mask.to(device),
        loser_mask=batch.loser_mask.to(device),
        margins=batch.margins.to(device),
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
    score_winner: torch.Tensor,
    score_loser: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights is None:
        return F.binary_cross_entropy_with_logits(
            score_winner - score_loser,
            torch.ones_like(score_winner),
        )
    per_example = F.binary_cross_entropy_with_logits(
        score_winner - score_loser,
        torch.ones_like(score_winner),
        reduction="none",
    )
    return (per_example * weights.view_as(per_example)).mean()


def _compute_margin_weights(
    margins: torch.Tensor, mode: str,
) -> torch.Tensor | None:
    """Map per-example margins to BCE weights for the configured weighting mode.

    ``"none"`` returns ``None`` so ``_pairwise_bce`` takes its unweighted path
    bit-for-bit identically to the pre-flag behavior. ``"linear"`` weights each
    pair by its absolute game-win margin (Bo7: 1-4). ``"log"`` dampens with
    ``log(1 + margin)`` and is the spec's recommended fallback if linear
    training proves unstable.
    """
    if mode == "none":
        return None
    floats = margins.float()
    if mode == "linear":
        return floats
    if mode == "log":
        return (1.0 + floats).log()
    raise ValueError(f"Unknown margin_weighting mode: {mode!r}")
