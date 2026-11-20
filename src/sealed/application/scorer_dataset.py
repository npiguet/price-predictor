"""The ``train-scorer`` dataset: the match outcomes read into training
examples over one shared embedding table, their seeded train/validation
split, the two data loaders and the scorer's per-feature normalization
statistics."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from sealed.application.scorer_log import _log
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.match_data_loader import (
    EmbeddingTable,
    MatchTrainingExample,
    build_training_examples,
    collate_training_examples,
    load_match_outcomes,
)

if TYPE_CHECKING:
    from sealed.application.train_scorer import TrainScorerConfig


def _load_dataset(
    config: TrainScorerConfig,
) -> tuple[list[MatchTrainingExample], list[MatchTrainingExample], EmbeddingTable]:
    """Load all outcomes, build a shared EmbeddingTable, and randomly split examples."""
    t0 = time.monotonic()
    outcomes = load_match_outcomes(config.outcomes_path)
    _log(f"  parsed {len(outcomes)} match outcomes ({time.monotonic() - t0:.1f}s)")

    t1 = time.monotonic()
    examples, embedding_table = build_training_examples(
        outcomes, config.cards_path, embeddings_path=config.embeddings_path,
    )
    _log(
        f"  built {len(examples)} training examples "
        f"(loaded card embeddings, {time.monotonic() - t1:.1f}s)",
    )
    if len(examples) < 2:
        return examples, [], embedding_table
    train_examples, val_examples = train_test_split(
        examples,
        test_size=config.val_fraction,
        random_state=config.random_seed,
        shuffle=True,
    )
    return train_examples, val_examples, embedding_table


def _make_loaders(
    train_examples: list[MatchTrainingExample],
    val_examples: list[MatchTrainingExample],
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


def _set_normalization_stats(
    model: SetTransformerScorer,
    examples: list[MatchTrainingExample],
    embedding_table: EmbeddingTable,
) -> None:
    """Compute per-feature mean and std for the deterministic-feature slice."""
    if not examples:
        return
    flat = torch.cat(
        [ex.winner_indices for ex in examples]
        + [ex.loser_indices for ex in examples]
    )
    mean, std = embedding_table.deterministic_feature_stats(flat)
    model.feat_mean.copy_(mean)
    model.feat_std.copy_(std)
