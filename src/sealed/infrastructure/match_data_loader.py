"""Parse match-outcomes.txt and build PyTorch training datasets."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from price_predictor.infrastructure.converted_card_parser import parse_converted_text
from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.domain.deck_stats import compute_deck_stats
from sealed.domain.scorer_model import ScorerConfig
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)


class MatchDropReason(Enum):
    MISSING_CARD = "missing_card"
    EMPTY_AFTER_BASICS = "empty_after_basics"


@dataclass
class MatchOutcome:
    """One match from ``match-outcomes.txt`` (see ``specs/sealed-deck-picker.md``)."""

    timestamp: str
    run_id: str
    set_code: str
    method_a: str
    method_b: str
    deck_a_names: list[str]
    deck_b_names: list[str]
    games: str  # per-game winner sequence, e.g. "ABB"
    play: str   # per-game play-first sequence, same length as games
    duration_s: int

    @property
    def wins_a(self) -> int:
        return self.games.count("A")

    @property
    def wins_b(self) -> int:
        return self.games.count("B")

    @property
    def winner_names(self) -> list[str]:
        return self.deck_a_names if self.wins_a > self.wins_b else self.deck_b_names

    @property
    def loser_names(self) -> list[str]:
        return self.deck_b_names if self.wins_a > self.wins_b else self.deck_a_names


@dataclass
class MatchTrainingExample:
    winner_indices: torch.Tensor      # (N,) long, rows into the EmbeddingTable
    loser_indices: torch.Tensor       # (M,) long
    winner_deck_stats: torch.Tensor   # (DECK_STATS_DIM,) float
    loser_deck_stats: torch.Tensor    # (DECK_STATS_DIM,) float


@dataclass
class TrainingBatch:
    winner_indices: torch.Tensor      # (batch, max_winner_cards) long
    loser_indices: torch.Tensor       # (batch, max_loser_cards) long
    winner_mask: torch.Tensor         # (batch, max_winner_cards) bool
    loser_mask: torch.Tensor          # (batch, max_loser_cards) bool
    winner_deck_stats: torch.Tensor   # (batch, DECK_STATS_DIM) float
    loser_deck_stats: torch.Tensor    # (batch, DECK_STATS_DIM) float


class EmbeddingTable(nn.Module):
    """Lookup table mapping card name → ``d_model``-dim card vector.

    Frozen by default; ``unfreeze()`` flips ``requires_grad`` so the optimizer
    can fine-tune card embeddings during the second phase of training.
    """

    def __init__(self, vectors: torch.Tensor, name_to_idx: dict[str, int]) -> None:
        super().__init__()
        num_cards, d_model = vectors.shape
        self.embedding = nn.Embedding(num_cards, d_model)
        with torch.no_grad():
            self.embedding.weight.copy_(vectors)
        self.embedding.weight.requires_grad = False
        self.name_to_idx = dict(name_to_idx)

    @property
    def num_cards(self) -> int:
        return self.embedding.num_embeddings

    def freeze(self) -> None:
        self.embedding.weight.requires_grad = False

    def unfreeze(self) -> None:
        self.embedding.weight.requires_grad = True

    def is_frozen(self) -> bool:
        return not self.embedding.weight.requires_grad

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def deterministic_feature_stats(
        self, indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` of the deterministic-feature slice over the given rows."""
        offset = self.embedding.embedding_dim - FEATURE_COUNT
        feats = self.embedding.weight.detach()[indices, offset:]
        mean = feats.mean(dim=0)
        std = feats.std(dim=0)
        std[std == 0] = 1.0
        return mean, std


def parse_match_outcome(line: str) -> MatchOutcome:
    """Parse a single 10-field line from match-outcomes.txt.

    Format: ``timestamp;run_id;set_code;method_A;method_B;deckA;deckB;games;play;duration_s``.
    """
    parts = line.strip().split(";")
    if len(parts) != 10:
        raise ValueError(
            f"Expected 10 semicolon-delimited fields in match-outcomes line, got {len(parts)}"
        )
    timestamp, run_id, set_code, method_a, method_b = parts[0:5]
    deck_a_names = parts[5].split("|")
    deck_b_names = parts[6].split("|")
    games, play = parts[7], parts[8]
    duration_s = int(parts[9])
    return MatchOutcome(
        timestamp=timestamp,
        run_id=run_id,
        set_code=set_code,
        method_a=method_a,
        method_b=method_b,
        deck_a_names=deck_a_names,
        deck_b_names=deck_b_names,
        games=games,
        play=play,
        duration_s=duration_s,
    )


def load_match_outcomes(path: Path) -> list[MatchOutcome]:
    """Load all match outcomes from a file."""
    outcomes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                outcomes.append(parse_match_outcome(line))
    return outcomes


def build_training_examples(
    outcomes: list[MatchOutcome],
    cards_path: Path,
) -> tuple[list[MatchTrainingExample], EmbeddingTable]:
    """Build a shared embedding table and per-match training examples.

    Walks every deck once, loading each unique card embedding into a single
    ``EmbeddingTable``. Each example then carries integer indices into that
    table instead of repeating the full card vectors. Matches that reference
    cards without an embedding on disk are skipped with a warning to stderr.

    Each match's two decks come from independent sealed pools — they share
    no cards by construction. Don't introduce cross-deck features (e.g.
    shared-card counts) here; they would be vacuously zero.
    """
    builder = _ExampleBuilder(ConvertedCardLocator(cards_path))
    examples = builder.build(outcomes)
    return examples, builder.into_table()


class _ExampleBuilder:
    def __init__(self, locator: ConvertedCardLocator) -> None:
        self._locator = locator
        self._name_to_idx: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._missing: set[str] = set()

    def build(self, outcomes: list[MatchOutcome]) -> list[MatchTrainingExample]:
        examples: list[MatchTrainingExample] = []
        drops: Counter[MatchDropReason] = Counter()
        for outcome in outcomes:
            winner = self._resolve_deck(outcome.winner_names)
            if winner is None:
                drops[MatchDropReason.MISSING_CARD] += 1
                continue
            loser = self._resolve_deck(outcome.loser_names)
            if loser is None:
                drops[MatchDropReason.MISSING_CARD] += 1
                continue
            winner_indices, winner_names = winner
            loser_indices, loser_names = loser
            if not winner_indices or not loser_indices:
                drops[MatchDropReason.EMPTY_AFTER_BASICS] += 1
                continue
            examples.append(MatchTrainingExample(
                winner_indices=torch.tensor(winner_indices, dtype=torch.long),
                loser_indices=torch.tensor(loser_indices, dtype=torch.long),
                winner_deck_stats=self._deck_stats_tensor(winner_names),
                loser_deck_stats=self._deck_stats_tensor(loser_names),
            ))
        self._report_drops(drops)
        return examples

    def _report_drops(self, drops: Counter[MatchDropReason]) -> None:
        if not drops:
            return
        missing = drops.get(MatchDropReason.MISSING_CARD, 0)
        empty = drops.get(MatchDropReason.EMPTY_AFTER_BASICS, 0)
        print(
            f"Skipped matches: {missing} missing-card "
            f"({len(self._missing)} unique cards), {empty} empty-after-basics",
            file=sys.stderr,
        )

    def into_table(self) -> EmbeddingTable:
        if self._vectors:
            stacked = torch.from_numpy(np.stack(self._vectors)).float()
        else:
            stacked = torch.zeros(1, ScorerConfig().d_model)
        return EmbeddingTable(stacked, self._name_to_idx)

    def _resolve_deck(
        self, names: list[str],
    ) -> tuple[list[int], list[str]] | None:
        indices: list[int] = []
        kept_names: list[str] = []
        for name in names:
            if name.lower() in BASIC_LAND_NAMES:
                continue
            idx = self._intern(name)
            if idx is None:
                return None
            indices.append(idx)
            kept_names.append(name)
        return indices, kept_names

    def _deck_stats_tensor(self, names: list[str]) -> torch.Tensor:
        """Compute the deck-stats vector for a deck identified by card names."""
        cards = []
        for name in names:
            text = self._locator.load_text(name)
            if text is None:
                continue
            try:
                cards.append(parse_converted_text(text))
            except ValueError:
                continue
        return torch.from_numpy(compute_deck_stats(cards))

    def _intern(self, name: str) -> int | None:
        existing = self._name_to_idx.get(name)
        if existing is not None:
            return existing
        emb = self._locator.load_embedding(name)
        if emb is None:
            if name not in self._missing:
                expected = self._locator.expected_path(name, ".npz")
                print(
                    f"Missing card embedding: {expected} (card: {name})",
                    file=sys.stderr,
                )
                self._missing.add(name)
            return None
        idx = len(self._vectors)
        self._vectors.append(emb)
        self._name_to_idx[name] = idx
        return idx


def collate_training_examples(batch: list[MatchTrainingExample]) -> TrainingBatch:
    """Collate variable-length training examples into a padded batch."""
    max_winner = max(ex.winner_indices.size(0) for ex in batch)
    max_loser = max(ex.loser_indices.size(0) for ex in batch)
    bs = len(batch)

    winner_indices = torch.zeros(bs, max_winner, dtype=torch.long)
    loser_indices = torch.zeros(bs, max_loser, dtype=torch.long)
    winner_mask = torch.zeros(bs, max_winner, dtype=torch.bool)
    loser_mask = torch.zeros(bs, max_loser, dtype=torch.bool)

    for i, ex in enumerate(batch):
        nw = ex.winner_indices.size(0)
        nl = ex.loser_indices.size(0)
        winner_indices[i, :nw] = ex.winner_indices
        loser_indices[i, :nl] = ex.loser_indices
        winner_mask[i, :nw] = True
        loser_mask[i, :nl] = True

    winner_deck_stats = torch.stack([ex.winner_deck_stats for ex in batch])
    loser_deck_stats = torch.stack([ex.loser_deck_stats for ex in batch])

    return TrainingBatch(
        winner_indices=winner_indices,
        loser_indices=loser_indices,
        winner_mask=winner_mask,
        loser_mask=loser_mask,
        winner_deck_stats=winner_deck_stats,
        loser_deck_stats=loser_deck_stats,
    )
