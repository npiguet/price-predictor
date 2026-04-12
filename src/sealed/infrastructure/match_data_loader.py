"""Parse match-outcomes.txt and build PyTorch training datasets."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)


@dataclass
class MatchOutcome:
    deck_a_names: list[str]
    deck_b_names: list[str]
    wins_a: int
    wins_b: int

    @property
    def winner_names(self) -> list[str]:
        return self.deck_a_names if self.wins_a > self.wins_b else self.deck_b_names

    @property
    def loser_names(self) -> list[str]:
        return self.deck_b_names if self.wins_a > self.wins_b else self.deck_a_names


@dataclass
class TrainingExample:
    winner_cards: torch.Tensor  # (N, 544)
    loser_cards: torch.Tensor   # (M, 544)


@dataclass
class TrainingBatch:
    winner_cards: torch.Tensor  # (batch, max_winner_cards, 544)
    loser_cards: torch.Tensor   # (batch, max_loser_cards, 544)
    winner_mask: torch.Tensor   # (batch, max_winner_cards) bool
    loser_mask: torch.Tensor    # (batch, max_loser_cards) bool


def parse_match_outcome(line: str) -> MatchOutcome:
    """Parse a single line from match-outcomes.txt."""
    parts = line.strip().split(";")
    deck_a_names = parts[0].split("|")
    deck_b_names = parts[1].split("|")
    wins_a = int(parts[2])
    wins_b = int(parts[3])
    return MatchOutcome(deck_a_names, deck_b_names, wins_a, wins_b)


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
) -> list[TrainingExample]:
    """Build training examples from match outcomes, filtering basic lands.

    Matches containing cards with missing embeddings are skipped with a
    warning printed to stderr.
    """
    locator = ConvertedCardLocator(cards_path)
    embedding_cache: dict[str, np.ndarray | None] = {}
    missing_cards: set[str] = set()
    examples: list[TrainingExample] = []
    skipped = 0

    for outcome in outcomes:
        winner_vecs = _resolve_deck(
            outcome.winner_names, locator, embedding_cache, missing_cards,
        )
        if winner_vecs is None:
            skipped += 1
            continue

        loser_vecs = _resolve_deck(
            outcome.loser_names, locator, embedding_cache, missing_cards,
        )
        if loser_vecs is None:
            skipped += 1
            continue

        if winner_vecs and loser_vecs:
            examples.append(TrainingExample(
                winner_cards=torch.from_numpy(np.stack(winner_vecs)),
                loser_cards=torch.from_numpy(np.stack(loser_vecs)),
            ))

    if skipped:
        print(
            f"Skipped {skipped} matches due to {len(missing_cards)} missing card(s)",
            file=sys.stderr,
        )

    return examples


def _resolve_deck(
    names: list[str],
    locator: ConvertedCardLocator,
    cache: dict[str, np.ndarray | None],
    missing: set[str],
) -> list[np.ndarray] | None:
    """Resolve a deck's nonland cards to embeddings, or None if any are missing."""
    vecs: list[np.ndarray] = []
    for name in names:
        if name.lower() in BASIC_LAND_NAMES:
            continue
        if name not in cache:
            cache[name] = locator.load_embedding(name)
        emb = cache[name]
        if emb is None:
            if name not in missing:
                expected = locator.expected_path(name, ".npz")
                print(
                    f"Missing card embedding: {expected} (card: {name})",
                    file=sys.stderr,
                )
                missing.add(name)
            return None
        vecs.append(emb)
    return vecs


def collate_training_examples(batch: list[TrainingExample]) -> TrainingBatch:
    """Collate variable-length training examples into a padded batch."""
    max_winner = max(ex.winner_cards.size(0) for ex in batch)
    max_loser = max(ex.loser_cards.size(0) for ex in batch)
    d = batch[0].winner_cards.size(1)
    bs = len(batch)

    winner_cards = torch.zeros(bs, max_winner, d)
    loser_cards = torch.zeros(bs, max_loser, d)
    winner_mask = torch.zeros(bs, max_winner, dtype=torch.bool)
    loser_mask = torch.zeros(bs, max_loser, dtype=torch.bool)

    for i, ex in enumerate(batch):
        nw = ex.winner_cards.size(0)
        nl = ex.loser_cards.size(0)
        winner_cards[i, :nw] = ex.winner_cards
        loser_cards[i, :nl] = ex.loser_cards
        winner_mask[i, :nw] = True
        loser_mask[i, :nl] = True

    return TrainingBatch(winner_cards, loser_cards, winner_mask, loser_mask)
