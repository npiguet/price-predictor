"""Parse match-outcomes.txt and build PyTorch training datasets."""

from __future__ import annotations

import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.domain.scorer_model import ScorerConfig
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
)

_LOAD_WORKERS = 8


class MatchDropReason(Enum):
    MISSING_CARD = "missing_card"
    EMPTY_AFTER_BASICS = "empty_after_basics"


@dataclass
class MatchOutcome:
    """One match from ``match-outcomes.txt`` (see ``specs/2026-03-28-sealed-deck-picker.md``)."""

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
    winner_indices: torch.Tensor  # (N,) long, rows into the EmbeddingTable
    loser_indices: torch.Tensor   # (M,) long
    margin: int                   # |wins_a - wins_b|, e.g. 1..4 for Bo7


@dataclass
class TrainingBatch:
    winner_indices: torch.Tensor  # (batch, max_winner_cards) long
    loser_indices: torch.Tensor   # (batch, max_loser_cards) long
    winner_mask: torch.Tensor     # (batch, max_winner_cards) bool
    loser_mask: torch.Tensor      # (batch, max_loser_cards) bool
    margins: torch.Tensor         # (batch,) float


class EmbeddingTable(nn.Module):
    """Lookup table mapping card name → ``d_model``-dim card vector.

    The leading ``embedding_dim - FEATURE_COUNT`` columns hold the encoder's
    pooled text vector (width depends on the encoder: ``2 * encoder_d_model``
    for the price / ``--pool-mode dual`` encoder, ``encoder_d_model`` for
    ``--pool-mode attn``); the trailing ``FEATURE_COUNT`` columns hold
    deterministic game features. ``d_model`` is read straight from the loaded
    ``.npz`` vectors — this table never assumes a fixed width. The table
    itself is never trained; in Phase B, ``set_text_vectors`` splices fresh
    encoder outputs into the leading slice each batch step so the scorer's
    forward pass sees live, gradient-tracking vectors that flow back into the
    encoder parameters.
    """

    def __init__(self, vectors: torch.Tensor, name_to_idx: dict[str, int]) -> None:
        super().__init__()
        num_cards, d_model = vectors.shape
        self.embedding = nn.Embedding(num_cards, d_model)
        with torch.no_grad():
            self.embedding.weight.copy_(vectors)
        self.embedding.weight.requires_grad = False
        self.name_to_idx = dict(name_to_idx)
        self._live_weight: torch.Tensor | None = None

    @property
    def num_cards(self) -> int:
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self) -> int:
        """Width of each card vector = encoder text-vector width + ``FEATURE_COUNT``."""
        return self.embedding.embedding_dim

    def set_text_vectors(
        self, indices: torch.Tensor, text_vectors: torch.Tensor,
    ) -> None:
        """Splice ``text_vectors`` into the leading ``embedding_dim - FEATURE_COUNT``
        text columns of the rows at ``indices``; trailing deterministic-feature
        columns survive. The resulting weight tensor is non-leaf, so a
        downstream loss backpropagates through ``text_vectors`` to its source
        (the encoder).
        """
        text_dim = self.embedding.embedding_dim - FEATURE_COUNT
        baseline = self.embedding.weight.detach()
        det_slice = baseline[indices, text_dim:]
        new_rows = torch.cat([text_vectors, det_slice], dim=-1)
        self._live_weight = baseline.clone().index_copy(0, indices, new_rows)

    def clear_live_weight(self) -> None:
        """Drop the per-step live weight so the next forward uses the baseline."""
        self._live_weight = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self._live_weight is not None:
            return F.embedding(indices, self._live_weight)
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
        self._preload_embeddings(outcomes)
        examples: list[MatchTrainingExample] = []
        drops: Counter[MatchDropReason] = Counter()
        for outcome in outcomes:
            result = self._resolve_match(outcome)
            if isinstance(result, MatchDropReason):
                drops[result] += 1
            else:
                examples.append(result)
        self._report_drops(drops)
        return examples

    def _preload_embeddings(self, outcomes: list[MatchOutcome]) -> None:
        """Load every unique non-basic card embedding once, in parallel.

        NumPy releases the GIL during ``.npz`` decompression, so a small
        thread pool gives a near-linear speedup over the sequential per-card
        ``np.load`` path. Names are sorted before submission so the resulting
        embedding-table row indices are deterministic across runs.
        """
        unique_names = sorted({
            name
            for outcome in outcomes
            for deck in (outcome.deck_a_names, outcome.deck_b_names)
            for name in deck
            if name.lower() not in BASIC_LAND_NAMES
        })
        with ThreadPoolExecutor(max_workers=_LOAD_WORKERS) as executor:
            embeddings = list(
                executor.map(self._locator.load_embedding, unique_names),
            )
        for name, emb in zip(unique_names, embeddings, strict=True):
            if emb is None:
                self._record_missing(name)
                continue
            self._name_to_idx[name] = len(self._vectors)
            self._vectors.append(emb)

    def _resolve_match(
        self, outcome: MatchOutcome,
    ) -> MatchTrainingExample | MatchDropReason:
        winner = self._resolve_deck(outcome.winner_names)
        loser = self._resolve_deck(outcome.loser_names)
        if winner is None or loser is None:
            return MatchDropReason.MISSING_CARD
        if not winner or not loser:
            return MatchDropReason.EMPTY_AFTER_BASICS
        return MatchTrainingExample(
            winner_indices=torch.tensor(winner, dtype=torch.long),
            loser_indices=torch.tensor(loser, dtype=torch.long),
            margin=abs(outcome.wins_a - outcome.wins_b),
        )

    def _report_drops(self, drops: Counter[MatchDropReason]) -> None:
        if not drops:
            return
        parts = [f"{reason.value}={drops[reason]}" for reason in MatchDropReason]
        print(
            f"Skipped matches: {', '.join(parts)} "
            f"({len(self._missing)} unique missing cards)",
            file=sys.stderr,
        )

    def into_table(self) -> EmbeddingTable:
        if self._vectors:
            stacked = torch.from_numpy(np.stack(self._vectors)).float()
        else:
            stacked = torch.zeros(1, ScorerConfig().d_model)
        return EmbeddingTable(stacked, self._name_to_idx)

    def _resolve_deck(self, names: list[str]) -> list[int] | None:
        indices: list[int] = []
        for name in names:
            if name.lower() in BASIC_LAND_NAMES:
                continue
            idx = self._intern(name)
            if idx is None:
                return None
            indices.append(idx)
        return indices

    def _intern(self, name: str) -> int | None:
        cached = self._name_to_idx.get(name)
        if cached is not None:
            return cached
        emb = self._locator.load_embedding(name)
        if emb is None:
            self._record_missing(name)
            return None
        idx = len(self._vectors)
        self._vectors.append(emb)
        self._name_to_idx[name] = idx
        return idx

    def _record_missing(self, name: str) -> None:
        if name in self._missing:
            return
        expected = self._locator.expected_path(name, ".npz")
        print(
            f"Missing card embedding: {expected} (card: {name})",
            file=sys.stderr,
        )
        self._missing.add(name)


def collate_training_examples(batch: list[MatchTrainingExample]) -> TrainingBatch:
    """Collate variable-length training examples into a padded batch."""
    winner_seqs = [ex.winner_indices for ex in batch]
    loser_seqs = [ex.loser_indices for ex in batch]

    winner_indices = pad_sequence(winner_seqs, batch_first=True, padding_value=0)
    loser_indices = pad_sequence(loser_seqs, batch_first=True, padding_value=0)
    winner_mask = _length_mask(winner_seqs, winner_indices.size(1))
    loser_mask = _length_mask(loser_seqs, loser_indices.size(1))
    margins = torch.tensor([ex.margin for ex in batch], dtype=torch.float)

    return TrainingBatch(winner_indices, loser_indices, winner_mask, loser_mask, margins)


def _length_mask(sequences: list[torch.Tensor], padded_len: int) -> torch.Tensor:
    lengths = torch.tensor([s.size(0) for s in sequences])
    return torch.arange(padded_len)[None, :] < lengths[:, None]
