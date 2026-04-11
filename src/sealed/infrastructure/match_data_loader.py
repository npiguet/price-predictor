"""Parse match-outcomes.txt and build PyTorch training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BASIC_LANDS = frozenset({"plains", "island", "swamp", "mountain", "forest"})


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


def _card_name_to_filename(name: str) -> str:
    """Convert a card name to its embedding filename (lowercase, spaces→underscores).

    Unicode accents are stripped (e.g. â → a) to match the on-disk filenames
    produced by the card-script converter.  Known filename typos are corrected
    via the corrections table.
    """
    import unicodedata

    from sealed.infrastructure.card_name_corrections import FILENAME_CORRECTIONS

    # Decompose then drop combining marks (accents)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    sanitized = (
        ascii_name.lower()
        .replace(" // ", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace(",", "")
        .replace(":", "")
        .replace("!", "")
        .replace('"', "")
        .replace("&", "")
        .replace("+", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("/", "")
    )
    return FILENAME_CORRECTIONS.get(sanitized, sanitized)


def _load_card_embedding(cards_path: Path, card_name: str) -> np.ndarray | None:
    """Load a single card's embedding from its .npz file.

    Double-faced / split / adventure cards have filenames like
    ``frontface_backface.npz`` but the match outcomes file uses only the
    front-face name.  When the exact file is missing we fall back to the
    first ``.npz`` whose name starts with the sanitized front-face name.
    """
    resolved = _card_name_to_filename(card_name)
    if "/" in resolved:
        first_letter, filename = resolved.split("/", 1)
    else:
        filename = resolved
        first_letter = filename[0] if filename else "_"
    letter_dir = cards_path / first_letter

    # Exact match first (fast path)
    npz_path = letter_dir / f"{filename}.npz"
    if npz_path.exists():
        return np.load(npz_path)["embedding"]

    # Prefix search for double-faced / split / adventure cards
    if letter_dir.is_dir():
        prefix = filename + "_"
        for candidate in letter_dir.iterdir():
            if candidate.suffix == ".npz" and candidate.stem.startswith(prefix):
                return np.load(candidate)["embedding"]

    return None


def build_training_examples(
    outcomes: list[MatchOutcome],
    cards_path: Path,
) -> list[TrainingExample]:
    """Build training examples from match outcomes, filtering basic lands.

    Matches containing cards with missing embeddings are skipped with a
    warning printed to stderr.
    """
    import sys

    embedding_cache: dict[str, np.ndarray | None] = {}
    missing_cards: set[str] = set()
    examples = []
    skipped = 0

    for outcome in outcomes:
        winner_names = [n for n in outcome.winner_names if n.lower() not in BASIC_LANDS]
        loser_names = [n for n in outcome.loser_names if n.lower() not in BASIC_LANDS]

        skip = False
        winner_vecs = []
        for name in winner_names:
            if name not in embedding_cache:
                embedding_cache[name] = _load_card_embedding(cards_path, name)
            emb = embedding_cache[name]
            if emb is None:
                if name not in missing_cards:
                    filename = _card_name_to_filename(name)
                    first_letter = filename[0] if filename else "_"
                    print(
                        f"Missing card embedding: {cards_path / first_letter / filename}.npz"
                        f" (card: {name})",
                        file=sys.stderr,
                    )
                    missing_cards.add(name)
                skip = True
                break
            winner_vecs.append(emb)

        if skip:
            skipped += 1
            continue

        loser_vecs = []
        for name in loser_names:
            if name not in embedding_cache:
                embedding_cache[name] = _load_card_embedding(cards_path, name)
            emb = embedding_cache[name]
            if emb is None:
                if name not in missing_cards:
                    filename = _card_name_to_filename(name)
                    first_letter = filename[0] if filename else "_"
                    print(
                        f"Missing card embedding: {cards_path / first_letter / filename}.npz"
                        f" (card: {name})",
                        file=sys.stderr,
                    )
                    missing_cards.add(name)
                skip = True
                break
            loser_vecs.append(emb)

        if skip:
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
