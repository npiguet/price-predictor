"""Shared fixtures for sealed unit tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from sealed.domain.card_embedding_layout import total_dim


class FakeProcess:
    """Fake subprocess.Popen that exits immediately or hangs until terminated."""

    def __init__(self, pid: int = 99, returncode: int = 0, hang: bool = False):
        self.pid = pid
        self.returncode = returncode
        self._hang = hang
        self._terminated = False

    def wait(self):
        if self._hang:
            while not self._terminated:
                time.sleep(0.01)
        return self.returncode

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def poll(self):
        return None if (self._hang and not self._terminated) else self.returncode


@pytest.fixture
def synthetic_cards_dir(tmp_path: Path) -> Path:
    """Create ``tmp_path/cards/<letter>/<name>.npz`` for 50 random card embeddings."""
    rng = np.random.default_rng(0)
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    for i in range(50):
        name = f"card_{i}"
        letter_dir = cards_dir / name[0]
        letter_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            letter_dir / f"{name}.npz",
            embedding=rng.standard_normal(total_dim(256)).astype(np.float32),
        )
    return cards_dir


@pytest.fixture
def synthetic_outcomes_file(tmp_path: Path, synthetic_cards_dir: Path) -> Path:
    """Write a 20-match outcomes.txt referencing only cards in ``synthetic_cards_dir``."""
    return _write_outcomes(tmp_path, n_matches=20, n_cards_per_deck=10)


@pytest.fixture
def synthetic_generated_decks_file(tmp_path: Path) -> Path:
    """Write a generated-decks.txt with three 40-card decks across two sets.

    Each line: ``SET_CODE;Card1|Card2|...|Card40``. Useful for tests that
    consume the generated-decks format (build-decks output, self-play index).
    """
    path = tmp_path / "generated-decks.txt"
    decks = [
        ("MH3", [f"mh3_card_{i}" for i in range(23)] + ["Plains"] * 8 + ["Island"] * 9),
        ("MH3", [f"mh3_card_{i}" for i in range(23, 46)] + ["Mountain"] * 8 + ["Forest"] * 9),
        ("BLB", [f"blb_card_{i}" for i in range(23)] + ["Swamp"] * 17),
    ]
    lines = [f"{set_code};" + "|".join(deck) for set_code, deck in decks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_outcomes(tmp_path: Path, n_matches: int, n_cards_per_deck: int) -> Path:
    rng = np.random.default_rng(0)
    card_names = [f"card_{i}" for i in range(50)]
    outcomes_file = tmp_path / "outcomes.txt"
    lines = []
    for _ in range(n_matches):
        deck_a = rng.choice(card_names, n_cards_per_deck, replace=True)
        deck_b = rng.choice(card_names, n_cards_per_deck, replace=True)
        wins_a, wins_b = (2, int(rng.choice([0, 1])))
        lines.append(f"{'|'.join(deck_a)};{'|'.join(deck_b)};{wins_a};{wins_b}")
    outcomes_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcomes_file
