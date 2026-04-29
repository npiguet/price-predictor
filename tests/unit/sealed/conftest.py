"""Shared fixtures for sealed unit tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from price_predictor.infrastructure.transformer_store import save_model
from sealed.domain.card_embedding_layout import total_dim
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore


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


@pytest.fixture
def phase_b_setup(tmp_path: Path):
    """Build a tiny synthetic Phase B environment.

    Creates:
    - A small ``output/cardsfolder/``-style directory with `.npz` and matching
      `.txt` files for 10 cards (all under one letter for simplicity).
    - A 10-match `outcomes.txt` referencing only those cards.
    - A price-predictor encoder `.pt` file with a tiny architecture and a
      `vocab.txt` next to it.
    - A pre-trained Phase A scorer checkpoint.

    Returns the four paths plus the encoder config. Slow to build (~1s) but
    enables real Phase B forward passes in tests.
    """
    encoder_d_model = 16
    encoder_cfg = TransformerConfig(
        d_model=encoder_d_model, n_layers=1, n_heads=2, ff_dim=32,
        max_seq_len=32, vocab_size=64, dropout=0.0,
    )
    encoder = CardPriceTransformerModel(encoder_cfg)
    encoder_dir = tmp_path / "encoder"
    save_model(encoder, encoder_cfg, encoder_dir, version="v1")
    vocab = {"[PAD]": 0, "[UNK]": 1}
    extras = [
        "creature", "instant", "land", "sorcery", "type", "types", "mana",
        "cost", "name", "spell", "card", "none", "the", "a", "of", "to",
    ]
    for token in extras:
        vocab[token] = len(vocab)
    save_vocabulary(vocab, encoder_dir / "vocab.txt")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    letter_dir = cards_dir / "c"
    letter_dir.mkdir()
    card_dim = total_dim(encoder_d_model)
    rng = np.random.default_rng(7)
    card_names: list[str] = []
    for i in range(10):
        name = f"card_{i}"
        card_names.append(name)
        np.savez_compressed(
            letter_dir / f"{name}.npz",
            embedding=rng.standard_normal(card_dim).astype(np.float32),
        )
        (letter_dir / f"{name}.txt").write_text(
            f"name: {name}\ntypes: creature\nmana cost: none\n",
            encoding="utf-8",
        )

    outcomes = tmp_path / "outcomes.txt"
    lines: list[str] = []
    rng2 = np.random.default_rng(11)
    for _ in range(10):
        deck_a = list(rng2.choice(card_names, 4, replace=False))
        deck_b = list(rng2.choice(card_names, 4, replace=False))
        lines.append(
            "2026-04-22T14:30:05Z;fixture;RVR;forge-best;forge-3sub;"
            f"{'|'.join(deck_a)};{'|'.join(deck_b)};AA;BA;12"
        )
    outcomes.write_text("\n".join(lines) + "\n", encoding="utf-8")

    scorer_cfg = ScorerConfig(d_model=card_dim)
    scorer = SetTransformerScorer(scorer_cfg)
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-3)
    phase_a_path = tmp_path / "phase_a.pt"
    ScorerStore().save_checkpoint(
        scorer, optimizer, epoch=0, best_val_accuracy=0.5,
        config=scorer_cfg, path=phase_a_path,
        train_config={
            "lr": 1e-5, "embedding_lr": 0.0, "patience": 5,
            "outcomes_path": str(outcomes), "cards_path": str(cards_dir),
        },
    )

    return {
        "encoder_checkpoint": encoder_dir / "latest.pt",
        "vocab_path": encoder_dir / "vocab.txt",
        "cards_dir": cards_dir,
        "outcomes_file": outcomes,
        "phase_a_checkpoint": phase_a_path,
        "scorer_config": scorer_cfg,
        "encoder_config": encoder_cfg,
    }


def _write_outcomes(tmp_path: Path, n_matches: int, n_cards_per_deck: int) -> Path:
    rng = np.random.default_rng(0)
    card_names = [f"card_{i}" for i in range(50)]
    outcomes_file = tmp_path / "outcomes.txt"
    lines = []
    for _ in range(n_matches):
        deck_a = rng.choice(card_names, n_cards_per_deck, replace=True)
        deck_b = rng.choice(card_names, n_cards_per_deck, replace=True)
        # A wins 2 games; B wins 0 or 1 → games is "AA" or "ABA".
        b_wins = int(rng.choice([0, 1]))
        games = "AA" if b_wins == 0 else "ABA"
        play = "B" * len(games)  # arbitrary; fixture consumers don't care who was on the play
        lines.append(
            "2026-04-22T14:30:05Z;fixture-run;RVR;forge-best;forge-3sub;"
            f"{'|'.join(deck_a)};{'|'.join(deck_b)};{games};{play};12"
        )
    outcomes_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outcomes_file
