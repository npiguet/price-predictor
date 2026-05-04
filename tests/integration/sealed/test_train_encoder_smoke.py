"""End-to-end smoke test for ``train-encoder`` against fixture data.

Runs one epoch on a tiny synthetic corpus + tiny vocab + the fixture
``cards-played.sample.txt``. Asserts that ``latest.pt`` round-trips
through ``SealedEncoderStore.load_encoder`` and that
``cards-win-rates.txt`` has been written.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from sealed.application.train_encoder import TrainEncoderConfig
from sealed.application.train_encoder import run as run_train_encoder
from sealed.infrastructure.encoder_store import SealedEncoderStore

_FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures"
_CARDS_PLAYED_FIXTURE = _FIXTURE_DIR / "sealed" / "cards-played.sample.txt"
_CONVERTED_DIR = _FIXTURE_DIR / "converted_cards_training"


def _ensure_letter_layout(dst: Path) -> None:
    """Copy the flat fixture corpus into the letter-keyed layout
    expected by ``ConvertedCardLocator``."""
    dst.mkdir(parents=True, exist_ok=True)
    for src_file in _CONVERTED_DIR.glob("*.txt"):
        if not src_file.name.endswith(".txt"):
            continue
        letter = src_file.stem[0]
        target_dir = dst / letter
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target_dir / src_file.name)


def _build_minimal_vocab(vocab_path: Path) -> None:
    """Tokens that suffice to encode every fixture card without UNK noise.

    The model still works with UNK, but pinning the vocab keeps the
    smoke test deterministic.
    """
    tokens = [
        "[PAD]", "[UNK]", "cardname",
        "name", "mana", "cost", "types", "spell", "type",
        "creature", "instant", "sorcery", "artifact", "enchantment",
        "land", "planeswalker", "deals", "damage", "to", "any",
        "target", "draw", "card", "destroy", "flying", "vigilance",
        "first_strike", "p", "t", "none", "trample", "haste",
        "{r}", "{u}", "{b}", "{g}", "{w}", "{1}", "{2}", "{3}",
        "{r}{r}", "{u}{u}", "{b}{b}", "{g}{g}", "{w}{w}",
    ]
    vocab = {tok: i for i, tok in enumerate(tokens)}
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocabulary(vocab, vocab_path)


@pytest.mark.integration
def test_train_encoder_smoke(tmp_path: Path):
    cards_folder = tmp_path / "cardsfolder"
    _ensure_letter_layout(cards_folder)

    vocab_path = tmp_path / "models" / "sealed" / "encoder" / "vocab.txt"
    _build_minimal_vocab(vocab_path)

    model_output = tmp_path / "models" / "sealed" / "encoder"

    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        config = TrainEncoderConfig(
            cards_played_path=_CARDS_PLAYED_FIXTURE,
            cards_folder=cards_folder,
            vocab_path=vocab_path,
            model_output_dir=model_output,
            batch_size=4,
            epochs=1,
            lr=1e-3,
            patience=1,
            n_layers=2,
            n_heads=2,
            n_pool_queries=2,
            shrinkage_k=20.0,
        )
        run_train_encoder(config)

        latest = model_output / "latest.pt"
        assert latest.exists(), "latest.pt must exist after training"

        loaded_model, loaded_cfg = SealedEncoderStore().load_encoder(latest)
        assert loaded_cfg.n_layers == 2
        assert loaded_cfg.n_pool_queries == 2

        win_rates = tmp_path / "output" / "sealed" / "cards-win-rates.txt"
        assert win_rates.exists(), "cards-win-rates.txt must be written"
        rows = win_rates.read_text(encoding="utf-8").splitlines()
        assert len(rows) >= 2  # header + at least one card
    finally:
        os.chdir(cwd_before)
