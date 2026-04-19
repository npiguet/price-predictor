"""Integration tests for the encode-cards CLI command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURE_CARDS = {
    "air_elemental.txt": "name: Air Elemental\ntype: Creature - Elemental\npt: 4/4\nflying\n",
    "llanowar_elves.txt": (
        "name: Llanowar Elves\ntype: Creature - Elf Druid\n"
        "pt: 1/1\nadd {G} to mana pool\n"
    ),
    "lightning_bolt.txt": "name: Lightning Bolt\ntype: Instant\ndeal 3 damage to any target\n",
    "counterspell.txt": "name: Counterspell\ntype: Instant\ncounter target spell\n",
    "forest.txt": "name: Forest\ntype: Basic Land - Forest\nbasicLandType: Forest\n",
}


def _write_fixture_cards(base_dir: Path) -> None:
    for filename, text in FIXTURE_CARDS.items():
        (base_dir / filename).write_text(text, encoding="utf-8")


@pytest.fixture()
def fixture_cards_dir(tmp_path):
    cards_dir = tmp_path / "cardsfolder"
    cards_dir.mkdir()
    _write_fixture_cards(cards_dir)
    return cards_dir


@pytest.fixture()
def encoder_artifacts(tmp_path):
    """Build a minimal real encoder and vocab, save them, return paths."""
    import torch

    from price_predictor.domain.entities import TransformerConfig
    from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel

    config = TransformerConfig(
        vocab_size=50,
        max_seq_len=64,
        d_model=8,
        n_heads=2,
        n_layers=1,
        ff_dim=16,
        dropout=0.0,
        meta_dim=4,
        regression_hidden_dim=8,
        log_offset=1.0,
    )
    model = CardPriceTransformerModel(config)
    model.eval()

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    encoder_path = model_dir / "latest.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, encoder_path)

    # Build a minimal vocab (line index = token ID; [PAD]=0, [UNK]=1 required)
    vocab_tokens = [
        "[PAD]", "[UNK]", "cardname", "creature", "instant", "land",
        "type", "name", "pt", "flying", "damage", "target", "counter",
        "spell", "mana", "pool", "forest", "elf", "elemental", "druid",
        "basic", "deal", "any", "{g}", "{r}", "{w}", "{u}", "{b}", "add",
    ]
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

    return encoder_path, vocab_path


class TestEncodeCardsFirstRun:
    def test_all_cards_encoded_on_first_run(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        result = subprocess.run(
            [sys.executable, "-m", "sealed", "encode-cards",
             "--encoder-path", str(encoder_path),
             "--vocab-path", str(vocab_path),
             "--cards-path", str(fixture_cards_dir)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        npz_files = list(fixture_cards_dir.glob("*.npz"))
        assert len(npz_files) == len(FIXTURE_CARDS)

    def test_npz_files_loadable(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        subprocess.run(
            [sys.executable, "-m", "sealed", "encode-cards",
             "--encoder-path", str(encoder_path),
             "--vocab-path", str(vocab_path),
             "--cards-path", str(fixture_cards_dir)],
            capture_output=True, text=True,
        )
        for txt_file in fixture_cards_dir.glob("*.txt"):
            npz_file = txt_file.with_suffix(".npz")
            assert npz_file.exists(), f"Missing .npz for {txt_file.name}"
            data = np.load(npz_file)
            assert "embedding" in data
            assert data["embedding"].ndim == 1


class TestEncodeCardsIdempotency:
    def test_second_run_processes_zero_cards(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        cmd = [sys.executable, "-m", "sealed", "encode-cards",
               "--encoder-path", str(encoder_path),
               "--vocab-path", str(vocab_path),
               "--cards-path", str(fixture_cards_dir)]

        subprocess.run(cmd, capture_output=True, text=True)

        result2 = subprocess.run(cmd, capture_output=True, text=True)
        assert result2.returncode == 0, result2.stderr
        assert "0 processed" in result2.stdout or "0 encoded" in result2.stdout

    def test_second_run_skips_all_cards(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        cmd = [sys.executable, "-m", "sealed", "encode-cards",
               "--encoder-path", str(encoder_path),
               "--vocab-path", str(vocab_path),
               "--cards-path", str(fixture_cards_dir)]

        subprocess.run(cmd, capture_output=True, text=True)
        result2 = subprocess.run(cmd, capture_output=True, text=True)
        count = len(FIXTURE_CARDS)
        assert str(count) + " skipped" in result2.stdout

    def test_npz_files_unchanged_after_second_run(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        cmd = [sys.executable, "-m", "sealed", "encode-cards",
               "--encoder-path", str(encoder_path),
               "--vocab-path", str(vocab_path),
               "--cards-path", str(fixture_cards_dir)]

        subprocess.run(cmd, capture_output=True, text=True)
        mtimes_before = {f: f.stat().st_mtime for f in fixture_cards_dir.glob("*.npz")}

        subprocess.run(cmd, capture_output=True, text=True)
        mtimes_after = {f: f.stat().st_mtime for f in fixture_cards_dir.glob("*.npz")}

        assert mtimes_before == mtimes_after


# ── US3: Incremental Re-Encoding After Encoder Retrain ────────────────────────


class TestIncrementalReEncoding:
    """US3: delete a subset of .npz files; re-run encodes only those, leaves others unchanged."""

    def test_deleted_cards_reprocessed_others_skipped(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        cmd = [sys.executable, "-m", "sealed", "encode-cards",
               "--encoder-path", str(encoder_path),
               "--vocab-path", str(vocab_path),
               "--cards-path", str(fixture_cards_dir)]

        # Full first run — all 5 cards encoded
        subprocess.run(cmd, capture_output=True, text=True)
        all_npz = sorted(fixture_cards_dir.glob("*.npz"))
        assert len(all_npz) == len(FIXTURE_CARDS)

        # Delete exactly 2 .npz files
        to_delete = all_npz[:2]
        surviving = all_npz[2:]
        surviving_mtimes = {f: f.stat().st_mtime for f in surviving}

        for f in to_delete:
            f.unlink()

        # Re-run — should process exactly 2, skip the rest
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "2 processed" in result.stdout or "2 encoded" in result.stdout

        expected_skipped = len(FIXTURE_CARDS) - 2
        assert str(expected_skipped) + " skipped" in result.stdout

        # Surviving files must be untouched
        for f in surviving:
            assert f.stat().st_mtime == surviving_mtimes[f], f"{f.name} was unexpectedly re-written"

        # Deleted files must be restored
        for f in to_delete:
            assert f.exists(), f"{f.name} was not re-encoded"


class TestEncodeCardsPartialFolder:
    def test_partial_folder_processes_only_missing(self, fixture_cards_dir, encoder_artifacts):
        encoder_path, vocab_path = encoder_artifacts
        cmd = [sys.executable, "-m", "sealed", "encode-cards",
               "--encoder-path", str(encoder_path),
               "--vocab-path", str(vocab_path),
               "--cards-path", str(fixture_cards_dir)]

        # Full first run
        subprocess.run(cmd, capture_output=True, text=True)

        # Delete 2 npz files
        npz_files = list(fixture_cards_dir.glob("*.npz"))
        to_delete = npz_files[:2]
        for f in to_delete:
            f.unlink()

        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "2 processed" in result.stdout or "2 encoded" in result.stdout
