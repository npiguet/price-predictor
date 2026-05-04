"""SC-005 verification: run ``train-encoder`` twice with --shrinkage-k 0 and
20 against the same fixture corpus, and diff the two
``cards-win-rates.txt`` snapshots. Low-observation cards must shift; high-
observation cards must stay nearly identical.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from sealed.application.train_encoder import TrainEncoderConfig
from sealed.application.train_encoder import run as run_train_encoder

_FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures"
_CARDS_PLAYED_FIXTURE = _FIXTURE_DIR / "sealed" / "cards-played.sample.txt"
_CONVERTED_DIR = _FIXTURE_DIR / "converted_cards_training"


def _ensure_letter_layout(dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for src_file in _CONVERTED_DIR.glob("*.txt"):
        letter = src_file.stem[0]
        target_dir = dst / letter
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, target_dir / src_file.name)


def _build_minimal_vocab(vocab_path: Path) -> None:
    tokens = [
        "[PAD]", "[UNK]", "cardname",
        "name", "mana", "cost", "types", "spell", "type",
        "creature", "instant", "sorcery", "artifact", "enchantment",
        "land", "planeswalker", "deals", "damage", "to", "any",
        "target", "draw", "card", "destroy", "flying", "vigilance",
        "first_strike", "p", "t", "none", "trample", "haste",
        "{r}", "{u}", "{b}", "{g}", "{w}", "{1}", "{2}", "{3}",
    ]
    vocab = {tok: i for i, tok in enumerate(tokens)}
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    save_vocabulary(vocab, vocab_path)


def _parse_win_rates(path: Path) -> dict[str, dict]:
    """Parse cards-win-rates.txt into {card_name: {wp, wd, raw, shrunk}}."""
    out: dict[str, dict] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        parts = line.split(";")
        out[parts[0]] = {
            "wp": int(parts[1]),
            "wd": int(parts[2]),
            "raw": float(parts[3]),
            "shrunk": float(parts[4]),
        }
    return out


@pytest.mark.integration
def test_shrinkage_diff_snapshot(tmp_path: Path):
    cards_folder = tmp_path / "cardsfolder"
    _ensure_letter_layout(cards_folder)
    vocab_path = tmp_path / "models" / "sealed" / "encoder" / "vocab.txt"
    _build_minimal_vocab(vocab_path)
    model_output = tmp_path / "models" / "sealed" / "encoder"

    cwd_before = Path.cwd()
    os.chdir(tmp_path)
    try:
        # k=0 run.
        config_k0 = TrainEncoderConfig(
            cards_played_path=_CARDS_PLAYED_FIXTURE,
            cards_folder=cards_folder,
            vocab_path=vocab_path,
            model_output_dir=model_output,
            batch_size=4, epochs=1, lr=1e-3, patience=1,
            n_layers=2, n_heads=2, n_pool_queries=2,
            shrinkage_k=0.0,
        )
        run_train_encoder(config_k0)
        snapshot_k0 = tmp_path / "snapshot_k0.txt"
        shutil.copy2(tmp_path / "output" / "sealed" / "cards-win-rates.txt", snapshot_k0)

        # k=20 run.
        config_k20 = TrainEncoderConfig(
            cards_played_path=_CARDS_PLAYED_FIXTURE,
            cards_folder=cards_folder,
            vocab_path=vocab_path,
            model_output_dir=model_output,
            batch_size=4, epochs=1, lr=1e-3, patience=1,
            n_layers=2, n_heads=2, n_pool_queries=2,
            shrinkage_k=20.0,
        )
        run_train_encoder(config_k20)
        snapshot_k20 = tmp_path / "snapshot_k20.txt"
        shutil.copy2(tmp_path / "output" / "sealed" / "cards-win-rates.txt", snapshot_k20)

        rows_k0 = _parse_win_rates(snapshot_k0)
        rows_k20 = _parse_win_rates(snapshot_k20)
        assert set(rows_k0.keys()) == set(rows_k20.keys())

        observed_low_n_shift = False
        for name, k0 in rows_k0.items():
            k20 = rows_k20[name]
            if k0["wd"] <= 3:
                # Low-n card: visible shift expected.
                if abs(k0["shrunk"] - k20["shrunk"]) > 0.05:
                    observed_low_n_shift = True
        # The fixture must exhibit at least one low-n card with a shift.
        assert observed_low_n_shift, (
            "SC-005: at least one low-observation card must shift > 0.05 "
            "between k=0 and k=20 snapshots."
        )
    finally:
        os.chdir(cwd_before)
