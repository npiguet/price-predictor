"""End-to-end smoke test for ``train-encoder`` against fixture data.

Runs a few epochs on a tiny synthetic corpus + tiny vocab + the fixture
``cards-played.sample.txt``. Asserts that ``latest.pt`` round-trips
through ``SealedEncoderStore.load_encoder``, that the saved file
contains no head/MLM keys (FR-020), that the new
``cards-win-rates.txt`` schema is in place, and that the optimizer's
LR schedule respects FR-022's 5%-warmup-then-constant shape.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path

import pytest

from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from sealed.application.train_encoder import (
    SealedEncoderModel,
    TrainEncoderConfig,
    _make_optimizer,
)
from sealed.application.train_encoder import run as run_train_encoder
from sealed.domain.encoder_model import SealedEncoderConfig
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
    """Seed enough tokens to cover the fixture corpus (including [MASK])."""
    tokens = [
        "[PAD]", "[UNK]", "cardname", "[MASK]",
        "name", "mana", "cost", "types", "spell", "type",
        "creature", "instant", "sorcery", "artifact", "enchantment",
        "land", "planeswalker", "deals", "damage", "to", "any",
        "target", "draw", "card", "destroy", "flying", "vigilance",
        "first_strike", "p", "t", "none", "trample", "haste",
        "{r}", "{u}", "{b}", "{g}", "{w}", "{c}", "{1}", "{2}", "{3}",
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
            epochs=3,
            lr=1e-3,
            patience=5,
            n_layers=2,
            n_heads=2,
            n_pool_queries=2,
            shrinkage_k=20.0,
        )
        run_train_encoder(config)

        latest = model_output / "latest.pt"
        assert latest.exists(), "latest.pt must exist after training"

        # Saved file: encoder children only.
        import torch
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        prefixes = {k.split(".", 1)[0] for k in payload["model_state_dict"]}
        assert prefixes == {"token_encoder", "card_encoder"}

        # No source-checkpoint field — guard against a future regression
        # where train-encoder silently loads weights instead of training
        # from random init (FR-016).
        assert "init_from" not in payload
        assert "source_checkpoint" not in payload

        loaded_model, loaded_cfg = SealedEncoderStore().load_encoder(latest)
        assert loaded_cfg.n_layers == 2
        assert loaded_cfg.n_pool_queries == 2

        win_rates = tmp_path / "output" / "sealed" / "cards-win-rates.txt"
        assert win_rates.exists(), "cards-win-rates.txt must be written"
        rows = win_rates.read_text(encoding="utf-8").splitlines()
        assert len(rows) >= 2  # header + at least one card
        # Header schema (23 columns: 1 + 4 + 9×2).
        header = rows[0]
        assert header.startswith("card_name;wins_when_played;")
        assert "shrunk_color_lift_G" in header
        assert len(header.split(";")) == 23
    finally:
        os.chdir(cwd_before)


@pytest.mark.integration
def test_lr_schedule_warmup_then_constant():
    """FR-022 LR schedule: linear warmup over ceil(0.05 * total_steps),
    then constant ``lr`` for the remainder.
    """
    cfg = SealedEncoderConfig(
        vocab_size=64, d_model=16, n_layers=1, n_heads=2,
        ff_dim=32, max_seq_len=8, dropout=0.0, n_pool_queries=2,
    )
    model = SealedEncoderModel(cfg)
    lr = 1e-3
    total_steps = 200
    optimizer, scheduler = _make_optimizer(model, lr=lr, total_steps=total_steps)
    warmup_steps = max(1, int(math.ceil(total_steps * 0.05)))
    # Step 0: factor 0/warmup_steps = 0 → lr = 0 < lr.
    initial = optimizer.param_groups[0]["lr"]
    assert initial < lr
    # Step warmup_steps reaches the constant tail at exactly `lr`.
    for _ in range(warmup_steps):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(lr)
    # Past the warmup: still equal to lr.
    for _ in range(total_steps - warmup_steps):
        optimizer.step()
        scheduler.step()
        assert optimizer.param_groups[0]["lr"] == pytest.approx(lr)
