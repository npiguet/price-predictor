"""Integration test for the default-flipped encode-cards path.

Creates a tiny sealed encoder via ``SealedEncoderStore.save_encoder`` (no
training), runs ``encode-cards`` against a tiny corpus, and verifies the
resulting ``.npz`` is shaped ``(2 * d_model + FEATURE_COUNT,)`` and that
the encoder weights used are the sealed ones.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from price_predictor.infrastructure.tokenizer_store import save_vocabulary
from sealed.domain.card_embedding_layout import FEATURE_COUNT
from sealed.domain.encoder_model import SealedEncoderConfig, SealedEncoderModel
from sealed.infrastructure.cli import build_parser, run_encode_cards
from sealed.infrastructure.encoder_store import SealedEncoderStore


@pytest.mark.integration
def test_sealed_encoder_default_flow(tmp_path: Path):
    cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        cards_path = tmp_path / "cardsfolder"
        (cards_path / "l").mkdir(parents=True)
        (cards_path / "l" / "lightning_bolt.txt").write_text(
            "name: Lightning Bolt\nmana cost: {R}\ntypes: instant\n"
            "spell[1]: CARDNAME deals 3 damage to any target.\n",
            encoding="utf-8",
        )

        encoder_dir = tmp_path / "models" / "sealed" / "encoder"
        encoder_dir.mkdir(parents=True)
        vocab_path = encoder_dir / "vocab.txt"
        save_vocabulary(
            {tok: i for i, tok in enumerate([
                "[PAD]", "[UNK]", "cardname", "name", "mana", "cost",
                "types", "spell", "deals", "damage", "to", "any",
                "target", "instant", "{r}", "none",
            ])},
            vocab_path,
        )

        config = SealedEncoderConfig(
            vocab_size=16, d_model=8, n_layers=1, n_heads=2,
            ff_dim=16, max_seq_len=32, dropout=0.0, n_pool_queries=2,
        )
        torch.manual_seed(123)
        model = SealedEncoderModel(config)
        SealedEncoderStore().save_encoder(model, config, encoder_dir, version="v1")
        latest = encoder_dir / "latest.pt"
        assert latest.exists()

        parser = build_parser()
        args = parser.parse_args([
            "encode-cards",
            "--cards-path", str(cards_path),
        ])
        rc = run_encode_cards(args)
        assert rc == 0

        npz_path = cards_path / "l" / "lightning_bolt.npz"
        assert npz_path.exists()
        with np.load(npz_path) as f:
            arr = f["embedding"]
        assert arr.shape == (2 * config.d_model + FEATURE_COUNT,)
    finally:
        os.chdir(cwd)
