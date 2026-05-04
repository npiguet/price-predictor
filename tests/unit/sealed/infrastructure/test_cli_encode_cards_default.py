"""Argparse-level tests for the flipped ``encode-cards`` defaults (FR-025)."""

from __future__ import annotations

from sealed.infrastructure.cli import (
    _ENCODE_CARDS_DEFAULT_ENCODER,
    build_parser,
)


class TestEncodeCardsDefaults:
    def test_default_encoder_path_is_sealed(self):
        assert _ENCODE_CARDS_DEFAULT_ENCODER == "models/sealed/encoder/latest.pt"

    def test_default_vocab_path_is_sealed(self):
        ns = build_parser().parse_args(["encode-cards"])
        assert ns.vocab_path == "models/sealed/encoder/vocab.txt"

    def test_explicit_encoder_checkpoint_overrides_default(self):
        ns = build_parser().parse_args([
            "encode-cards",
            "--encoder-checkpoint", "models/price-predictor/transformer/latest.pt",
        ])
        assert ns.encoder_checkpoint == "models/price-predictor/transformer/latest.pt"

    def test_explicit_vocab_path_overrides_default(self):
        ns = build_parser().parse_args([
            "encode-cards",
            "--vocab-path", "models/price-predictor/transformer/vocab.txt",
        ])
        assert ns.vocab_path == "models/price-predictor/transformer/vocab.txt"
