"""Argparse-level tests for the ``build-vocab`` subcommand (FR-009)."""

from __future__ import annotations

from sealed.infrastructure.cli import build_parser


class TestBuildVocabArgs:
    def test_defaults(self):
        ns = build_parser().parse_args(["build-vocab"])
        assert ns.cards_folder == "output/cardsfolder/"
        assert ns.vocab_path == "models/sealed/encoder/vocab.txt"
        assert ns.target_size == 5000

    def test_overrides(self):
        ns = build_parser().parse_args([
            "build-vocab",
            "--cards-folder", "/tmp/cards",
            "--vocab-path", "/tmp/vocab.txt",
            "--target-size", "1000",
        ])
        assert ns.cards_folder == "/tmp/cards"
        assert ns.vocab_path == "/tmp/vocab.txt"
        assert ns.target_size == 1000
