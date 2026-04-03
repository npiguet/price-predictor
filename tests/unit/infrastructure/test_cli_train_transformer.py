"""Tests for train transformer CLI subcommand."""

from __future__ import annotations

import pytest

from price_predictor.infrastructure.cli import build_parser


class TestTrainTransformerParser:
    def test_subcommand_registered(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.command == "train"
        assert args.model == "transformer"

    def test_default_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.output_dir == "./output"

    def test_default_model_output(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.model_output == "./models/transformer/"

    def test_default_batch_size(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.batch_size == 64

    def test_default_epochs(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.epochs == 20

    def test_default_lr(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.lr == 1e-4

    def test_default_patience(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.patience == 5

    def test_default_random_seed(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.random_seed == 42

    def test_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "train", "transformer",
            "--batch-size", "32",
            "--epochs", "10",
            "--lr", "0.001",
            "--patience", "3",
        ])
        assert args.batch_size == 32
        assert args.epochs == 10
        assert args.lr == 0.001
        assert args.patience == 3


# ─────────────────────────────────────────────────────────────────────────────
# T008 — --aux-lambda CLI flag
# ─────────────────────────────────────────────────────────────────────────────

class TestAuxLambdaCLIFlag:
    """T008: --aux-lambda flag is parsed as float and passed to train_transformer()."""

    def test_default_aux_lambda_is_zero(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer"])
        assert args.aux_lambda == pytest.approx(0.0)

    def test_aux_lambda_parsed_as_float(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer", "--aux-lambda", "0.2"])
        assert args.aux_lambda == pytest.approx(0.2)

    def test_aux_lambda_other_values(self):
        parser = build_parser()
        args = parser.parse_args(["train", "transformer", "--aux-lambda", "0.5"])
        assert args.aux_lambda == pytest.approx(0.5)

    def test_aux_lambda_passed_to_train_function(self, tmp_path):
        """run_train_transformer_new() must pass aux_lambda to train_transformer()."""
        from unittest.mock import patch, MagicMock
        from price_predictor.infrastructure.cli import run_train_transformer_new
        import argparse

        vocab_path = tmp_path / "vocab.txt"
        vocab_path.write_text("[PAD]\n[UNK]\n", encoding="utf-8")

        args = argparse.Namespace(
            vocab_path=str(vocab_path),
            output_dir="./output",
            prices_path="resources/AllPricesToday.json",
            printings_path="resources/AllPrintings.json",
            model_output="./models/price-predictor/transformer/",
            batch_size=64,
            epochs=100,
            lr=1e-4,
            patience=20,
            random_seed=42,
            log_offset=2.0,
            dropout=0.1,
            sampler_exponent=1.0,
            d_model=256,
            n_layers=2,
            n_heads=4,
            ff_dim=1024,
            aux_lambda=0.3,
        )

        with patch("price_predictor.application.train_transformer.train_transformer") as mock_train:
            mock_train.return_value = MagicMock()
            run_train_transformer_new(args)
            call_kwargs = mock_train.call_args[1]
            assert "aux_lambda" in call_kwargs
            assert call_kwargs["aux_lambda"] == pytest.approx(0.3)
