"""Tests for train-transformer CLI subcommand."""

from __future__ import annotations

import pytest

from price_predictor.infrastructure.cli import build_parser


class TestTrainTransformerParser:
    def test_subcommand_registered(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.command == "train-transformer"

    def test_default_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.output_dir == "output/"

    def test_default_model_output(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.model_output == "models/transformer/"

    def test_default_batch_size(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.batch_size == 64

    def test_default_epochs(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.epochs == 20

    def test_default_lr(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.lr == 1e-4

    def test_default_patience(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.patience == 5

    def test_default_random_seed(self):
        parser = build_parser()
        args = parser.parse_args(["train-transformer"])
        assert args.random_seed == 42

    def test_custom_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "train-transformer",
            "--batch-size", "32",
            "--epochs", "10",
            "--lr", "0.001",
            "--patience", "3",
        ])
        assert args.batch_size == 32
        assert args.epochs == 10
        assert args.lr == 0.001
        assert args.patience == 3
