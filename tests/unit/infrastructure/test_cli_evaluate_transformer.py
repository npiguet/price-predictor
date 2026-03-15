"""Tests for evaluate transformer CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from price_predictor.infrastructure.cli import build_parser


class TestEvaluateTransformerParser:
    def test_subcommand_registered(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "transformer"])
        assert args.command == "evaluate"
        assert args.model == "transformer"

    def test_default_model_path(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "transformer"])
        assert args.model_path == "./models/transformer/"

    def test_default_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "transformer"])
        assert args.output_dir == "./output"

    def test_default_random_seed(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "transformer"])
        assert args.random_seed == 42

    def test_custom_model_path(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate", "transformer", "--model-path", "custom/path/"])
        assert args.model_path == "custom/path/"


class TestRunEvaluateTransformerNew:
    @patch("price_predictor.application.evaluate_transformer.evaluate_transformer")
    def test_success_prints_json(self, mock_eval, capsys):
        from price_predictor.infrastructure.cli import run_evaluate_transformer_new

        mock_result = MagicMock()
        mock_result.model_version = "transformer"
        mock_result.mean_absolute_error_eur = 1.23
        mock_result.median_percentage_error = 65.2
        mock_result.median_abs_error_log = 0.19
        mock_result.top_20_overlap = 0.45
        mock_result.sample_count = 4032
        mock_result.per_card = None
        mock_eval.return_value = mock_result

        import argparse
        args = argparse.Namespace(
            model_path="./models/transformer/",
            output_dir="./output",
            prices_path="resources/AllPricesToday.json",
            printings_path="resources/AllPrintings.json",
            random_seed=42,
        )
        exit_code = run_evaluate_transformer_new(args)

        assert exit_code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["model_version"] == "transformer"
        assert data["mean_absolute_error_eur"] == 1.23
        assert data["median_percentage_error"] == 65.2
        assert data["median_abs_error_log"] == 0.19
        assert data["top_20_overlap"] == 0.45
        assert data["sample_count"] == 4032
