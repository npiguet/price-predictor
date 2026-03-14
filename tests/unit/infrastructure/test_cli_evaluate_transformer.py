"""Tests for evaluate-transformer CLI subcommand."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from price_predictor.infrastructure.cli import build_parser


class TestEvaluateTransformerParser:
    def test_subcommand_registered(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer"])
        assert args.command == "evaluate-transformer"

    def test_default_model_path(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer"])
        assert args.model_path == "models/transformer/"

    def test_default_output_dir(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer"])
        assert args.output_dir == "output/"

    def test_default_random_seed(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer"])
        assert args.random_seed == 42

    def test_default_output_csv_is_none(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer"])
        assert args.output_csv is None

    def test_custom_model_path(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer", "--model-path", "custom/path/"])
        assert args.model_path == "custom/path/"

    def test_custom_output_csv(self):
        parser = build_parser()
        args = parser.parse_args(["evaluate-transformer", "--output-csv", "results.csv"])
        assert args.output_csv == "results.csv"


class TestRunEvaluateTransformer:
    @patch("price_predictor.application.evaluate_transformer.evaluate_transformer")
    def test_success_prints_json(self, mock_eval, capsys):
        from price_predictor.infrastructure.cli import run_evaluate_transformer

        mock_result = MagicMock()
        mock_result.model_path = Path("models/transformer/model.pt")
        mock_result.mean_absolute_error_eur = 1.23
        mock_result.median_percentage_error = 42.5
        mock_result.sample_count = 4032
        mock_result.per_card = None
        mock_eval.return_value = mock_result

        import argparse
        args = argparse.Namespace(
            model_path="models/transformer/",
            output_dir="output/",
            prices_path="resources/AllPricesToday.json",
            printings_path="resources/AllPrintings.json",
            forge_cards_path="../forge/forge-gui/res/cardsfolder/",
            random_seed=42,
            output_csv=None,
        )
        exit_code = run_evaluate_transformer(args)

        assert exit_code == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["mean_absolute_error_eur"] == 1.23
        assert data["median_percentage_error"] == 42.5
        assert data["sample_count"] == 4032
