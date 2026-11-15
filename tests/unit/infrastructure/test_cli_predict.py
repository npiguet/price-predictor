"""Tests for predict CLI subcommand (new nested interface).

The old predict command with --types, --mana-cost etc. has been removed (FR-010).
New predict command uses: predict sklearn --file/--card or predict transformer --file/--card.
Full tests for the new predict handlers will be added in US2 (Phase 4).
"""

from __future__ import annotations

import pytest

from price_predictor.infrastructure.cli import build_parser


class TestPredictParserStructure:
    def test_predict_sklearn_subcommand_registered(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["predict", "sklearn", "--file", "card.txt"])
        assert args.command == "predict"
        assert args.model == "sklearn"

    def test_predict_transformer_subcommand_registered(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "predict", "transformer", "--card", "name: test\ntypes: creature",
        ])
        assert args.command == "predict"
        assert args.model == "transformer"

    def test_file_and_card_mutually_exclusive(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["predict", "sklearn", "--file", "a.txt", "--card", "text"])

    def test_missing_file_and_card_exits(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["predict", "sklearn"])

    def test_old_predict_with_types_no_longer_works(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["predict", "--types", "Creature"])


class TestThePredictorsNameTheirModel:
    """``--model-path`` names the model a prediction reads, defaulting to the
    rolling checkpoint each trainer writes by default."""

    def test_the_defaults(self) -> None:
        parser = build_parser()
        assert parser.parse_args(["predict", "sklearn", "--card", "x"]).model_path == (
            "models/price-predictor/sklearn/latest.joblib")
        assert parser.parse_args(["predict", "transformer", "--card", "x"]).model_path == (
            "models/price-predictor/transformer/")

    def test_the_sklearn_prediction_reads_the_named_model(self, monkeypatch, tmp_path) -> None:
        from pathlib import Path

        from price_predictor.application import predict as predict_module
        from price_predictor.infrastructure import cli

        seen = {}

        class FakeUseCase:
            def execute(self, card, model_path):
                seen["model_path"] = model_path
                from types import SimpleNamespace
                return SimpleNamespace(predicted_price_eur=1.0, model_version="v")

        monkeypatch.setattr(predict_module, "PredictPriceUseCase", FakeUseCase)
        monkeypatch.setattr(cli, "_resolve_optional_printing_data", lambda name: None)
        model = tmp_path / "models" / "run" / "latest.joblib"
        args = build_parser().parse_args([
            "predict", "sklearn", "--card", "name: test\ntypes: creature",
            "--model-path", str(model),
        ])
        assert cli.run_predict_sklearn(args) == 0
        assert seen["model_path"] == Path(model)

    def test_serve_names_its_optional_transformer(self) -> None:
        args = build_parser().parse_args(["serve"])
        assert args.transformer_path == "models/price-predictor/transformer/"
        args = build_parser().parse_args(["serve", "--transformer-path", "m/run/"])
        assert args.transformer_path == "m/run/"
