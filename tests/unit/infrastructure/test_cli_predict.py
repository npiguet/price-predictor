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
        args = parser.parse_args(["predict", "transformer", "--card", "name: test\ntypes: creature"])
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
