"""T005 — Unit tests for CLI validate-embeddings subcommand."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sealed.infrastructure.cli import build_parser, run_validate_embeddings


class TestValidateEmbeddingsParser:
    def test_subcommand_recognised(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings", "--cards-path", "some/path"])
        assert args.command == "validate-embeddings"

    def test_cards_path_default(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings"])
        assert args.cards_path == "output/cardsfolder/"

    def test_cards_path_custom(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings", "--cards-path", "my/cards"])
        assert args.cards_path == "my/cards"

    def test_threshold_accuracy_default(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings"])
        assert args.threshold_accuracy == pytest.approx(0.95)

    def test_threshold_r2_default(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings"])
        assert args.threshold_r2 == pytest.approx(0.85)

    def test_threshold_accuracy_custom(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings", "--threshold-accuracy", "0.99"])
        assert args.threshold_accuracy == pytest.approx(0.99)

    def test_threshold_r2_custom(self):
        parser = build_parser()
        args = parser.parse_args(["validate-embeddings", "--threshold-r2", "0.90"])
        assert args.threshold_r2 == pytest.approx(0.90)


class TestRunValidateEmbeddings:
    def test_missing_cards_path_exits_2(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args([
            "validate-embeddings",
            "--cards-path", str(tmp_path / "nonexistent"),
        ])
        result = run_validate_embeddings(args)
        assert result == 2

    def test_all_pass_exits_0(self, tmp_path):
        from sealed.domain.embedding_probe import ProbeResult, ValidationResult

        parser = build_parser()
        args = parser.parse_args([
            "validate-embeddings",
            "--cards-path", str(tmp_path),
        ])
        tmp_path.mkdir(exist_ok=True)

        passing_result = ValidationResult(
            probe_results=[
                ProbeResult("Is land", 1.0, 0.99, True, 100),
            ],
            n_cards=100,
            n_lands=30,
            all_passed=True,
        )

        with patch("sealed.application.validate_embeddings.ValidateEmbeddingsUseCase") as MockUC:
            MockUC.return_value.execute.return_value = passing_result
            result = run_validate_embeddings(args)

        assert result == 0

    def test_any_fail_exits_1(self, tmp_path):
        from sealed.domain.embedding_probe import ProbeResult, ValidationResult

        parser = build_parser()
        args = parser.parse_args([
            "validate-embeddings",
            "--cards-path", str(tmp_path),
        ])
        tmp_path.mkdir(exist_ok=True)

        failing_result = ValidationResult(
            probe_results=[
                ProbeResult("Is land", 0.50, 0.99, False, 100),
            ],
            n_cards=100,
            n_lands=30,
            all_passed=False,
        )

        with patch("sealed.application.validate_embeddings.ValidateEmbeddingsUseCase") as MockUC:
            MockUC.return_value.execute.return_value = failing_result
            result = run_validate_embeddings(args)

        assert result == 1

    def test_insufficient_data_exits_2(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args([
            "validate-embeddings",
            "--cards-path", str(tmp_path),
        ])
        tmp_path.mkdir(exist_ok=True)

        with patch("sealed.application.validate_embeddings.ValidateEmbeddingsUseCase") as MockUC:
            MockUC.return_value.execute.side_effect = ValueError("insufficient data")
            result = run_validate_embeddings(args)

        assert result == 2
