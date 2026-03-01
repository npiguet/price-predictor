"""Tests for evaluate progress messages (T038)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from price_predictor.application.evaluate import EvaluateModelUseCase
from price_predictor.application.train import TrainModelUseCase


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures"


@pytest.fixture
def trained_model_path(fixtures_dir: Path, tmp_path: Path) -> Path:
    use_case = TrainModelUseCase()
    result = use_case.execute(
        forge_cards_path=fixtures_dir / "forge_cards",
        prices_path=fixtures_dir / "allprices_sample.json",
        printings_path=fixtures_dir / "allprintings_sample.json",
        output_path=tmp_path,
    )
    return result.model_path


class TestEvaluateProgressLogging:
    """Verify that evaluation emits stage-level log messages with counts."""

    def test_evaluate_emits_loading_model_message(
        self, trained_model_path: Path, fixtures_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.evaluate"):
            use_case = EvaluateModelUseCase()
            use_case.execute(
                model_path=trained_model_path,
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
            )

        messages = [r.message for r in caplog.records]
        assert any("Loading model" in m for m in messages)

    def test_evaluate_emits_prediction_message(
        self, trained_model_path: Path, fixtures_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.evaluate"):
            use_case = EvaluateModelUseCase()
            use_case.execute(
                model_path=trained_model_path,
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
            )

        messages = [r.message for r in caplog.records]
        assert any("predictions" in m.lower() or "predicting" in m.lower() for m in messages)

    def test_evaluate_emits_complete_message(
        self, trained_model_path: Path, fixtures_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.evaluate"):
            use_case = EvaluateModelUseCase()
            use_case.execute(
                model_path=trained_model_path,
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
            )

        messages = [r.message for r in caplog.records]
        assert any("Evaluation complete" in m for m in messages)

    def test_evaluate_messages_include_counts(
        self, trained_model_path: Path, fixtures_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log messages should include numeric counts."""
        with caplog.at_level(logging.INFO, logger="price_predictor.application.evaluate"):
            use_case = EvaluateModelUseCase()
            use_case.execute(
                model_path=trained_model_path,
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
            )

        messages = [r.message for r in caplog.records]
        messages_with_numbers = [m for m in messages if any(c.isdigit() for c in m)]
        assert len(messages_with_numbers) >= 1, (
            f"Expected at least 1 message with counts, got: {messages}"
        )

    def test_evaluate_all_stages_present(
        self, trained_model_path: Path, fixtures_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Evaluation should emit multiple stage-level messages."""
        with caplog.at_level(logging.INFO, logger="price_predictor"):
            use_case = EvaluateModelUseCase()
            use_case.execute(
                model_path=trained_model_path,
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
            )

        messages = [r.message for r in caplog.records]
        # Expect at least 4 messages: loading model, parsing, predicting, complete
        assert len(messages) >= 4, (
            f"Expected >= 4 log messages for evaluation, "
            f"got {len(messages)}: {messages}"
        )
