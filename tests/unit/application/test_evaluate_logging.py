"""Tests for evaluate progress messages."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from price_predictor.application.evaluate import EvaluateModelUseCase
from price_predictor.application.train import TrainModelUseCase

EVALUATE_LOGGER = "price_predictor.application.evaluate"
LoggedMessages = Callable[[str], list[str]]


@pytest.fixture
def trained_model_path(card_data_paths: dict[str, Path], tmp_path: Path) -> Path:
    result = TrainModelUseCase().execute(**card_data_paths, output_path=tmp_path)
    return result.model_path


@pytest.fixture
def logged_messages(
    trained_model_path: Path,
    card_data_paths: dict[str, Path],
    caplog: pytest.LogCaptureFixture,
) -> LoggedMessages:
    """Run one evaluation and return the messages the named logger emitted."""

    def run(logger_name: str) -> list[str]:
        with caplog.at_level(logging.INFO, logger=logger_name):
            EvaluateModelUseCase().execute(model_path=trained_model_path, **card_data_paths)
        return [r.message for r in caplog.records]

    return run


class TestEvaluateProgressLogging:
    """Verify that evaluation emits stage-level log messages with counts."""

    def test_evaluate_emits_loading_model_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(EVALUATE_LOGGER)
        assert any("Loading model" in m for m in messages)

    def test_evaluate_emits_prediction_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(EVALUATE_LOGGER)
        assert any("predictions" in m.lower() or "predicting" in m.lower() for m in messages)

    def test_evaluate_emits_complete_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(EVALUATE_LOGGER)
        assert any("Evaluation complete" in m for m in messages)

    def test_evaluate_messages_include_counts(self, logged_messages: LoggedMessages) -> None:
        """Log messages should include numeric counts."""
        messages = logged_messages(EVALUATE_LOGGER)
        messages_with_numbers = [m for m in messages if any(c.isdigit() for c in m)]
        assert len(messages_with_numbers) >= 1, (
            f"Expected at least 1 message with counts, got: {messages}"
        )

    def test_evaluate_all_stages_present(self, logged_messages: LoggedMessages) -> None:
        """Evaluation should emit multiple stage-level messages."""
        messages = logged_messages("price_predictor")
        # Expect at least 4 messages: loading model, parsing, predicting, complete
        assert len(messages) >= 4, (
            f"Expected >= 4 log messages for evaluation, "
            f"got {len(messages)}: {messages}"
        )
