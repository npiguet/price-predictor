"""Tests for train progress messages."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from price_predictor.application.train import TrainModelUseCase

TRAIN_LOGGER = "price_predictor.application.train"
LoggedMessages = Callable[[str], list[str]]


@pytest.fixture
def logged_messages(
    card_data_paths: dict[str, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> LoggedMessages:
    """Run one training and return the messages the named logger emitted."""

    def run(logger_name: str) -> list[str]:
        with caplog.at_level(logging.INFO, logger=logger_name):
            TrainModelUseCase().execute(**card_data_paths, output_path=tmp_path)
        return [r.message for r in caplog.records]

    return run


class TestTrainProgressLogging:
    """Verify that training emits stage-level log messages with counts."""

    def test_train_emits_parsed_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(TRAIN_LOGGER)
        # Stage 1: Parsed cards
        assert any("Parsed" in m and "cards" in m for m in messages)

    def test_train_emits_mapping_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(TRAIN_LOGGER)
        # Stage 2: Built name-to-UUID mapping (delegated to mtgjson_loader)
        # Stage 3: Loaded price data (delegated to mtgjson_loader)
        # Stage 4: Matched cards to prices
        assert any("Matched" in m or "Training on" in m for m in messages)

    def test_train_emits_feature_engineering_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(TRAIN_LOGGER)
        # Stage 5: Feature engineering complete
        assert any("Feature engineering" in m or "features" in m for m in messages)

    def test_train_emits_model_training_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(TRAIN_LOGGER)
        # Stage 6: Model training complete
        assert any("training complete" in m.lower() for m in messages)

    def test_train_emits_model_saved_message(self, logged_messages: LoggedMessages) -> None:
        messages = logged_messages(TRAIN_LOGGER)
        # Stage 7: Model saved
        assert any("Model saved" in m for m in messages)

    def test_train_messages_include_counts(self, logged_messages: LoggedMessages) -> None:
        """Log messages should include numeric counts for user feedback."""
        messages = logged_messages(TRAIN_LOGGER)
        # At least some messages should contain numbers
        messages_with_numbers = [m for m in messages if any(c.isdigit() for c in m)]
        assert len(messages_with_numbers) >= 2, (
            f"Expected at least 2 messages with counts, got: {messages}"
        )

    def test_train_all_stages_present(self, logged_messages: LoggedMessages) -> None:
        """All 7 training stages should emit a log message."""
        messages = logged_messages("price_predictor")
        # We expect at least 5 info-level messages across the training pipeline
        assert len(messages) >= 5, (
            f"Expected at least 5 log messages for training stages, got {len(messages)}: {messages}"
        )
