"""Tests for train progress messages (T037)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from price_predictor.application.train import TrainModelUseCase


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures"


class TestTrainProgressLogging:
    """Verify that training emits stage-level log messages with counts."""

    def test_train_emits_parsed_message(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.train"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # Stage 1: Parsed cards
        assert any("Parsed" in m and "cards" in m for m in messages)

    def test_train_emits_mapping_message(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.train"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # Stage 2: Built name-to-UUID mapping (delegated to mtgjson_loader)
        # Stage 3: Loaded price data (delegated to mtgjson_loader)
        # Stage 4: Matched cards to prices
        assert any("Matched" in m or "Training on" in m for m in messages)

    def test_train_emits_feature_engineering_message(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.train"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # Stage 5: Feature engineering complete
        assert any("Feature engineering" in m or "features" in m for m in messages)

    def test_train_emits_model_training_message(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.train"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # Stage 6: Model training complete
        assert any("training complete" in m.lower() for m in messages)

    def test_train_emits_model_saved_message(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="price_predictor.application.train"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # Stage 7: Model saved
        assert any("Model saved" in m for m in messages)

    def test_train_messages_include_counts(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Log messages should include numeric counts for user feedback."""
        with caplog.at_level(logging.INFO, logger="price_predictor.application.train"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # At least some messages should contain numbers
        messages_with_numbers = [m for m in messages if any(c.isdigit() for c in m)]
        assert len(messages_with_numbers) >= 2, (
            f"Expected at least 2 messages with counts, got: {messages}"
        )

    def test_train_all_stages_present(
        self, fixtures_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """All 7 training stages should emit a log message."""
        with caplog.at_level(logging.INFO, logger="price_predictor"):
            use_case = TrainModelUseCase()
            use_case.execute(
                forge_cards_path=fixtures_dir / "forge_cards",
                prices_path=fixtures_dir / "allprices_sample.json",
                printings_path=fixtures_dir / "allprintings_sample.json",
                output_path=tmp_path,
            )

        messages = [r.message for r in caplog.records]
        # We expect at least 5 info-level messages across the training pipeline
        assert len(messages) >= 5, (
            f"Expected at least 5 log messages for training stages, got {len(messages)}: {messages}"
        )
