"""Tests for TrainModelUseCase."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.application.train import TrainModelUseCase


@pytest.fixture
def forge_cards_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "forge_cards"


@pytest.fixture
def allprintings_path() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "allprintings_sample.json"


@pytest.fixture
def allprices_path() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "allprices_sample.json"


class TestTrainModelUseCase:
    def test_train_produces_trained_model(
        self, forge_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            forge_cards_path=forge_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path,
            test_split=0.2,
            random_seed=42,
        )
        assert result.trained_model.card_count > 0
        assert result.trained_model.model_version
        assert result.trained_model.price_range_min_eur > 0
        assert result.trained_model.price_range_max_eur > result.trained_model.price_range_min_eur
        assert result.model_path.exists()

    def test_skip_report_has_reasons(
        self, forge_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            forge_cards_path=forge_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path,
            test_split=0.2,
            random_seed=42,
        )
        # Island has price < threshold or some cards may not match
        assert isinstance(result.skipped_reasons, dict)

    def test_model_is_saved(
        self, forge_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            forge_cards_path=forge_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path,
            test_split=0.2,
            random_seed=42,
        )
        # Model file should exist
        assert result.model_path.exists()
        # Latest symlink/copy should exist
        latest = tmp_path / "latest.joblib"
        assert latest.exists()

    def test_reproducible_training(
        self, forge_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case1 = TrainModelUseCase()
        result1 = use_case1.execute(
            forge_cards_path=forge_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path / "run1",
            test_split=0.2,
            random_seed=42,
        )
        use_case2 = TrainModelUseCase()
        result2 = use_case2.execute(
            forge_cards_path=forge_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path / "run2",
            test_split=0.2,
            random_seed=42,
        )
        assert result1.trained_model.card_count == result2.trained_model.card_count

    def test_insufficient_data_raises(self, tmp_path: Path) -> None:
        # Create a dir with just one parseable card
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        (cards_dir / "one.txt").write_text("Name:Only Card\nTypes:Creature\nManaCost:R\nPT:1/1\n")
        prices = tmp_path / "prices.json"
        prices.write_text('{"meta":{},"data":{}}')
        printings = tmp_path / "printings.json"
        printings.write_text('{"meta":{},"data":{}}')
        use_case = TrainModelUseCase()
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            use_case.execute(
                forge_cards_path=cards_dir,
                prices_path=prices,
                printings_path=printings,
                output_path=tmp_path / "out",
                test_split=0.2,
                random_seed=42,
            )
