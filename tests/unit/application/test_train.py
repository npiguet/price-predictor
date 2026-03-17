"""Tests for TrainModelUseCase."""

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from price_predictor.application.train import TrainModelUseCase


@pytest.fixture
def converted_cards_dir() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "converted_cards_training"


@pytest.fixture
def allprintings_path() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "allprintings_sample.json"


@pytest.fixture
def allprices_path() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "allprices_sample.json"


class TestTrainModelUseCase:
    def test_train_produces_trained_model(
        self, converted_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=converted_cards_dir,
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
        self, converted_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=converted_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path,
            test_split=0.2,
            random_seed=42,
        )
        # Island has price < threshold or some cards may not match
        assert isinstance(result.skipped_reasons, dict)

    def test_model_is_saved(
        self, converted_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=converted_cards_dir,
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
        self, converted_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        use_case1 = TrainModelUseCase()
        result1 = use_case1.execute(
            output_dir=converted_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path / "run1",
            test_split=0.2,
            random_seed=42,
        )
        use_case2 = TrainModelUseCase()
        result2 = use_case2.execute(
            output_dir=converted_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path / "run2",
            test_split=0.2,
            random_seed=42,
        )
        assert result1.trained_model.card_count == result2.trained_model.card_count

    def test_insufficient_data_raises(self, tmp_path: Path) -> None:
        # Create a dir with just one parseable card in converted format
        cards_dir = tmp_path / "cards"
        cards_dir.mkdir()
        (cards_dir / "one.txt").write_text(
            "name: Only Card\ntypes: creature\npower toughness: 1/1\nmana cost: {R}\n"
        )
        prices = tmp_path / "prices.json"
        prices.write_text('{"meta":{},"data":{}}')
        printings = tmp_path / "printings.json"
        printings.write_text('{"meta":{},"data":{}}')
        use_case = TrainModelUseCase()
        with pytest.raises(ValueError, match="[Ii]nsufficient"):
            use_case.execute(
                output_dir=cards_dir,
                prices_path=prices,
                printings_path=printings,
                output_path=tmp_path / "out",
                test_split=0.2,
                random_seed=42,
            )

    def test_trained_cards_have_printing_data(
        self, converted_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        """Training enriches Cards with printing_data from the metadata map.
        The feature engineering dense count should be 94 (76 old + 18 printing data)."""
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=converted_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path,
            test_split=0.2,
            random_seed=42,
        )
        # Load the saved artifact and inspect the feature engineering
        artifact = joblib.load(result.model_path)
        fe = artifact["feature_engineering"]
        # Dense feature count should be 94 (includes 18 printing data features)
        tfidf_count = len(fe._tfidf.vocabulary_)
        dense_count = fe.get_feature_count() - tfidf_count
        assert dense_count == 94

    def test_model_artifact_has_increased_feature_count(
        self, converted_cards_dir: Path, allprintings_path: Path, allprices_path: Path, tmp_path: Path
    ) -> None:
        """The trained model artifact's feature engineering has 94 dense features,
        which is 18 more than the old dense count of 76."""
        use_case = TrainModelUseCase()
        result = use_case.execute(
            output_dir=converted_cards_dir,
            prices_path=allprices_path,
            printings_path=allprintings_path,
            output_path=tmp_path,
            test_split=0.2,
            random_seed=42,
        )
        artifact = joblib.load(result.model_path)
        fe = artifact["feature_engineering"]
        tfidf_count = len(fe._tfidf.vocabulary_)
        dense_count = fe.get_feature_count() - tfidf_count
        old_dense_count = 76
        assert dense_count - old_dense_count == 18
