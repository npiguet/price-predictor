"""Tests for TrainModelUseCase."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
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


@pytest.fixture
def train_kwargs(
    converted_cards_dir: Path,
    allprintings_path: Path,
    allprices_path: Path,
    tmp_path: Path,
) -> dict:
    return {
        "output_dir": converted_cards_dir,
        "prices_path": allprices_path,
        "printings_path": allprintings_path,
        "output_path": tmp_path,
        "test_split": 0.2,
        "random_seed": 42,
    }


class TestTrainModelUseCase:
    def test_train_produces_trained_model(self, train_kwargs: dict) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(**train_kwargs)
        assert result.trained_model.card_count > 0
        assert result.trained_model.model_version
        assert result.trained_model.price_range_min_eur > 0
        assert result.trained_model.price_range_max_eur > result.trained_model.price_range_min_eur
        assert result.model_path.exists()

    def test_skip_report_has_reasons(self, train_kwargs: dict) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(**train_kwargs)
        assert isinstance(result.skipped_reasons, dict)

    def test_model_is_saved(self, train_kwargs: dict, tmp_path: Path) -> None:
        use_case = TrainModelUseCase()
        result = use_case.execute(**train_kwargs)
        assert result.model_path.exists()
        latest = tmp_path / "latest.joblib"
        assert latest.exists()

    def test_reproducible_training(self, train_kwargs: dict, tmp_path: Path) -> None:
        use_case1 = TrainModelUseCase()
        result1 = use_case1.execute(**{**train_kwargs, "output_path": tmp_path / "run1"})
        use_case2 = TrainModelUseCase()
        result2 = use_case2.execute(**{**train_kwargs, "output_path": tmp_path / "run2"})
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

    def test_trained_cards_have_printing_data(self, train_kwargs: dict) -> None:
        """Training enriches Cards with printing_data so the printing-data feature
        block is non-zero for cards whose metadata was found."""
        from price_predictor.domain.entities import Card
        from price_predictor.domain.value_objects import ManaCost, PrintingData

        use_case = TrainModelUseCase()
        result = use_case.execute(**train_kwargs)
        artifact = joblib.load(result.model_path)
        fe = artifact["feature_engineering"]

        bare = Card(name="Bare", types=["Creature"], mana_cost=ManaCost.parse("1 G"))
        enriched = Card(
            name="Enriched", types=["Creature"], mana_cost=ManaCost.parse("1 G"),
            printing_data=PrintingData(
                rarity="mythic", printings_count=3, release_year=2020,
                legalities=["modern", "legacy"], is_reserved=True,
            ),
        )
        bare_vec = fe.transform([bare])[0]
        enriched_vec = fe.transform([enriched])[0]
        assert not np.array_equal(bare_vec, enriched_vec)
