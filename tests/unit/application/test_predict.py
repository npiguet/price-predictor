"""Tests for PredictPriceUseCase."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.application.predict import PredictPriceUseCase
from price_predictor.application.train import TrainModelUseCase
from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost
from price_predictor.infrastructure.model_store import ModelNotFoundError


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    """Train a small model on fixture data and return the model path."""
    fixtures = Path(__file__).parents[2] / "fixtures"
    use_case = TrainModelUseCase()
    result = use_case.execute(
        forge_cards_path=fixtures / "forge_cards",
        prices_path=fixtures / "allprices_sample.json",
        printings_path=fixtures / "allprintings_sample.json",
        output_path=tmp_path,
        test_split=0.2,
        random_seed=42,
    )
    return result.model_path


class TestPredictPriceUseCase:
    def test_predict_with_complete_attributes(self, trained_model_path: Path) -> None:
        card = Card(
            name="Test Creature",
            types=["Creature"],
            subtypes=["Bear"],
            mana_cost=ManaCost.parse("1 G"),
            power="2",
            toughness="2",
            keywords=["Trample"],
            oracle_text="Trample",
        )
        use_case = PredictPriceUseCase()
        result = use_case.execute(card, trained_model_path)
        assert result.predicted_price_eur > 0
        assert result.model_version

    def test_predict_with_partial_attributes(self, trained_model_path: Path) -> None:
        card = Card(
            name="Minimal",
            types=["Creature"],
            mana_cost=ManaCost.parse("2 W"),
        )
        use_case = PredictPriceUseCase()
        result = use_case.execute(card, trained_model_path)
        assert result.predicted_price_eur > 0

    def test_predict_made_up_card(self, trained_model_path: Path) -> None:
        card = Card(
            name="Completely Made Up",
            types=["Enchantment"],
            mana_cost=ManaCost.parse("3 B B"),
            oracle_text="Whenever a creature dies, draw a card and lose 1 life.",
            ability_count=2,
        )
        use_case = PredictPriceUseCase()
        result = use_case.execute(card, trained_model_path)
        assert result.predicted_price_eur > 0

    def test_same_input_same_output(self, trained_model_path: Path) -> None:
        """FR-005: Reproducible predictions."""
        card = Card(
            name="Bolt",
            types=["Instant"],
            mana_cost=ManaCost.parse("R"),
            oracle_text="Deal 3 damage to any target.",
            ability_count=1,
        )
        use_case = PredictPriceUseCase()
        r1 = use_case.execute(card, trained_model_path)
        r2 = use_case.execute(card, trained_model_path)
        assert r1.predicted_price_eur == r2.predicted_price_eur

    def test_model_not_found_raises(self) -> None:
        card = Card(name="Test", types=["Creature"])
        use_case = PredictPriceUseCase()
        with pytest.raises(ModelNotFoundError):
            use_case.execute(card, Path("nonexistent/model.joblib"))
