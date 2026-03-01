"""End-to-end integration test: train → predict → evaluate pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.application.evaluate import EvaluateModelUseCase
from price_predictor.application.predict import PredictPriceUseCase
from price_predictor.application.train import TrainModelUseCase
from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost

FIXTURES_DIR = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.mark.integration
class TestEndToEnd:
    def test_full_pipeline(self, tmp_path: Path, fixtures_dir: Path) -> None:
        """Train → predict → evaluate full pipeline."""
        # 1. Train
        train_uc = TrainModelUseCase()
        train_result = train_uc.execute(
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            output_path=tmp_path / "models",
            test_split=0.2,
            random_seed=42,
        )
        model_path = train_result.model_path
        assert model_path.exists()
        assert train_result.trained_model.card_count > 0

        # 2. Predict a known card
        predict_uc = PredictPriceUseCase()
        card = Card(
            name="Grizzly Bears",
            types=["Creature"],
            subtypes=["Bear"],
            mana_cost=ManaCost.parse("1 G"),
            power="2",
            toughness="2",
        )
        prediction = predict_uc.execute(card, model_path)
        assert prediction.predicted_price_eur > 0

        # 3. Predict a made-up card
        made_up = Card(
            name="Imaginary Dragon",
            types=["Creature"],
            subtypes=["Dragon"],
            mana_cost=ManaCost.parse("4 R R"),
            power="6",
            toughness="6",
            keywords=["Flying", "Haste"],
            oracle_text=(
                "Flying, haste. When Imaginary Dragon enters the "
                "battlefield, deal 3 damage to any target."
            ),
            ability_count=2,
        )
        made_up_pred = predict_uc.execute(made_up, model_path)
        assert made_up_pred.predicted_price_eur > 0

        # 4. Evaluate
        eval_uc = EvaluateModelUseCase()
        eval_result = eval_uc.execute(
            model_path=model_path,
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            test_split=0.2,
            random_seed=42,
        )
        assert eval_result.metrics.sample_count > 0
        assert eval_result.metrics.mean_absolute_error_eur >= 0

    def test_predictions_reproducible(self, tmp_path: Path, fixtures_dir: Path) -> None:
        """Same model + same input = same output across multiple runs."""
        train_uc = TrainModelUseCase()
        train_result = train_uc.execute(
            forge_cards_path=fixtures_dir / "forge_cards",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            output_path=tmp_path / "models",
            test_split=0.2,
            random_seed=42,
        )
        card = Card(
            name="Test",
            types=["Instant"],
            mana_cost=ManaCost.parse("R"),
            oracle_text="Deal 3 damage.",
            ability_count=1,
        )
        predict_uc = PredictPriceUseCase()
        r1 = predict_uc.execute(card, train_result.model_path)
        r2 = predict_uc.execute(card, train_result.model_path)
        assert r1.predicted_price_eur == r2.predicted_price_eur
