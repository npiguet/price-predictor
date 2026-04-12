"""End-to-end integration test: train → predict → evaluate pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
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
            output_dir=fixtures_dir / "converted_cards",
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
            output_dir=fixtures_dir / "converted_cards",
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
            output_dir=fixtures_dir / "converted_cards",
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

    def test_train_with_metadata_enrichment_and_predict(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Train sklearn model on fixture data with metadata enrichment,
        verify the model loads and predicts without error, and the feature
        engineering has the new feature count (593 = 93 dense + 500 tfidf)."""
        # Train with the training fixtures (which include enriched cards)
        train_uc = TrainModelUseCase()
        train_result = train_uc.execute(
            output_dir=fixtures_dir / "converted_cards_training",
            prices_path=fixtures_dir / "allprices_sample.json",
            printings_path=fixtures_dir / "allprintings_sample.json",
            output_path=tmp_path / "models",
            test_split=0.2,
            random_seed=42,
        )
        model_path = train_result.model_path
        assert model_path.exists()

        # Load the artifact and verify feature count
        artifact = joblib.load(model_path)
        fe = artifact["feature_engineering"]
        # Dense features should be 95 (76 old + 18 printing data + 1 has_mana_cost)
        # TF-IDF vocab may be smaller than 500 on small fixture data
        tfidf_count = len(fe.tfidf_.vocabulary_)
        dense_count = fe.get_feature_count() - tfidf_count
        assert dense_count == 95  # 76 old dense + 18 printing data + 1 has_mana_cost

        # Predict with a card that has no printing_data -- should still work
        card_without_pd = Card(
            name="Test Creature",
            types=["Creature"],
            mana_cost=ManaCost.parse("1 G"),
            power="2",
            toughness="2",
        )
        predict_uc = PredictPriceUseCase()
        prediction = predict_uc.execute(card_without_pd, model_path)
        assert prediction.predicted_price_eur > 0

        # Predict with a card that has printing_data -- should also work
        from price_predictor.domain.value_objects import PrintingData
        card_with_pd = Card(
            name="Enriched Test",
            types=["Creature"],
            mana_cost=ManaCost.parse("3 R R"),
            power="5",
            toughness="5",
            keywords=["Flying"],
            oracle_text="Flying. When this creature enters, deal 2 damage.",
            ability_count=2,
            printing_data=PrintingData(
                is_reserved=False,
                rarity="rare",
                printings_count=2,
                release_year=2020,
                legalities=["commander", "legacy", "vintage", "modern"],
            ),
        )
        prediction_enriched = predict_uc.execute(card_with_pd, model_path)
        assert prediction_enriched.predicted_price_eur > 0
