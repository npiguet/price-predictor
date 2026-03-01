"""Predict price use case: load model and predict card price."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.domain.entities import Card, PriceEstimate
from price_predictor.infrastructure.model_store import load_model


class PredictPriceUseCase:
    """Load a trained model and predict the EUR price for a card."""

    def execute(self, card: Card, model_path: Path) -> PriceEstimate:
        """Predict the price for a single card.

        Args:
            card: Card entity with attributes to predict from.
            model_path: Path to the trained model .joblib file.

        Returns:
            PriceEstimate with predicted EUR price and model version.
        """
        artifact = load_model(model_path)
        model = artifact["model"]
        fe: FeatureEngineering = artifact["feature_engineering"]

        # Transform card to feature vector
        X = fe.transform([card])

        # Predict log-price and exp-transform back to EUR
        log_price = model.predict(X)[0]
        predicted_price = float(np.exp(log_price))

        # Extract version from model path filename
        model_version = model_path.stem
        if model_version == "latest":
            model_version = "latest"

        return PriceEstimate(
            predicted_price_eur=round(predicted_price, 2),
            model_version=model_version,
        )
