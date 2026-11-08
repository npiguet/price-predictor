"""Evaluate model use case: compute accuracy metrics on held-out test data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from price_predictor.application.converted_card_dataset import load_parsed_cards
from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.application.metrics import compute_regression_metrics
from price_predictor.domain.entities import EvaluationMetrics
from price_predictor.infrastructure.model_store import load_model
from price_predictor.infrastructure.mtgjson_loader import build_printing_data_map

logger = logging.getLogger(__name__)


@dataclass
class EvaluateResult:
    """Result of an evaluation run."""

    metrics: EvaluationMetrics
    model_version: str
    per_card: list[dict] | None = None


class EvaluateModelUseCase:
    """Load a trained model, re-derive the held-out split, and compute accuracy metrics."""

    def execute(
        self,
        model_path: Path,
        output_dir: Path,
        prices_path: Path,
        printings_path: Path,
        held_out_fraction: float = 0.2,
        random_seed: int = 42,
    ) -> EvaluateResult:
        # Load model
        logger.info("Loading model from %s...", model_path)
        artifact = load_model(model_path)
        model = artifact["model"]
        fe: FeatureEngineering = artifact["feature_engineering"]

        printing_data_map, price_map = build_printing_data_map(printings_path, prices_path)
        dataset = load_parsed_cards(output_dir, price_map, printing_data_map)
        eval_cards = dataset.cards
        eval_prices = dataset.prices

        if len(eval_cards) < 2:
            raise ValueError("Insufficient data for evaluation")

        # Re-derive the held-out split using the same seed
        all_indices = list(range(len(eval_cards)))
        _, held_out_indices = train_test_split(
            all_indices, test_size=held_out_fraction, random_state=random_seed
        )

        held_out_cards = [eval_cards[i] for i in held_out_indices]
        held_out_prices = np.array([eval_prices[i] for i in held_out_indices])

        if len(held_out_cards) == 0:
            raise ValueError("Held-out split is empty")

        # Predict
        logger.info(
            "Computing predictions on the held-out split (%d cards)...", len(held_out_cards),
        )
        X_held_out = fe.transform(held_out_cards)
        log_predicted = model.predict(X_held_out)
        predicted_prices = np.exp(log_predicted)

        metrics = compute_regression_metrics(held_out_prices, predicted_prices)

        abs_errors = np.abs(predicted_prices - held_out_prices)
        per_card = []
        for i, card in enumerate(held_out_cards):
            per_card.append({
                "name": card.name,
                "actual_price_eur": round(float(held_out_prices[i]), 2),
                "predicted_price_eur": round(float(predicted_prices[i]), 2),
                "absolute_error_eur": round(float(abs_errors[i]), 2),
            })

        logger.info("Evaluation complete")

        model_version = model_path.stem

        return EvaluateResult(
            metrics=metrics,
            model_version=model_version,
            per_card=per_card,
        )
