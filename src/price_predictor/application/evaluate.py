"""Evaluate model use case: compute accuracy metrics on held-out test data."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.application.metrics import compute_regression_metrics
from price_predictor.domain.card_name_resolver import CardNameResolver
from price_predictor.domain.entities import EvaluationMetrics
from price_predictor.infrastructure.converted_card_parser import parse_converted_cards
from price_predictor.infrastructure.model_store import load_model
from price_predictor.infrastructure.mtgjson_loader import (
    build_metadata_map,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluateResult:
    """Result of an evaluation run."""

    metrics: EvaluationMetrics
    model_version: str
    per_card: list[dict] | None = None


class EvaluateModelUseCase:
    """Load a trained model, re-derive test split, and compute accuracy metrics."""

    def execute(
        self,
        model_path: Path,
        output_dir: Path,
        prices_path: Path,
        printings_path: Path,
        test_split: float = 0.2,
        random_seed: int = 42,
    ) -> EvaluateResult:
        # Load model
        logger.info("Loading model from %s...", model_path)
        artifact = load_model(model_path)
        model = artifact["model"]
        fe: FeatureEngineering = artifact["feature_engineering"]

        # Re-derive the dataset (same pipeline as training)
        cards, parse_errors = parse_converted_cards(output_dir)
        logger.info("Parsed %d cards (%d parse errors)", len(cards), len(parse_errors))
        metadata_map, price_map = build_metadata_map(printings_path, prices_path)
        resolver = CardNameResolver(price_map, metadata_map)

        eval_cards = []
        eval_prices = []
        for card in cards:
            resolved = resolver.resolve(card.name)
            if resolved is None:
                continue
            enriched_card = replace(card, printing_data=resolved.printing_data)
            eval_cards.append(enriched_card)
            eval_prices.append(resolved.price_eur)

        logger.info(
            "Matched %d cards to prices, skipped %d",
            len(eval_cards), len(cards) - len(eval_cards),
        )

        if len(eval_cards) < 2:
            raise ValueError("Insufficient data for evaluation")

        # Re-derive test split using same seed
        all_indices = list(range(len(eval_cards)))
        _, test_indices = train_test_split(
            all_indices, test_size=test_split, random_state=random_seed
        )

        test_cards = [eval_cards[i] for i in test_indices]
        test_prices = np.array([eval_prices[i] for i in test_indices])

        if len(test_cards) == 0:
            raise ValueError("Test set is empty after split")

        # Predict
        logger.info("Computing predictions on test set (%d cards)...", len(test_cards))
        X_test = fe.transform(test_cards)
        log_predicted = model.predict(X_test)
        predicted_prices = np.exp(log_predicted)

        metrics = compute_regression_metrics(test_prices, predicted_prices)

        abs_errors = np.abs(predicted_prices - test_prices)
        per_card = []
        for i, card in enumerate(test_cards):
            per_card.append({
                "name": card.name,
                "actual_price_eur": round(float(test_prices[i]), 2),
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
