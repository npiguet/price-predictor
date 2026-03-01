"""Evaluate model use case: compute accuracy metrics on held-out test data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.domain.entities import EvaluationMetrics
from price_predictor.infrastructure.forge_parser import parse_forge_cards
from price_predictor.infrastructure.model_store import load_model
from price_predictor.infrastructure.mtgjson_loader import build_name_to_uuids, build_price_map

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
        forge_cards_path: Path,
        prices_path: Path,
        printings_path: Path,
        test_split: float = 0.2,
        random_seed: int = 42,
    ) -> EvaluateResult:
        # Load model
        artifact = load_model(model_path)
        model = artifact["model"]
        fe: FeatureEngineering = artifact["feature_engineering"]

        # Re-derive the dataset (same pipeline as training)
        cards, _ = parse_forge_cards(forge_cards_path)
        name_to_uuids = build_name_to_uuids(printings_path)
        price_map = build_price_map(prices_path, name_to_uuids)

        eval_cards = []
        eval_prices = []
        for card in cards:
            card_name = card.name
            if card_name not in name_to_uuids:
                for full_name in name_to_uuids:
                    if full_name.startswith(card_name + " // "):
                        card_name = full_name
                        break
            if card_name in price_map:
                eval_cards.append(card)
                eval_prices.append(price_map[card_name])

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
        X_test = fe.transform(test_cards)
        log_predicted = model.predict(X_test)
        predicted_prices = np.exp(log_predicted)

        # Compute metrics
        abs_errors = np.abs(predicted_prices - test_prices)
        mean_absolute_error = float(np.mean(abs_errors))

        pct_errors = np.abs(predicted_prices - test_prices) / test_prices * 100
        median_percentage_error = float(np.median(pct_errors))

        # Top-20% overlap
        n_top = max(1, int(len(test_prices) * 0.2))
        actual_top_indices = set(np.argsort(test_prices)[-n_top:])
        predicted_top_indices = set(np.argsort(predicted_prices)[-n_top:])
        overlap = len(actual_top_indices & predicted_top_indices) / n_top
        top_20_overlap = float(overlap)

        metrics = EvaluationMetrics(
            mean_absolute_error_eur=round(mean_absolute_error, 2),
            median_percentage_error=round(median_percentage_error, 1),
            top_20_overlap=round(top_20_overlap, 2),
            sample_count=len(test_cards),
        )

        # Per-card breakdown
        per_card = []
        for i, card in enumerate(test_cards):
            per_card.append({
                "name": card.name,
                "actual_price_eur": round(float(test_prices[i]), 2),
                "predicted_price_eur": round(float(predicted_prices[i]), 2),
                "absolute_error_eur": round(float(abs_errors[i]), 2),
            })

        model_version = model_path.stem

        return EvaluateResult(
            metrics=metrics,
            model_version=model_version,
            per_card=per_card,
        )
