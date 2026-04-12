"""Train model use case: orchestrate card parsing, price joining, and model training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from price_predictor.application.converted_card_dataset import load_parsed_cards
from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.domain.entities import TrainedModel
from price_predictor.infrastructure.model_store import save_model
from price_predictor.infrastructure.mtgjson_loader import build_metadata_map

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """Result of a training run."""

    trained_model: TrainedModel
    model_path: Path
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    cards_skipped: int = 0


class TrainModelUseCase:
    """Orchestrate training: parse cards, join prices, train model, save artifact."""

    def execute(
        self,
        output_dir: Path,
        prices_path: Path,
        printings_path: Path,
        output_path: Path,
        test_split: float = 0.2,
        random_seed: int = 42,
    ) -> TrainResult:
        metadata_map, price_map = build_metadata_map(printings_path, prices_path)
        dataset = load_parsed_cards(output_dir, price_map, metadata_map)
        logger.info(
            "Parsed %d cards (%d parse errors)",
            len(dataset.cards) + dataset.skipped_reasons.get("no_printings_match", 0),
            dataset.skipped_reasons.get("parse_error", 0),
        )

        if len(dataset.cards) < 2:
            raise ValueError(
                f"Insufficient training data: only {len(dataset.cards)} cards with prices. "
                "Need at least 2 cards to train."
            )

        training_cards = dataset.cards
        training_prices = dataset.prices
        skipped_reasons = dataset.skipped_reasons
        total_skipped = dataset.total_skipped

        logger.info(
            "Training on %d cards, skipped %d", len(training_cards), total_skipped
        )

        # 5. Log-transform prices
        log_prices = np.log(np.array(training_prices))

        # 6. Fit feature engineering
        fe = FeatureEngineering(random_seed=random_seed)
        fe.fit(training_cards)

        # 7. Transform to feature matrix
        X = fe.transform(training_cards)
        y = log_prices

        logger.info("Feature engineering complete (%d features)", X.shape[1])

        # 8. Train/test split
        if len(training_cards) >= 5:
            X_train, _X_test, y_train, _y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_seed
            )
        else:
            # Too few samples for a meaningful split
            X_train, y_train = X, y

        # 9. Train model
        logger.info("Training model...")
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_seed,
        )
        model.fit(X_train, y_train)
        logger.info("Model training complete")

        # 10. Save model + feature engineering
        output_path.mkdir(parents=True, exist_ok=True)
        version, model_path = save_model(
            {"model": model, "feature_engineering": fe},
            output_path,
        )
        logger.info("Model saved: %s", version)

        # 11. Build metadata
        trained_model = TrainedModel(
            model_version=version,
            training_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            card_count=len(training_cards),
            price_range_min_eur=min(training_prices),
            price_range_max_eur=max(training_prices),
        )

        return TrainResult(
            trained_model=trained_model,
            model_path=model_path,
            skipped_reasons=skipped_reasons,
            cards_skipped=total_skipped,
        )
