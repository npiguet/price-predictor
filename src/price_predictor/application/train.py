"""Train model use case: orchestrate card parsing, price joining, and model training."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from price_predictor.application.feature_engineering import FeatureEngineering
from price_predictor.domain.entities import TrainedModel
from price_predictor.infrastructure.forge_parser import parse_forge_cards
from price_predictor.infrastructure.model_store import save_model
from price_predictor.infrastructure.mtgjson_loader import build_name_to_uuids, build_price_map

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
        forge_cards_path: Path,
        prices_path: Path,
        printings_path: Path,
        output_path: Path,
        test_split: float = 0.2,
        random_seed: int = 42,
    ) -> TrainResult:
        # 1. Parse Forge cards
        cards, parse_errors = parse_forge_cards(forge_cards_path)
        logger.info("Parsed %d cards, %d parse errors", len(cards), len(parse_errors))

        # 2. Build name→UUIDs mapping
        name_to_uuids = build_name_to_uuids(printings_path)

        # 3. Build price map
        price_map = build_price_map(prices_path, name_to_uuids)

        # 4. Join cards to prices
        training_cards = []
        training_prices = []
        skipped_reasons: dict[str, int] = {
            "parse_error": len(parse_errors),
            "no_printings_match": 0,
            "no_price": 0,
        }

        for card in cards:
            # For split cards, look up by front face name and also try "Front // Back"
            card_name = card.name
            if card_name not in name_to_uuids:
                # Try split card naming convention
                found = False
                for full_name in name_to_uuids:
                    if full_name.startswith(card_name + " // "):
                        card_name = full_name
                        found = True
                        break
                if not found:
                    skipped_reasons["no_printings_match"] += 1
                    continue

            if card_name not in price_map:
                skipped_reasons["no_price"] += 1
                continue

            training_cards.append(card)
            training_prices.append(price_map[card_name])

        total_skipped = sum(skipped_reasons.values())

        if len(training_cards) < 2:
            raise ValueError(
                f"Insufficient training data: only {len(training_cards)} cards with prices. "
                "Need at least 2 cards to train."
            )

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

        # 8. Train/test split
        if len(training_cards) >= 5:
            X_train, _X_test, y_train, _y_test = train_test_split(
                X, y, test_size=test_split, random_state=random_seed
            )
        else:
            # Too few samples for a meaningful split
            X_train, y_train = X, y

        # 9. Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_seed,
        )
        model.fit(X_train, y_train)

        # 10. Save model + feature engineering
        output_path.mkdir(parents=True, exist_ok=True)
        version, model_path = save_model(
            {"model": model, "feature_engineering": fe},
            output_path,
        )

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
