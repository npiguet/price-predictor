"""Evaluate transformer use case: compute accuracy metrics on held-out validation data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from price_predictor.infrastructure.forge_parser import parse_forge_cards
from price_predictor.infrastructure.mtgjson_loader import build_name_to_uuids, build_price_map
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_store import load_model

logger = logging.getLogger(__name__)


@dataclass
class TransformerEvalResult:
    """Result of a transformer evaluation run."""

    model_path: Path
    mean_absolute_error_eur: float
    median_abs_error_log: float
    sample_count: int
    per_card: list[dict] | None = None


def _match_cards_to_texts(
    cards: list,
    name_to_uuids: dict,
    price_map: dict,
    output_dir: Path,
) -> list[tuple[str, str, float]]:
    """Match cards to their converted text files and prices."""
    matched = []
    for card in cards:
        card_name = card.name
        if card_name not in name_to_uuids:
            for full_name in name_to_uuids:
                if full_name.startswith(card_name + " // "):
                    card_name = full_name
                    break
        if card_name not in price_map:
            continue

        slug = card.name.lower().replace(" ", "_").replace(",", "").replace("'", "")
        first_letter = slug[0] if slug else "_"
        text_path = output_dir / first_letter / f"{slug}.txt"
        if not text_path.exists():
            continue

        text = text_path.read_text(encoding="utf-8")
        matched.append((card.name, text, price_map[card_name]))
    return matched


def evaluate_transformer(
    model_dir: Path,
    output_dir: Path,
    forge_cards_path: Path,
    prices_path: Path,
    printings_path: Path,
    random_seed: int = 42,
) -> TransformerEvalResult:
    """Load a saved transformer model and evaluate on the validation split."""
    logger.info("Loading transformer model from %s...", model_dir)
    model, config = load_model(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Re-derive the dataset (same pipeline as training)
    cards, parse_errors = parse_forge_cards(forge_cards_path)
    logger.info("Parsed %d cards (%d parse errors)", len(cards), len(parse_errors))
    name_to_uuids = build_name_to_uuids(printings_path)
    price_map = build_price_map(prices_path, name_to_uuids)

    matched = _match_cards_to_texts(cards, name_to_uuids, price_map, output_dir)
    logger.info("Matched %d cards to texts and prices", len(matched))

    if len(matched) < 2:
        raise ValueError("Insufficient data for evaluation")

    # Re-derive 80/20 split using same seed
    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(
        matched, test_size=0.2, random_state=random_seed
    )

    logger.info("Validation set: %d cards", len(val_data))

    dataset = TransformerTrainingDataset(val_data, max_seq_len=config.max_seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"]

            outputs = model(input_ids, attention_mask)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets)

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    # Convert from shifted-log space back to EUR: exp(x) - 2
    predicted_prices = np.exp(predictions) - 2
    actual_prices = np.exp(targets) - 2

    # Clamp to non-negative
    predicted_prices = np.maximum(predicted_prices, 0.0)

    # Compute metrics
    abs_errors = np.abs(predicted_prices - actual_prices)
    mae = float(np.mean(abs_errors))

    # Median absolute error in shifted-log space: median(|log(actual+2) - log(predicted+2)|)
    log_errors = np.abs(np.log(actual_prices + 2) - np.log(predicted_prices + 2))
    median_log_error = float(np.median(log_errors))

    # Per-card breakdown
    per_card = []
    for i, (name, _text, _price) in enumerate(val_data):
        per_card.append({
            "name": name,
            "actual_price_eur": round(float(actual_prices[i]), 2),
            "predicted_price_eur": round(float(predicted_prices[i]), 2),
            "absolute_error_eur": round(float(abs_errors[i]), 2),
        })

    logger.info(
        "Evaluation complete — MAE: €%.2f, median abs error (log): %.3f",
        mae, median_log_error,
    )

    return TransformerEvalResult(
        model_path=model_dir,
        mean_absolute_error_eur=round(mae, 2),
        median_abs_error_log=round(median_log_error, 3),
        sample_count=len(val_data),
        per_card=per_card,
    )
