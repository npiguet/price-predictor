"""Evaluate transformer use case: compute accuracy metrics on held-out validation data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from price_predictor.infrastructure.converted_card_parser import parse_converted_cards
from price_predictor.infrastructure.mtgjson_loader import build_name_to_uuids, build_price_map
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_store import load_model

logger = logging.getLogger(__name__)


@dataclass
class TransformerEvalResult:
    """Result of a transformer evaluation run."""

    model_version: str
    mean_absolute_error_eur: float
    median_percentage_error: float
    median_abs_error_log: float
    top_20_overlap: float
    sample_count: int
    per_card: list[dict] | None = None


def _match_cards_to_prices(
    cards: list,
    name_to_uuids: dict,
    price_map: dict,
    output_dir: Path,
) -> list[tuple[str, str, float]]:
    """Match parsed cards to prices and read their converted text files."""
    lower_to_canonical: dict[str, str] = {k.lower(): k for k in name_to_uuids}
    matched = []
    for card in cards:
        card_name_lower = card.name.lower()
        canonical = lower_to_canonical.get(card_name_lower)
        if canonical is None:
            for full_lower, full_canonical in lower_to_canonical.items():
                if full_lower.startswith(card_name_lower + " // "):
                    canonical = full_canonical
                    break
        if canonical is None or canonical not in price_map:
            continue

        slug = card.name.lower().replace(" ", "_").replace(",", "").replace("'", "")
        first_letter = slug[0] if slug else "_"
        text_path = output_dir / first_letter / f"{slug}.txt"
        if text_path.exists():
            text = text_path.read_text(encoding="utf-8")
        else:
            # Fall back: the file might be at the root of output_dir (e.g. in tests)
            flat_path = output_dir / f"{slug}.txt"
            if flat_path.exists():
                text = flat_path.read_text(encoding="utf-8")
            else:
                continue

        matched.append((card.name, text, price_map[canonical]))
    return matched


def evaluate_transformer(
    model_dir: Path,
    output_dir: Path,
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
    cards, parse_errors = parse_converted_cards(output_dir)
    logger.info("Parsed %d cards (%d parse errors)", len(cards), len(parse_errors))
    name_to_uuids = build_name_to_uuids(printings_path)
    price_map = build_price_map(prices_path, name_to_uuids)

    matched = _match_cards_to_prices(cards, name_to_uuids, price_map, output_dir)
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

    # Median percentage error
    pct_errors = np.abs(predicted_prices - actual_prices) / np.maximum(actual_prices, 0.01) * 100
    median_percentage_error = float(np.median(pct_errors))

    # Median absolute error in shifted-log space: median(|log(actual+2) - log(predicted+2)|)
    log_errors = np.abs(np.log(actual_prices + 2) - np.log(predicted_prices + 2))
    median_log_error = float(np.median(log_errors))

    # Top-20% overlap
    n_top = max(1, int(len(actual_prices) * 0.2))
    actual_top_indices = set(np.argsort(actual_prices.flatten())[-n_top:])
    predicted_top_indices = set(np.argsort(predicted_prices.flatten())[-n_top:])
    top_20_overlap = float(len(actual_top_indices & predicted_top_indices) / n_top)

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

    # Model version from directory name
    model_version = model_dir.name or "transformer"

    return TransformerEvalResult(
        model_version=model_version,
        mean_absolute_error_eur=round(mae, 2),
        median_percentage_error=round(median_percentage_error, 1),
        median_abs_error_log=round(median_log_error, 3),
        top_20_overlap=round(top_20_overlap, 2),
        sample_count=len(val_data),
        per_card=per_card,
    )
