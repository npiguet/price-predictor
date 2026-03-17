"""Evaluate transformer use case: compute accuracy metrics on held-out validation data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from price_predictor.application.card_enrichment import enrich_card_text
from price_predictor.infrastructure.mtgjson_loader import (
    build_metadata_map,
    build_name_to_uuids,
)
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
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


def _match_texts_to_prices(
    output_dir: Path,
    name_to_uuids: dict,
    price_map: dict,
    metadata_map: dict | None = None,
) -> list[tuple[str, str, float]]:
    """Read converted text files directly and match to prices by name.

    No Card parsing needed — the transformer ingests raw text.
    """
    lower_to_canonical: dict[str, str] = {k.lower(): k for k in name_to_uuids}
    matched = []
    skipped = 0
    for txt_file in sorted(output_dir.rglob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except OSError:
            skipped += 1
            continue

        # Extract card name from the name: line
        card_name = None
        for line in text.splitlines():
            line = line.strip()
            if line.lower().startswith("name:"):
                card_name = line[len("name:"):].strip()
                break

        if not card_name:
            skipped += 1
            continue

        card_name_lower = card_name.lower()
        canonical = lower_to_canonical.get(card_name_lower)
        if canonical is None:
            for full_lower, full_canonical in lower_to_canonical.items():
                if full_lower.startswith(card_name_lower + " // "):
                    canonical = full_canonical
                    break

        if canonical is None or canonical not in price_map:
            continue

        # Enrich text with printing data if metadata available
        if metadata_map and canonical in metadata_map:
            text = enrich_card_text(text, metadata_map[canonical])

        matched.append((card_name, text, price_map[canonical]))

    if skipped > 0:
        logger.info("Skipped %d text files (unreadable or missing name)", skipped)
    return matched


def evaluate_transformer(
    model_dir: Path,
    output_dir: Path,
    prices_path: Path,
    printings_path: Path,
    vocab_path: Path = Path("models/transformer/vocab.txt"),
    random_seed: int = 42,
) -> TransformerEvalResult:
    """Load a saved transformer model and evaluate on the validation split."""
    logger.info("Loading transformer model from %s...", model_dir)
    model, config = load_model(model_dir)
    tokenizer = load_tokenizer(vocab_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Read text files directly and match to prices (with metadata enrichment)
    metadata_map, price_map = build_metadata_map(printings_path, prices_path)
    name_to_uuids, _uuid_meta = build_name_to_uuids(printings_path)

    matched = _match_texts_to_prices(output_dir, name_to_uuids, price_map, metadata_map)
    logger.info("Matched %d cards to texts and prices", len(matched))

    if len(matched) < 2:
        raise ValueError("Insufficient data for evaluation")

    # Re-derive 80/20 split using same seed
    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(
        matched, test_size=0.2, random_state=random_seed
    )

    logger.info("Validation set: %d cards", len(val_data))

    dataset = TransformerTrainingDataset(
        val_data, max_seq_len=config.max_seq_len, tokenizer=tokenizer,
        log_offset=config.log_offset,
    )
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

    # Convert from shifted-log space back to EUR: exp(x) - log_offset
    predicted_prices = np.exp(predictions) - config.log_offset
    actual_prices = np.exp(targets) - config.log_offset

    # Clamp to non-negative
    predicted_prices = np.maximum(predicted_prices, 0.0)

    # Compute metrics
    abs_errors = np.abs(predicted_prices - actual_prices)
    mae = float(np.mean(abs_errors))

    # Median percentage error
    pct_errors = np.abs(predicted_prices - actual_prices) / np.maximum(actual_prices, 0.01) * 100
    median_percentage_error = float(np.median(pct_errors))

    # Median absolute error in shifted-log space
    log_errors = np.abs(np.log(actual_prices + config.log_offset) - np.log(predicted_prices + config.log_offset))
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
