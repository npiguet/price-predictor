"""Evaluate transformer use case: compute accuracy metrics on held-out validation data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from price_predictor.application.metrics import compute_regression_metrics
from price_predictor.domain.card_name_resolver import CardNameResolver
from price_predictor.domain.tokenizer import extract_card_name
from price_predictor.domain.value_objects import PrintingData
from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_store import load_model

logger = logging.getLogger(__name__)


_PRICE_BUCKETS: list[tuple[float, float, str]] = [
    (0.0,   0.1,          "<€0.10"),
    (0.1,   0.5,          "€0.10–0.50"),
    (0.5,   2.0,          "€0.50–2"),
    (2.0,   10.0,         "€2–10"),
    (10.0,  50.0,         "€10–50"),
    (50.0,  float("inf"), ">€50"),
]


@dataclass
class BucketMetrics:
    """Per-price-bucket evaluation metrics."""

    label: str
    count: int
    median_pct_error: float
    median_log_error: float
    median_signed_log_error: float  # negative = underprediction


@dataclass
class TransformerEvalResult:
    """Result of a transformer evaluation run."""

    model_version: str
    mean_absolute_error_eur: float
    median_percentage_error: float
    median_abs_error_log: float
    top_20_overlap: float
    sample_count: int
    per_bucket: list[BucketMetrics] | None = None
    per_card: list[dict] | None = None


def _match_texts_to_prices(
    output_dir: Path,
    price_map: dict,
    metadata_map: dict | None = None,
) -> list[tuple[str, str, float, PrintingData]]:
    """Read converted text files directly and match to prices and PrintingData by name.

    Returns list of (card_name, text, price_eur, printing_data) tuples.
    """
    resolver = CardNameResolver(price_map, metadata_map)
    matched = []
    skipped = 0
    for txt_file in sorted(output_dir.rglob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8")
        except OSError:
            skipped += 1
            continue

        card_name = extract_card_name(text)
        if not card_name:
            skipped += 1
            continue

        resolved = resolver.resolve(card_name)
        if resolved is None:
            continue

        matched.append(
            (card_name, text, resolved.price_eur, resolved.printing_data)
        )

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

    # Read text files directly and match to prices and PrintingData
    metadata_map, price_map = build_metadata_map(printings_path, prices_path)

    matched = _match_texts_to_prices(output_dir, price_map, metadata_map)
    logger.info("Matched %d cards to texts and prices", len(matched))

    if len(matched) < 2:
        raise ValueError("Insufficient data for evaluation")

    # Re-derive 80/20 split using same seed
    from sklearn.model_selection import train_test_split

    train_data, val_data = train_test_split(
        matched, test_size=0.2, random_state=random_seed
    )

    logger.info("Validation set: %d cards", len(val_data))

    val_tuples = [(n, t, p) for n, t, p, _ in val_data]
    val_pd = [pd for _, _, _, pd in val_data]

    dataset = TransformerTrainingDataset(
        val_tuples, max_seq_len=config.max_seq_len, tokenizer=tokenizer,
        log_offset=config.log_offset, printing_data_list=val_pd,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["target"]
            meta = batch["meta"].to(device)

            outputs = model(input_ids, attention_mask, meta)
            all_predictions.append(outputs.cpu())
            all_targets.append(targets)

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()

    # Convert from shifted-log space back to EUR: exp(x) - log_offset
    predicted_prices = np.maximum(np.exp(predictions) - config.log_offset, 0.0)
    actual_prices = np.exp(targets) - config.log_offset

    metrics = compute_regression_metrics(
        actual_prices, predicted_prices, log_offset=config.log_offset,
    )

    per_bucket = _compute_per_bucket(actual_prices, predicted_prices, config.log_offset)

    abs_errors = np.abs(predicted_prices - actual_prices)
    per_card = []
    for i, (name, _text, _price, _pd) in enumerate(val_data):
        per_card.append({
            "name": name,
            "actual_price_eur": round(float(actual_prices[i]), 2),
            "predicted_price_eur": round(float(predicted_prices[i]), 2),
            "absolute_error_eur": round(float(abs_errors[i]), 2),
        })

    logger.info(
        "Evaluation complete — MAE: €%.2f, median abs error (log): %.3f",
        metrics.mean_absolute_error_eur, metrics.median_abs_error_log,
    )

    # Model version from directory name
    model_version = model_dir.name or "transformer"

    return TransformerEvalResult(
        model_version=model_version,
        mean_absolute_error_eur=metrics.mean_absolute_error_eur,
        median_percentage_error=metrics.median_percentage_error,
        median_abs_error_log=metrics.median_abs_error_log,
        top_20_overlap=metrics.top_20_overlap,
        sample_count=metrics.sample_count,
        per_bucket=per_bucket,
        per_card=per_card,
    )


def _compute_per_bucket(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    log_offset: float,
) -> list[BucketMetrics]:
    """Slice cards by price bucket and report local error metrics."""
    actual = actual_prices.flatten()
    predicted = predicted_prices.flatten()

    safe_actual = np.maximum(actual, 0.01)
    pct_errors = np.abs(predicted - actual) / safe_actual * 100
    log_errors = np.abs(np.log(actual + log_offset) - np.log(predicted + log_offset))
    signed_log_errors = (
        np.log(np.maximum(predicted, 0.01)) - np.log(safe_actual)
    )

    per_bucket = []
    for lo, hi, label in _PRICE_BUCKETS:
        mask = (actual >= lo) & (actual < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        per_bucket.append(BucketMetrics(
            label=label,
            count=n,
            median_pct_error=round(float(np.median(pct_errors[mask])), 1),
            median_log_error=round(float(np.median(log_errors[mask])), 3),
            median_signed_log_error=round(float(np.median(signed_log_errors[mask])), 3),
        ))
    return per_bucket
