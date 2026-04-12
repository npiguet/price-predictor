"""Evaluate transformer use case: compute accuracy metrics on held-out validation data."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from price_predictor.application.converted_card_dataset import load_training_samples
from price_predictor.application.metrics import compute_regression_metrics
from price_predictor.application.transformer_inference import predict_batch
from price_predictor.domain.price_buckets import REPORTING_BUCKETS
from price_predictor.infrastructure.mtgjson_loader import build_metadata_map
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from price_predictor.infrastructure.transformer_dataset import TransformerTrainingDataset
from price_predictor.infrastructure.transformer_store import load_model

logger = logging.getLogger(__name__)


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

    def format_per_bucket_table(self) -> str:
        """Render the per-price-bucket breakdown as a fixed-width text table."""
        if not self.per_bucket:
            return ""
        lines = [
            "Per-bucket breakdown:",
            f"  {'Bucket':<10} {'n':>6}   {'med%err':>8}   {'med|log|':>9}   {'med_signed_log':>15}",
            f"  {'-'*10} {'-'*6}   {'-'*8}   {'-'*9}   {'-'*15}",
        ]
        for b in self.per_bucket:
            sign = "+" if b.median_signed_log_error >= 0 else ""
            lines.append(
                f"  {b.label:<10} {b.count:>6}    {b.median_pct_error:>7.1f}%"
                f"      {b.median_log_error:>6.3f}"
                f"          {sign}{b.median_signed_log_error:>6.3f}"
            )
        return "\n".join(lines)


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

    metadata_map, price_map = build_metadata_map(printings_path, prices_path)
    samples = load_training_samples(output_dir, price_map, metadata_map)
    logger.info("Matched %d cards to texts and prices", len(samples))

    if len(samples) < 2:
        raise ValueError("Insufficient data for evaluation")

    # Re-derive 80/20 split using same seed
    from sklearn.model_selection import train_test_split

    _train_data, val_data = train_test_split(
        samples, test_size=0.2, random_state=random_seed
    )

    logger.info("Validation set: %d cards", len(val_data))

    dataset = TransformerTrainingDataset(
        [(s.name, s.text, s.price_eur) for s in val_data],
        max_seq_len=config.max_seq_len,
        tokenizer=tokenizer,
        log_offset=config.log_offset,
        printing_data_list=[s.printing_data for s in val_data],
    )

    predictions, targets = predict_batch(model, dataset, device)

    predicted_prices = np.maximum(np.exp(predictions) - config.log_offset, 0.0)
    actual_prices = np.exp(targets) - config.log_offset

    metrics = compute_regression_metrics(
        actual_prices, predicted_prices, log_offset=config.log_offset,
    )

    per_bucket = _compute_per_bucket(actual_prices, predicted_prices, config.log_offset)

    abs_errors = np.abs(predicted_prices - actual_prices)
    per_card = []
    for i, sample in enumerate(val_data):
        per_card.append({
            "name": sample.name,
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
    for bucket in REPORTING_BUCKETS:
        mask = (actual >= bucket.low) & (actual < bucket.high)
        n = int(mask.sum())
        if n == 0:
            continue
        per_bucket.append(BucketMetrics(
            label=bucket.label,
            count=n,
            median_pct_error=round(float(np.median(pct_errors[mask])), 1),
            median_log_error=round(float(np.median(log_errors[mask])), 3),
            median_signed_log_error=round(float(np.median(signed_log_errors[mask])), 3),
        ))
    return per_bucket
