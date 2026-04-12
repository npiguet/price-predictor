"""Shared regression metrics for price predictor evaluation."""

from __future__ import annotations

import numpy as np

from price_predictor.domain.entities import EvaluationMetrics


def compute_regression_metrics(
    actual_eur: np.ndarray,
    predicted_eur: np.ndarray,
    log_offset: float = 2.0,
) -> EvaluationMetrics:
    """Compute the four standard regression metrics on EUR prices.

    - mean absolute error (EUR)
    - median percentage error
    - median absolute error in shifted-log space (`log(price + log_offset)`)
    - top-20% overlap between actual and predicted price rankings

    Both arrays are flattened so callers can pass either 1-D or column shapes.
    """
    actual = np.asarray(actual_eur, dtype=float).flatten()
    predicted = np.asarray(predicted_eur, dtype=float).flatten()

    abs_errors = np.abs(predicted - actual)
    mae = float(np.mean(abs_errors))

    safe_actual = np.maximum(actual, 0.01)
    pct_errors = np.abs(predicted - actual) / safe_actual * 100
    median_pct_error = float(np.median(pct_errors))

    log_errors = np.abs(
        np.log(actual + log_offset) - np.log(predicted + log_offset)
    )
    median_log_error = float(np.median(log_errors))

    n_top = max(1, int(len(actual) * 0.2))
    actual_top = set(np.argsort(actual)[-n_top:])
    predicted_top = set(np.argsort(predicted)[-n_top:])
    top_20_overlap = float(len(actual_top & predicted_top) / n_top)

    return EvaluationMetrics(
        mean_absolute_error_eur=round(mae, 2),
        median_percentage_error=round(median_pct_error, 1),
        median_abs_error_log=round(median_log_error, 3),
        top_20_overlap=round(top_20_overlap, 2),
        sample_count=len(actual),
    )
