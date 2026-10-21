"""The gates' arithmetic: thresholds and metrics over arrays already collected.

Pure numpy, so the evaluator, gate 1's measurement and the geometry checks share
one definition of each number without importing one another. Gate 2's own
thresholds live beside its table, in :mod:`effects.domain.damage_step_keywords`.

Every metric reports NaN where it has nothing to measure, rather than a number
that reads as a result; the gates fail a NaN margin as unmeasured.
"""

from __future__ import annotations

import random

import numpy as np

from effects.domain.effect_head_input import MAGNITUDE_CAP

# ── gate thresholds (FR-118, FR-123). Contract, not tuning. ──────────────
GATE1_MIN_GATE_F1_GAIN = 0.05
GATE1_MIN_ZONE_ACCURACY_GAIN = 0.05
GATE1_MIN_DEVIANCE_REDUCTION = 0.05
GATE3_MAX_MEAN_COSINE = 0.5
GATE3_COSINE_PAIRS = 10_000
GATE3_MAX_TOP_COMPONENT = 0.30


def f1_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Binary F1 over the affected/unaffected gate."""
    predicted = np.asarray(predicted).astype(bool)
    observed = np.asarray(observed).astype(bool)
    true_positive = float(np.sum(predicted & observed))
    if true_positive == 0:
        return 0.0
    precision = true_positive / float(np.sum(predicted))
    recall = true_positive / float(np.sum(observed))
    return 2 * precision * recall / (precision + recall)


def balanced_accuracy(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Accuracy averaged over the observed classes, each weighing the same (FR-109).

    Balanced because one class dominates a categorical field — ``stayed`` the
    zone outcome — and a plain accuracy reads as high for a model that answers
    that class every time.
    """
    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    if not observed.size:
        return float("nan")
    return float(np.mean([
        np.mean(predicted[observed == outcome] == outcome)
        for outcome in np.unique(observed)
    ]))


def poisson_deviance(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean Poisson deviance, the count analogue of squared error.

    Gate 1 reads it over the magnitude side of the count-valued fields, which is
    where "how much damage" and "how many cards" live — the numbers a model that
    is not reading the text can only predict at the corpus average.
    """
    predicted = np.asarray(predicted, dtype=np.float64)
    # Capped as the training loss caps it, so one runaway amount cannot
    # dominate the mean the model was never asked to fit.
    observed = np.minimum(np.asarray(observed, dtype=np.float64), MAGNITUDE_CAP)
    if predicted.size == 0:
        return float("nan")
    safe_predicted = np.clip(predicted, 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(observed > 0, observed * np.log(observed / safe_predicted), 0.0)
    return float(2.0 * np.mean(ratio - (observed - safe_predicted)))


def mean_pairwise_cosine(
    vectors: np.ndarray, *, pairs: int = GATE3_COSINE_PAIRS, seed: int = 42,
) -> float:
    """Mean cosine similarity over ``pairs`` random distinct pairs.

    Sampled rather than exhaustive because the corpus has ~38k unique texts and
    the full pairwise matrix is 700 million entries for a number a sample
    estimates to three decimals.
    """
    count = vectors.shape[0]
    if count < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-12, None)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(pairs):
        i = rng.randrange(count)
        j = rng.randrange(count - 1)
        if j >= i:
            j += 1
        total += float(np.dot(unit[i], unit[j]))
    return total / pairs


def top_component_share(vectors: np.ndarray) -> float:
    """Fraction of total variance the largest principal component explains."""
    if vectors.shape[0] < 2:
        return float("nan")
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    variance = singular ** 2
    total = float(variance.sum())
    # No variance at all is every vector on one point: the whole collapse.
    return float(variance[0] / total) if total > 0 else 1.0


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.dot(a, b)) / denominator
