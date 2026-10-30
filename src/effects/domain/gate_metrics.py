"""The gates' arithmetic: thresholds, metrics over arrays already collected,
and gate 1's margins pooled from the per-entity head's outputs.

Gate 2's thresholds live in :mod:`effects.domain.damage_step_keywords`. Every
metric reports NaN where it has nothing to measure; the gates fail a NaN
margin as unmeasured. The margins live here rather than in the gate-1
command, so the training loop's progress reading imports no corpus or
checkpoint loader.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch

from effects.domain.effect_head_input import MAGNITUDE_CAP
from effects.domain.effect_head_layout import FIELDS_BY_NAME, GATE_INDEX, HeadField
from effects.domain.effect_losses import magnitude_target
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.head_decoding import (
    expected_magnitude,
    field_outputs,
)

# Gate thresholds (FR-118, FR-123): contract, not tuning.
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
    if not observed.size:
        return float("nan")
    true_positive = float(np.sum(predicted & observed))
    if true_positive == 0:
        return 0.0
    precision = true_positive / float(np.sum(predicted))
    recall = true_positive / float(np.sum(observed))
    return 2 * precision * recall / (precision + recall)


def balanced_accuracy(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Accuracy averaged over the observed classes, each weighing the same
    (FR-109), so always answering the dominant ``stayed`` scores as chance."""
    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    if not observed.size:
        return float("nan")
    return float(np.mean([
        np.mean(predicted[observed == outcome] == outcome)
        for outcome in np.unique(observed)
    ]))


def poisson_deviance(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean Poisson deviance, the count analogue of squared error."""
    predicted = np.asarray(predicted, dtype=np.float64)
    # Capped as the training loss caps it.
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
    """Mean cosine similarity over ``pairs`` random distinct pairs."""
    count = vectors.shape[0]
    if count < 2:
        return float("nan")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-12, None)
    # Drawn one at a time so a seed keeps picking the same pairs.
    rng = random.Random(seed)
    first = np.empty(pairs, dtype=np.intp)
    second = np.empty(pairs, dtype=np.intp)
    for k in range(pairs):
        i = rng.randrange(count)
        j = rng.randrange(count - 1)
        first[k], second[k] = i, (j + 1 if j >= i else j)
    dots = np.einsum("ij,ij->i", unit[first], unit[second])
    return float(dots.astype(np.float64).sum()) / pairs


def top_component_share(vectors: np.ndarray) -> float:
    """Fraction of total variance the largest principal component explains."""
    if vectors.shape[0] < 2:
        return float("nan")
    centred = (vectors - vectors.mean(axis=0, keepdims=True)).astype(np.float64)
    # The small scatter matrix's eigenvalues are the squared singular values,
    # far cheaper than an SVD; clamped since rounding can go a hair negative.
    variance = np.clip(np.linalg.eigvalsh(centred.T @ centred), 0.0, None)
    total = float(variance.sum())
    # No variance at all is every vector on one point: the whole collapse.
    return float(variance[-1] / total) if total > 0 else 1.0


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return float("nan")
    return 1.0 - float(np.dot(a, b)) / denominator


@dataclass(frozen=True, slots=True)
class GateOneMetrics:
    """One model's scores on the card-disjoint split's unique-text stratum."""

    affected_gate_f1: float
    zone_outcome_accuracy: float
    mean_poisson_deviance: float


class MarginAccumulator:
    """The three margins' predictions and observations, chunk by chunk.

    Pooled over the whole pass rather than averaged per chunk, so a chunk with
    one affected entity weighs what that one entity does.
    """

    def __init__(self, fields) -> None:
        self._magnitude_specs = [
            spec for spec in fields if spec.type.has_magnitude
        ]
        self._gate_pred: list[np.ndarray] = []
        self._gate_true: list[np.ndarray] = []
        self._zone_pred: list[np.ndarray] = []
        self._zone_true: list[np.ndarray] = []
        self._count_pred: list[np.ndarray] = []
        self._count_true: list[np.ndarray] = []

    def add(self, gathered: torch.Tensor, targets: EntityTargetBatch) -> None:
        """Fold one chunk in: the gate over every real entity, zone and
        magnitude only where they train (target affected, field present)."""
        field_targets = targets.fields
        real = targets.real
        self._gate_pred.append(
            (gathered[..., GATE_INDEX][real] > 0).numpy().astype(np.int8)
        )
        self._gate_true.append(targets.gate[real].numpy().astype(np.int8))

        zone = HeadField.ZONE_OUTCOME
        if zone in field_targets:
            scored = targets.scored(zone)
            logits = field_outputs(gathered, FIELDS_BY_NAME[zone])[scored]
            if logits.numel():
                self._zone_pred.append(logits.argmax(-1).numpy())
                self._zone_true.append(field_targets.values[zone][scored].numpy())
        for spec in self._magnitude_specs:
            target = field_targets.values.get(spec.name)
            if target is None:
                continue
            scored = targets.scored(spec.name)
            slice_ = field_outputs(gathered, spec)[scored]
            if not slice_.numel():
                continue
            self._count_pred.append(expected_magnitude(spec, slice_).numpy())
            self._count_true.append(magnitude_target(spec, target[scored]).numpy())

    def metrics(self) -> GateOneMetrics:
        """The three margins over everything added; NaN where nothing was."""
        return GateOneMetrics(
            affected_gate_f1=f1_score(_cat(self._gate_pred), _cat(self._gate_true)),
            zone_outcome_accuracy=balanced_accuracy(
                _cat(self._zone_pred), _cat(self._zone_true),
            ),
            mean_poisson_deviance=poisson_deviance(
                _cat(self._count_pred), _cat(self._count_true),
            ),
        )


def _cat(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0)
