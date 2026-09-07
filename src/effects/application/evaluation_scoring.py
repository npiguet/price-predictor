"""Scoring a checkpoint over the recorded split, per kind and per stratum.

Separated from ``evaluate_effect_model`` for the same reason ``training_loop``
is separated from ``train_effect_model``: the gate arithmetic and the strata
definitions are pure functions of their inputs and testable with no torch, while
running a model over a corpus is not.

Two rules the reported numbers depend on:

- **Only the recorded games are scored.** The corpus is append-only and grows
  between the training run and the evaluation, so a record from a game the model
  never saw is neither training nor validation — it is unclassified, and scoring
  it would quietly change what the split means.
- **Metrics condition on affected entities and are class-balanced.** Most
  entities on most boards are unaffected, so an unconditioned accuracy is
  dominated by the easy negatives and would read as high for a model that
  predicts "nothing happens" everywhere.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from effects.application.evaluate_effect_model import (
    GateOneMetrics,
    Stratum,
    f1_score,
    poisson_deviance,
)
from effects.domain.effect_model import (
    ZONE_OUTCOMES,
    FieldType,
    per_entity_fields_of_type,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)


@dataclass
class Predictions:
    """One model's raw outputs over a scored population.

    Kept as flat arrays rather than per-record structures because every metric
    below is a reduction over entities, and the reductions are what the gates
    read.
    """

    gate_predicted: list[float] = field(default_factory=list)
    gate_observed: list[float] = field(default_factory=list)
    zone_predicted: list[int] = field(default_factory=list)
    zone_observed: list[int] = field(default_factory=list)
    count_predicted: list[float] = field(default_factory=list)
    count_observed: list[float] = field(default_factory=list)

    def gate_f1(self) -> float:
        if not self.gate_observed:
            return float("nan")
        return f1_score(
            np.array(self.gate_predicted) >= 0.5, np.array(self.gate_observed),
        )

    def zone_accuracy(self) -> float:
        """Accuracy over affected entities only, class-balanced.

        Balanced because "stayed" dominates the corpus: an unbalanced accuracy
        would read as high for a model that answers "stayed" every time.
        """
        if not self.zone_observed:
            return float("nan")
        predicted = np.array(self.zone_predicted)
        observed = np.array(self.zone_observed)
        per_class = []
        for outcome in range(len(ZONE_OUTCOMES)):
            mask = observed == outcome
            if mask.any():
                per_class.append(float((predicted[mask] == outcome).mean()))
        return float(np.mean(per_class)) if per_class else float("nan")

    def mean_deviance(self) -> float:
        if not self.count_observed:
            return float("nan")
        return poisson_deviance(
            np.array(self.count_predicted), np.array(self.count_observed),
        )

    def as_gate_one_metrics(self) -> GateOneMetrics:
        return GateOneMetrics(
            affected_gate_f1=self.gate_f1(),
            zone_outcome_accuracy=self.zone_accuracy(),
            mean_poisson_deviance=self.mean_deviance(),
        )

    def __len__(self) -> int:
        return len(self.gate_observed)


def stratify(
    record: EffectRecord,
    *,
    training_texts: set[str],
    training_api_types: set[str],
    numeric_range: tuple[float, float],
    text_of,
    api_types_of,
    numbers_of,
) -> Stratum:
    """Which held-out stratum a record belongs to (FR-106).

    The four separate "has the model seen this exact text" from "has it seen
    these parts": a model that only memorized texts scores well on
    ``shared-text`` and badly on the other three, which is exactly the failure
    gate 1 exists to catch.
    """
    text = text_of(record)
    if text is not None and text in training_texts:
        return Stratum.SHARED_TEXT

    low, high = numeric_range
    for value in numbers_of(record):
        if value < low or value > high:
            return Stratum.NUMERIC_EXTRAPOLATION

    api_types = set(api_types_of(record))
    if api_types and api_types <= training_api_types:
        return Stratum.NOVEL_COMBINATION

    return Stratum.UNIQUE_TEXT


@dataclass
class StratifiedReport:
    """Per-kind and per-stratum predictions for one model."""

    by_kind: dict[str, Predictions] = field(
        default_factory=lambda: defaultdict(Predictions),
    )
    by_stratum: dict[Stratum, Predictions] = field(
        default_factory=lambda: defaultdict(Predictions),
    )
    overall: Predictions = field(default_factory=Predictions)

    def observe(
        self,
        kind: str,
        stratum: Stratum,
        *,
        gate_predicted: float,
        gate_observed: float,
        zone_predicted: int | None = None,
        zone_observed: int | None = None,
        count_predicted: float | None = None,
        count_observed: float | None = None,
    ) -> None:
        for target in (self.by_kind[kind], self.by_stratum[stratum], self.overall):
            target.gate_predicted.append(gate_predicted)
            target.gate_observed.append(gate_observed)
            # Conditional fields are recorded only where the entity was
            # actually affected: predicting a zone outcome for an untouched
            # permanent is not a question the record asked.
            if gate_observed and zone_observed is not None:
                target.zone_predicted.append(int(zone_predicted or 0))
                target.zone_observed.append(int(zone_observed))
            if gate_observed and count_observed is not None:
                target.count_predicted.append(float(count_predicted or 0.0))
                target.count_observed.append(float(count_observed))

    def kinds(self) -> list[str]:
        return sorted(self.by_kind)

    def render(self, label: str, floor: StratifiedReport | None = None) -> str:
        """A table per kind, with the ``state-only`` floor beside each row.

        Every kind reports that floor because a number is only interpretable
        against what a model with no ability information achieves — some kinds
        are nearly determined by the board alone.
        """
        lines = [f"{label}:"]
        for kind in self.kinds():
            predictions = self.by_kind[kind]
            floor_text = ""
            if floor is not None and kind in floor.by_kind:
                floor_text = f"  (state-only floor {floor.by_kind[kind].gate_f1():.3f})"
            lines.append(
                f"  {kind:<24} n={len(predictions):<7} "
                f"gate F1 {predictions.gate_f1():.3f}"
                f"  zone {predictions.zone_accuracy():.3f}"
                f"  deviance {predictions.mean_deviance():.3f}{floor_text}"
            )
        for stratum in Stratum:
            predictions = self.by_stratum.get(stratum)
            if predictions and len(predictions):
                lines.append(
                    f"  [{stratum.value}] n={len(predictions)} "
                    f"gate F1 {predictions.gate_f1():.3f}"
                )
        return "\n".join(lines)


def count_valued_fields() -> tuple[str, ...]:
    """The fields gate 1's deviance is computed over (FR-118, FR-080).

    Counts and the magnitude half of every signed delta — the numbers a model
    that is not reading the text can only predict at the corpus average.
    """
    return tuple(
        spec.name for spec in per_entity_fields_of_type(
            FieldType.COUNT, FieldType.SIGNED_DELTA,
        )
    )


def observed_gate(record: EffectRecord, entity_id: str) -> float:
    """Whether this record's events name ``entity_id``."""
    targets = derive_targets(record)
    entry = targets.get(entity_id)
    return 1.0 if entry is not None and entry.affected else 0.0
