"""Gate 1: the shipping model against the ``identity`` baseline (FR-118).

Scored on the card-disjoint split's unique-text stratum, the only slice where
a baseline holding a free vector per text cannot simply recall the answer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import Protocol

import numpy as np
import torch

from effects.application.ability_text import KeyResolver
from effects.application.evaluation_context import EvaluationContext
from effects.domain.ability_encoder import surface_of
from effects.domain.effect_head_layout import FIELDS_BY_NAME, GATE_INDEX
from effects.domain.effect_losses import magnitude_target
from effects.domain.effect_targets import derive_targets
from effects.domain.entity_field_targets import EntityFieldTargets
from effects.domain.evaluation_report import (
    Bound,
    CheckName,
    CheckResult,
    CheckStatus,
    Margin,
    gate_verdict,
)
from effects.domain.gate_metrics import (
    GATE1_MIN_DEVIANCE_REDUCTION,
    GATE1_MIN_GATE_F1_GAIN,
    GATE1_MIN_ZONE_ACCURACY_GAIN,
    balanced_accuracy,
    f1_score,
    poisson_deviance,
)
from effects.domain.head_decoding import (
    MAGNITUDE_TYPES,
    expected_magnitude,
    field_outputs,
)
from effects.domain.records import RecordKind, acting_keys
from effects.domain.surface_collation import entity_target_tensors, gather_entities
from effects.infrastructure.corpus_store import (
    CorpusStore,
    card_disjoint_resolutions,
    training_acting_keys,
)
from effects.infrastructure.model_runner import BATCH_RECORDS, Runnable

logger = logging.getLogger(__name__)

ZONE_OUTCOME = "zone_outcome"


@dataclass(frozen=True, slots=True)
class GateOneMetrics:
    """One model's scores on the card-disjoint split's unique-text stratum."""

    affected_gate_f1: float
    zone_outcome_accuracy: float
    mean_poisson_deviance: float


class TextsOf(Protocol):
    """Anything that names a key's text on the checkpoint's surface: a
    ``SurfaceBatcher`` or the ``KeyResolver`` it reads through."""

    def text_of(self, key) -> str | None: ...


def acting_text(record, texts: TextsOf) -> str | None:
    """The text of the line that acted, or None where nothing did."""
    return _text_of_keys(acting_keys(record), texts)


def _text_of_keys(keys, texts: TextsOf) -> str | None:
    for key in keys:
        text = texts.text_of(key)
        if text is not None:
            return text
    return None


def training_texts(acting, texts: TextsOf) -> set[str]:
    """Every acting-line text among ``acting``, the training records' distinct
    ``acting_keys`` tuples (the records themselves would not fit in memory)."""
    found = {_text_of_keys(keys, texts) for keys in acting}
    found.discard(None)
    return found


def unique_text_records(records, texts: TextsOf, *, seen) -> list:
    """The unique-text stratum: resolution records whose text is not in ``seen``.

    ``records`` is read once, so it may be a stream.
    """
    stratum = []
    for record in records:
        if record.kind is not RecordKind.RESOLUTION:
            continue
        text = acting_text(record, texts)
        if text is not None and text not in seen:
            stratum.append(record)
    return stratum


def unique_text_stratum(corpus_path, card_disjoint_games, texts: TextsOf) -> list:
    """Gate 1's population, read from the curated corpus the checkpoint trained on.

    Not from the raw shards: a raw game outside the recorded validation sets
    is not necessarily a training game (curation drops held-out games, and
    collection continues after a build).
    """
    seen = training_texts(
        training_acting_keys(CorpusStore(corpus_path).training_dir), texts,
    )
    logger.info("gate 1: %d acting texts in the training stratum", len(seen))
    return unique_text_records(
        card_disjoint_resolutions(corpus_path, frozenset(card_disjoint_games)),
        texts, seen=seen,
    )


def forward_chunks(records, runnable: Runnable) -> Iterator[tuple[list, tuple]]:
    """``(chunk, (gathered, gate, field_targets, mask))`` per chunk of ``records``,
    shared by the margins and :mod:`effects.application.record_scoring`."""
    for chunk in runnable.inference_chunks(records, BATCH_RECORDS):
        yield chunk, _forward(chunk, runnable)


def measure(records, runnable: Runnable) -> GateOneMetrics:
    """Run the model over ``records`` and score the three margins."""
    margins = MarginAccumulator(runnable.fields)
    for _chunk, forward in forward_chunks(records, runnable):
        margins.add(*forward)
    return margins.metrics()


def _forward(chunk, runnable: Runnable):
    """``(gathered, gate, field_targets, mask)`` for one chunk, on the host,
    ``gathered`` aligned column for column with the targets."""
    model = runnable.model
    batch, surfaces, _encoded = runnable.batcher.build(chunk, runnable.encoder)
    outputs = model.per_entity(model(**batch))
    targets = [derive_targets(record) for record in chunk]
    gate, field_targets, mask, index = entity_target_tensors(
        surfaces, targets, runnable.fields,
    )
    gathered = gather_entities(outputs, index.to(outputs.device)).float().cpu()
    return gathered, gate, field_targets, mask


class MarginAccumulator:
    """The three margins' predictions and observations, chunk by chunk.

    Pooled over the whole pass rather than averaged per chunk, so a chunk with
    one affected entity weighs what that one entity does.
    """

    def __init__(self, fields) -> None:
        self._magnitude_specs = [
            spec for spec in fields if spec.type in MAGNITUDE_TYPES
        ]
        self._gate_pred: list[np.ndarray] = []
        self._gate_true: list[np.ndarray] = []
        self._zone_pred: list[np.ndarray] = []
        self._zone_true: list[np.ndarray] = []
        self._count_pred: list[np.ndarray] = []
        self._count_true: list[np.ndarray] = []

    def add(
        self, gathered: torch.Tensor, gate: torch.Tensor,
        field_targets: EntityFieldTargets, mask: torch.Tensor,
    ) -> None:
        """Fold one chunk in: the gate over every real entity, zone and
        magnitude only where they train (target affected, field present)."""
        real = mask.bool()
        self._gate_pred.append(
            (gathered[..., GATE_INDEX][real] > 0).numpy().astype(np.int8)
        )
        self._gate_true.append(gate[real].numpy().astype(np.int8))

        affected = real & gate.bool()
        if ZONE_OUTCOME in field_targets:
            scored = affected & field_targets.present[ZONE_OUTCOME]
            logits = field_outputs(gathered, FIELDS_BY_NAME[ZONE_OUTCOME])[scored]
            if logits.numel():
                self._zone_pred.append(logits.argmax(-1).numpy())
                self._zone_true.append(field_targets.values[ZONE_OUTCOME][scored].numpy())
        for spec in self._magnitude_specs:
            target = field_targets.values.get(spec.name)
            if target is None:
                continue
            scored = affected & field_targets.present[spec.name]
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


def run_gate_one(context: EvaluationContext, identity) -> CheckResult:
    """Score both models on the split's unique-text stratum and compare."""
    games =context.main.provenance.card_disjoint_games
    if not games:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.SKIPPED,
            "the checkpoint records no card-disjoint games, so there is no "
            "held-out stratum to score gate 1 on",
        )
    # Both checkpoints read one vocabulary (require_same_split), so one
    # surface's texts decide the stratum for both: the batcher's own join.
    texts = KeyResolver(context.sidecars, surface_of(context.vocab_path))
    stratum = unique_text_stratum(context.corpus_path, games, texts)
    if not stratum:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.SKIPPED,
            "no resolution record in the card-disjoint split has an "
            "ability text absent from training; the identity baseline "
            "could recall every text in it, so the comparison would "
            "measure memorization on both sides rather than reading",
        )
    logger.info("gate 1: %d records in the unique-text stratum", len(stratum))

    scores = {}
    loaders = {
        "model": context.main_runnable,
        "identity": partial(context.runnable_of, identity),
    }
    for name, load in loaders.items():
        scores[name] = measure(stratum, load(stratum))
    return evaluate_gate_one(scores["model"], scores["identity"])


def _relative_reduction(baseline: float, value: float) -> float:
    """How far ``value`` falls below ``baseline``, as a share of it; NaN for
    an unmeasured baseline, 0 for a zero one."""
    if math.isnan(baseline):
        return baseline
    return (baseline - value) / baseline if baseline > 0 else 0.0


def evaluate_gate_one(
    model: GateOneMetrics, identity: GateOneMetrics,
) -> CheckResult:
    """All three margins must hold (FR-118).

    They fail differently: the gate F1 says whether the model knows *that*
    something happens, the zone accuracy *what*, the deviance *how much*. A
    model can beat the baseline on one and tie on the others by learning the
    corpus's priors.
    """
    return gate_verdict(
        CheckName.GATE_1,
        (
            Margin(
                "affected_gate_f1_gain", "affected-gate F1 gain",
                model.affected_gate_f1 - identity.affected_gate_f1,
                GATE1_MIN_GATE_F1_GAIN, Bound.FLOOR, value_format="+.3f",
            ),
            Margin(
                "zone_outcome_accuracy_gain", "zone-outcome accuracy gain",
                model.zone_outcome_accuracy - identity.zone_outcome_accuracy,
                GATE1_MIN_ZONE_ACCURACY_GAIN, Bound.FLOOR, value_format="+.3f",
            ),
            Margin(
                "poisson_deviance_reduction", "Poisson deviance reduction",
                _relative_reduction(
                    identity.mean_poisson_deviance, model.mean_poisson_deviance,
                ),
                GATE1_MIN_DEVIANCE_REDUCTION, Bound.FLOOR,
                value_format="+.1%", limit_format=".0%",
            ),
        ),
        fail_prefix="the encoder is not reading text",
        pass_detail="beats the identity baseline on all three margins",
    )
