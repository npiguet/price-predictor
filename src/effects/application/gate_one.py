"""Gate 1: the shipping model against the ``identity`` baseline (FR-118).

Scored on the card-disjoint split's unique-text stratum, the only slice where
a baseline holding a free vector per text cannot simply recall the answer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterator
from functools import partial

import torch

from effects.application.ability_text import KeyResolver
from effects.application.evaluation_context import EvaluationContext
from effects.domain.ability_encoder import surface_of
from effects.domain.entity_field_targets import EntityTargetBatch
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
    GateOneMetrics,
    MarginAccumulator,
)
from effects.domain.provenance import ProvenanceKey, first_text
from effects.domain.records import RecordKind, acting_keys
from effects.domain.surface_collation import per_entity_targets
from effects.infrastructure.corpus_store import (
    CorpusStore,
    card_disjoint_resolutions,
    training_acting_keys,
)
from effects.infrastructure.model_runner import BATCH_RECORDS, Runnable

logger = logging.getLogger(__name__)


#: A key's text on the checkpoint's surface: ``SurfaceBatcher.text_of`` or
#: the ``KeyResolver.text_of`` it reads through.
TextOf = Callable[[ProvenanceKey], str | None]


def training_texts(acting, text_of: TextOf) -> set[str]:
    """Every acting-line text among ``acting``, the training records' distinct
    ``acting_keys`` tuples (the records themselves would not fit in memory)."""
    found = {first_text(keys, text_of) for keys in acting}
    found.discard(None)
    return found


def unique_text_records(records, text_of: TextOf, *, seen) -> list:
    """The unique-text stratum: resolution records whose text is not in ``seen``.

    ``records`` is read once, so it may be a stream.
    """
    stratum = []
    for record in records:
        if record.kind is not RecordKind.RESOLUTION:
            continue
        text = first_text(acting_keys(record), text_of)
        if text is not None and text not in seen:
            stratum.append(record)
    return stratum


def unique_text_stratum(corpus_path, card_disjoint_games, text_of: TextOf) -> list:
    """Gate 1's population, read from the curated corpus the checkpoint trained on.

    Not from the raw shards: a raw game outside the recorded validation sets
    is not necessarily a training game (curation drops held-out games, and
    collection continues after a build).
    """
    seen = training_texts(
        training_acting_keys(CorpusStore(corpus_path).training_dir), text_of,
    )
    logger.info("gate 1: %d acting texts in the training stratum", len(seen))
    return unique_text_records(
        card_disjoint_resolutions(corpus_path, frozenset(card_disjoint_games)),
        text_of, seen=seen,
    )


def forward_chunks(
    records, runnable: Runnable,
) -> Iterator[tuple[list, torch.Tensor, EntityTargetBatch]]:
    """``(chunk, gathered, targets)`` per chunk of ``records``, shared by the
    margins and :mod:`effects.application.record_scoring`."""
    for chunk in runnable.inference_chunks(records, BATCH_RECORDS):
        yield chunk, *_forward(chunk, runnable)


def measure(records, runnable: Runnable) -> GateOneMetrics:
    """Run the model over ``records`` and score the three margins."""
    margins = MarginAccumulator(runnable.fields)
    for _chunk, gathered, targets in forward_chunks(records, runnable):
        margins.add(gathered, targets)
    return margins.metrics()


def _forward(chunk, runnable: Runnable) -> tuple[torch.Tensor, EntityTargetBatch]:
    """``(gathered, targets)`` for one chunk, on the host, ``gathered``
    aligned column for column with the targets."""
    gathered, surfaces = runnable.per_entity_outputs(chunk)
    return gathered, per_entity_targets(chunk, surfaces, runnable.fields)


def run_gate_one(context: EvaluationContext, identity) -> CheckResult:
    """Score both models on the split's unique-text stratum and compare."""
    games = context.main.provenance.card_disjoint_games
    if not games:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.SKIPPED,
            "the checkpoint records no card-disjoint games, so there is no "
            "held-out stratum to score gate 1 on",
        )
    # Both checkpoints read one vocabulary (require_same_split), so one
    # surface's texts decide the stratum for both: the batcher's own join.
    texts = KeyResolver(context.sidecars, surface_of(context.vocab_path))
    stratum = unique_text_stratum(context.corpus_path, games, texts.text_of)
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
