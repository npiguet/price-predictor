"""Gate 1: the identity baseline, measured by running both checkpoints over the split.

Gate 1 asks whether the encoder is reading text at all, and it answers by
comparing the shipping model against an ``identity`` baseline that can memorize
which ability a line is but cannot read a word of it. This module produces each
side's numbers, compares them and renders the verdict (FR-118).

Two choices here are the whole point of the gate.

The stratum is the **unique-text** slice of the card-disjoint split: resolution
records whose acting line's text appears on no training card. The identity
baseline holds a free vector per text, so on a text it saw in training it can
simply recall what happened, and a comparison over the whole split would mostly
measure memorization on both sides. On a text absent from training the baseline
has nothing to recall, and only reading the words helps.

The three margins are measured together on one pass rather than three, because
they must describe the same records: the gate F1 says whether the model knows
*that* something happens, the zone accuracy *what*, and the deviance *how much*.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from effects.application.breakdowns import TOTAL, RecordResult
from effects.application.surface_batching import SurfaceBatcher
from effects.domain.damage_step_keywords import KeywordResolver, keyword_of_line
from effects.domain.effect_head_layout import FIELDS_BY_NAME, GATE_INDEX, EntityFieldTargets
from effects.domain.effect_losses import per_entity_loss
from effects.domain.effect_targets import derive_targets
from effects.domain.evaluation_report import (
    CheckName,
    CheckResult,
    CheckStatus,
    breaches,
    margin,
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
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import RecordKind, acting_keys
from effects.domain.rule_families import rule_family
from effects.domain.surface_collation import entity_target_tensors, gather_entities

if TYPE_CHECKING:
    from effects.application.evaluation_context import EvaluationContext

logger = logging.getLogger(__name__)

BATCH_RECORDS = 32

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
    """Every acting-line text the training records contain.

    ``acting`` is the training records' distinct ``acting_keys`` tuples rather
    than the records: the distinct acting lines are what gate 1 needs and what
    fits in memory.
    """
    found = {_text_of_keys(keys, texts) for keys in acting}
    found.discard(None)
    return found


def unique_text_records(records, texts: TextsOf, *, seen) -> list:
    """The unique-text stratum: resolution records whose text is not in ``seen``.

    Resolution records only, because the gate's three margins are about what an
    ability does when it resolves. A combat record has no acting line at all,
    so it could not be stratified by text even in principle. ``records`` is
    read once, so it may be a stream.
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

    "Seen" is every text a record of the corpus's ``training/`` acts with —
    what the identity baseline could have memorized — and the candidates are
    the card-disjoint stratum's resolutions, streamed so only the stratum is
    held. Neither comes from the raw shards: a raw game outside the recorded
    validation sets is not necessarily a training game (curation drops
    held-out games, and collection continues after a build).
    """
    from effects.infrastructure.corpus_store import CorpusStore
    from effects.infrastructure.model_runner import (
        card_disjoint_resolutions,
        training_acting_keys,
    )

    seen = training_texts(
        training_acting_keys(CorpusStore(corpus_path).training_dir), texts,
    )
    logger.info("gate 1: %d acting texts in the training stratum", len(seen))
    return unique_text_records(
        card_disjoint_resolutions(corpus_path, frozenset(card_disjoint_games)),
        texts, seen=seen,
    )


# ── scoring: one forward pass, two readers ──────────────────────────────


def _forward_chunks(
    records, encoder, model, batcher: SurfaceBatcher, fields,
) -> Iterator[tuple[list, tuple]]:
    """``(chunk, (gathered, gate, field_targets, mask))`` per chunk of ``records``.

    The one forward pass both readers share: the margins pool it, the
    per-record scorer splits it by row.
    """
    encoder.eval()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_RECORDS):
            chunk = records[start : start + BATCH_RECORDS]
            yield chunk, _forward(chunk, encoder, model, batcher, fields)


def measure(records, encoder, model, batcher: SurfaceBatcher, *, fields) -> GateOneMetrics:
    """Run the model over ``records`` and score the three margins.

    ``fields`` is the active-field tuple the run trains with, so the gate reads
    the same heads the loss shaped.
    """
    margins = MarginAccumulator(fields)
    for _chunk, forward in _forward_chunks(records, encoder, model, batcher, fields):
        margins.add(*forward)
    return margins.metrics()


def score_records(
    records, encoder, model, batcher: SurfaceBatcher, *, fields,
    legality_mode_counts=None,
) -> list[RecordResult]:
    """Every record's own losses and grouping keys, for the breakdowns.

    ``legality_mode_counts`` is the dataset manifest's, so each record's family
    is the one the build placed it in.
    """
    # One resolver for the whole pass: it memoizes each provenance key's
    # keyword, and the strata repeat the same few hundred cards.
    resolver = KeywordResolver(batcher.sidecars)
    results: list[RecordResult] = []
    for chunk, (gathered, gate, field_targets, mask) in _forward_chunks(
        records, encoder, model, batcher, fields,
    ):
        for row, record in enumerate(chunk):
            results.append(_record_result(
                record, batcher, resolver,
                _record_losses(row, gathered, gate, field_targets, mask, fields),
                legality_mode_counts,
            ))
    return results


def _forward(chunk, encoder, model, batcher: SurfaceBatcher, fields):
    """``(gathered, gate, field_targets, mask)`` for one chunk, on the host.

    ``gathered`` is the per-entity head's output at each target entity's slot,
    aligned column for column with the targets ``entity_target_tensors`` built.
    """
    batch, surfaces, _encoded = batcher.build(chunk, encoder)
    outputs = model.per_entity(model(**batch))
    targets = [derive_targets(record) for record in chunk]
    gate, field_targets, mask, index = entity_target_tensors(
        surfaces, targets, fields,
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
        """Fold one chunk's gathered outputs and targets in.

        The gate is read over every real entity; zone and magnitude only over
        the entities the gate's *target* marks affected and whose record
        supervises the field, the ones those fields are trained on.
        """
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
            self._count_true.append(np.abs(target[scored].numpy()))

    def metrics(self) -> GateOneMetrics:
        """The three margins over everything added; NaN where nothing was."""
        return GateOneMetrics(
            affected_gate_f1=_f1(self._gate_pred, self._gate_true),
            zone_outcome_accuracy=balanced_accuracy(
                _cat(self._zone_pred), _cat(self._zone_true),
            ),
            mean_poisson_deviance=_deviance(self._count_pred, self._count_true),
        )


def _record_losses(
    row: int, gathered: torch.Tensor, gate: torch.Tensor,
    field_targets: EntityFieldTargets, mask: torch.Tensor, fields,
) -> dict[str, float]:
    """One record's per-entity loss by field: ``per_entity_loss`` on it alone.

    The trainer's own loss over a batch of one, so the breakdowns cannot drift
    from what training minimized: the gate over every real entity, then each
    field over the affected entities its record supervises. A field with no
    such entity is left out rather than reported as zero. One record means the
    per-record normalization divides by one, so each part is that record's raw
    sum. Read on the host, where the batch already
    is, so scoring a record costs no device synchronization.
    """
    rows = slice(row, row + 1)
    _total, losses = per_entity_loss(
        gathered[rows], gate[rows],
        field_targets.select(rows),
        mask[rows], fields=fields, report_parts=True,
    )
    losses[TOTAL] = sum(losses.values())
    return losses


def _record_result(
    record, batcher: SurfaceBatcher, resolver: KeywordResolver,
    losses: dict[str, float], legality_mode_counts=None,
) -> RecordResult:
    """The grouping keys of one scored record, beside its losses."""
    try:
        family = rule_family(
            record, batcher.sidecars, resolver=resolver,
            legality_mode_counts=legality_mode_counts,
        )
    except KeyError:
        # The sidecar does not describe the card the record names; the record
        # still scored, it just cannot be placed in a family.
        family = None
    return RecordResult(
        record_id=record.record_id,
        game_id=record.game_id,
        kind=record.kind.value,
        subkind=record.subkind.value if record.subkind else None,
        text=acting_text(record, batcher),
        family=family,
        random_seat=bool(record.random_seat),
        what_if=record.what_if,
        keywords=record_keywords(record, batcher, resolver),
        losses=losses,
    )


def record_keywords(
    record, batcher: SurfaceBatcher, resolver: KeywordResolver,
) -> frozenset[str]:
    """Every keyword on the acting line or carried by a board entity.

    The acting line is a keyword when the record is that keyword's own
    resolution (a ward trigger, say); the board's entities carry theirs through
    both channels the resolver reads.
    """
    found: set[str] = set()
    for key in acting_keys(record):
        line = batcher.sidecars.line_or_none(key)
        keyword = keyword_of_line(line) if line is not None else None
        if keyword is not None:
            found.add(normalize_keyword(keyword))
    for entity in record.state.entities:
        found.update(normalize_keyword(k) for k in resolver.keywords_of(entity))
    return frozenset(found)


def _cat(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty(0)


def _f1(pred: list[np.ndarray], true: list[np.ndarray]) -> float:
    predicted, observed = _cat(pred), _cat(true)
    return f1_score(predicted, observed) if predicted.size else float("nan")


def _deviance(pred: list[np.ndarray], true: list[np.ndarray]) -> float:
    predicted, observed = _cat(pred), _cat(true)
    return poisson_deviance(predicted, observed) if predicted.size else float("nan")


# ── the gate ────────────────────────────────────────────────────────────


def run_gate_one(context: EvaluationContext, identity) -> CheckResult:
    """Score both models on the split's unique-text stratum and compare.

    Loading a second model and running the corpus twice is the expensive part
    of an evaluation, so it happens only when the baseline is actually
    supplied. The baseline is loaded for this gate alone and released after it.
    """
    from effects.application.ability_text import KeyResolver
    from effects.domain.ability_encoder import surface_of

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
        runnable = load(stratum)
        scores[name] = measure(
            stratum, runnable.encoder, runnable.model, runnable.batcher,
            fields=runnable.fields,
        )
    return evaluate_gate_one(scores["model"], scores["identity"])


def _relative_reduction(baseline: float, value: float) -> float:
    """How far ``value`` falls below ``baseline``, as a share of it.

    An unmeasured baseline leaves the reduction unmeasured; a zero one leaves
    nothing to reduce.
    """
    if math.isnan(baseline):
        return baseline
    return (baseline - value) / baseline if baseline > 0 else 0.0


def evaluate_gate_one(
    model: GateOneMetrics, identity: GateOneMetrics,
) -> CheckResult:
    """All three margins must hold (FR-118). Blocks shipping.

    Three margins rather than one because they fail differently: the gate F1
    says whether the model knows *that* something happens, the zone accuracy
    whether it knows *what*, and the deviance whether it knows *how much*. A
    model can beat the baseline on one and tie on the others by learning the
    corpus's priors.
    """
    gate_gain = model.affected_gate_f1 - identity.affected_gate_f1
    zone_gain = model.zone_outcome_accuracy - identity.zone_outcome_accuracy
    deviance_reduction = _relative_reduction(
        identity.mean_poisson_deviance, model.mean_poisson_deviance,
    )
    values = {
        "affected_gate_f1_gain": gate_gain,
        "zone_outcome_accuracy_gain": zone_gain,
        "poisson_deviance_reduction": deviance_reduction,
    }
    failures = breaches(
        margin(
            "affected-gate F1 gain", gate_gain,
            gate_gain < GATE1_MIN_GATE_F1_GAIN,
            f"{gate_gain:+.3f} < {GATE1_MIN_GATE_F1_GAIN}",
        ),
        margin(
            "zone-outcome accuracy gain", zone_gain,
            zone_gain < GATE1_MIN_ZONE_ACCURACY_GAIN,
            f"{zone_gain:+.3f} < {GATE1_MIN_ZONE_ACCURACY_GAIN}",
        ),
        margin(
            "Poisson deviance reduction", deviance_reduction,
            deviance_reduction < GATE1_MIN_DEVIANCE_REDUCTION,
            f"{deviance_reduction:+.1%} < {GATE1_MIN_DEVIANCE_REDUCTION:.0%}",
        ),
    )
    if failures:
        return CheckResult(
            CheckName.GATE_1, CheckStatus.FAIL,
            "the encoder is not reading text: " + "; ".join(failures),
            values,
        )
    return CheckResult(
        CheckName.GATE_1, CheckStatus.PASS,
        "beats the identity baseline on all three margins", values,
    )
