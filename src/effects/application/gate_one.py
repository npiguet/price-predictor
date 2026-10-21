"""Gate 1's three numbers, measured by running a checkpoint over the split.

Gate 1 asks whether the encoder is reading text at all, and it answers by
comparing the shipping model against an ``identity`` baseline that can memorize
which ability a line is but cannot read a word of it. That comparison lives in
``evaluate_effect_model``; producing each side's numbers is this module's job.

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
from dataclasses import dataclass

import numpy as np
import torch

from effects.application.breakdowns import TOTAL, RecordResult
from effects.application.surface_batching import SurfaceBatcher
from effects.domain.damage_step_keywords import KeywordResolver, keyword_of_line
from effects.domain.effect_model import (
    FIELDS_BY_NAME,
    GATE_INDEX,
    entity_target_tensors,
    gather_entities,
    per_entity_loss,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.gate_metrics import balanced_accuracy, f1_score, poisson_deviance
from effects.domain.head_decoding import (
    MAGNITUDE_TYPES,
    expected_magnitude,
    field_outputs,
)
from effects.domain.keyword_names import normalize_keyword
from effects.domain.records import RecordKind, acting_keys
from effects.domain.rule_families import rule_family

logger = logging.getLogger(__name__)

BATCH_RECORDS = 32

ZONE_OUTCOME = "zone_outcome"


@dataclass(frozen=True, slots=True)
class GateOneMetrics:
    """One model's scores on the card-disjoint split's unique-text stratum.

    ``records`` holds every scored record's own losses beside the pooled
    margins, for the evaluator's per-text, per-family and per-policy
    breakdowns (``application.breakdowns``). The margins never read it.
    """

    affected_gate_f1: float
    zone_outcome_accuracy: float
    mean_poisson_deviance: float
    records: tuple[RecordResult, ...] = ()


def acting_text(record, batcher: SurfaceBatcher) -> str | None:
    """The text of the line that acted, or None where nothing did."""
    return _text_of_keys(acting_keys(record), batcher)


def _text_of_keys(keys, batcher: SurfaceBatcher) -> str | None:
    for key in keys:
        text = batcher.text_of(key)
        if text is not None:
            return text
    return None


def training_texts(acting, batcher: SurfaceBatcher) -> set[str]:
    """Every acting-line text the training games contain.

    ``acting`` is the training records' distinct ``acting_keys`` tuples rather
    than the records: the training stratum is the whole raw corpus, and its
    distinct acting lines are what fits in memory.
    """
    texts = {_text_of_keys(keys, batcher) for keys in acting}
    texts.discard(None)
    return texts


def unique_text_records(records, batcher: SurfaceBatcher, *, seen) -> list:
    """The unique-text stratum: resolution records whose text is not in ``seen``.

    Resolution records only, because the gate's three margins are about what an
    ability does when it resolves. A combat record has no acting line at all,
    so it could not be stratified by text even in principle.
    """
    stratum = []
    for record in records:
        if record.kind is not RecordKind.RESOLUTION:
            continue
        text = acting_text(record, batcher)
        if text is not None and text not in seen:
            stratum.append(record)
    return stratum


def measure(
    records, encoder, model, batcher: SurfaceBatcher, *, fields,
    legality_mode_counts=None,
) -> GateOneMetrics:
    """Run the model over ``records`` and score the three margins.

    ``fields`` is the active-field tuple the run trains with, so the gate reads
    the same heads the loss shaped. ``legality_mode_counts`` is the dataset
    manifest's, so each record's family is the one the build placed it in.
    """
    margins = MarginAccumulator(fields)
    results: list[RecordResult] = []
    # One resolver for the whole pass: it memoizes each provenance key's
    # keyword, and the stratum repeats the same few hundred cards.
    resolver = KeywordResolver(batcher.sidecars)

    encoder.eval()
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), BATCH_RECORDS):
            chunk = records[start : start + BATCH_RECORDS]
            gathered, gate, field_targets, mask = _forward(
                chunk, encoder, model, batcher, fields,
            )
            for row, record in enumerate(chunk):
                results.append(_record_result(
                    record, batcher, resolver,
                    _record_losses(row, gathered, gate, field_targets, mask, fields),
                    legality_mode_counts,
                ))
            margins.add(gathered, gate, field_targets, mask)
    return margins.metrics(records=tuple(results))


def _forward(chunk, encoder, model, batcher: SurfaceBatcher, fields):
    """``(gathered, gate, field_targets, mask)`` for one chunk, on the host.

    ``gathered`` is the per-entity head's output at each target entity's slot,
    aligned column for column with the targets ``entity_target_tensors`` built.
    """
    batch, surfaces = batcher.build(chunk, encoder)
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
        field_targets: dict[str, torch.Tensor], mask: torch.Tensor,
    ) -> None:
        """Fold one chunk's gathered outputs and targets in.

        The gate is read over every real entity; zone and magnitude only over
        the entities the gate's *target* marks affected, the ones those fields
        are supervised on.
        """
        real = mask.bool()
        self._gate_pred.append(
            (gathered[..., GATE_INDEX][real] > 0).numpy().astype(np.int8)
        )
        self._gate_true.append(gate[real].numpy().astype(np.int8))

        affected = real & gate.bool()
        if ZONE_OUTCOME in field_targets:
            logits = field_outputs(gathered, FIELDS_BY_NAME[ZONE_OUTCOME])[affected]
            if logits.numel():
                self._zone_pred.append(logits.argmax(-1).numpy())
                self._zone_true.append(field_targets[ZONE_OUTCOME][affected].numpy())
        for spec in self._magnitude_specs:
            target = field_targets.get(spec.name)
            if target is None:
                continue
            slice_ = field_outputs(gathered, spec)[affected]
            if not slice_.numel():
                continue
            self._count_pred.append(expected_magnitude(spec, slice_).numpy())
            self._count_true.append(np.abs(target[affected].numpy()))

    def metrics(self, *, records: tuple[RecordResult, ...] = ()) -> GateOneMetrics:
        """The three margins over everything added; NaN where nothing was."""
        return GateOneMetrics(
            affected_gate_f1=_f1(self._gate_pred, self._gate_true),
            zone_outcome_accuracy=balanced_accuracy(
                _cat(self._zone_pred), _cat(self._zone_true),
            ),
            mean_poisson_deviance=_deviance(self._count_pred, self._count_true),
            records=records,
        )


def _record_losses(
    row: int, gathered: torch.Tensor, gate: torch.Tensor,
    field_targets: dict[str, torch.Tensor], mask: torch.Tensor, fields,
) -> dict[str, float]:
    """One record's per-entity loss by field: ``per_entity_loss`` on it alone.

    The trainer's own loss over a batch of one, so the breakdowns cannot drift
    from what training minimized: the gate over every real entity, then each
    field over the entities the gate's *target* marks affected. A field with no
    affected entity is not supervised and is left out rather than reported as
    zero. One record means the per-record normalization divides by one, so each
    part is that record's raw sum. Read on the host, where the batch already
    is, so scoring a record costs no device synchronization.
    """
    rows = slice(row, row + 1)
    _total, losses = per_entity_loss(
        gathered[rows], gate[rows],
        {name: target[rows] for name, target in field_targets.items()},
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
        try:
            line = batcher.sidecars.line_for(key)
        except KeyError:
            line = None
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
