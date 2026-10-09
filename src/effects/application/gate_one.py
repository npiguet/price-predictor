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
from torch.nn import functional

from effects.application.breakdowns import TOTAL, RecordResult, normalize_keyword
from effects.application.surface_batching import SurfaceBatcher
from effects.domain.damage_step_keywords import KeywordResolver, keyword_of_line
from effects.domain.effect_model import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    FieldType,
    entity_target_tensors,
    field_loss,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.records import RecordKind
from effects.domain.rule_families import rule_family

logger = logging.getLogger(__name__)

#: Fields whose magnitude the Poisson deviance is read over — "how much damage",
#: "how many cards". A model not reading the text can only answer at the corpus
#: average.
_COUNT_TYPES = (FieldType.COUNT, FieldType.SIGNED_DELTA)

BATCH_RECORDS = 32


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
    for key in record.ability or ():
        text = batcher.text_of(key)
        if text is not None:
            return text
    return None


def training_texts(records, batcher: SurfaceBatcher) -> set[str]:
    """Every acting-line text the training games contain."""
    texts = set()
    for record in records:
        text = acting_text(record, batcher)
        if text is not None:
            texts.add(text)
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
    gate_pred: list[np.ndarray] = []
    gate_true: list[np.ndarray] = []
    zone_pred: list[np.ndarray] = []
    zone_true: list[np.ndarray] = []
    count_pred: list[np.ndarray] = []
    count_true: list[np.ndarray] = []

    zone_spec = FIELDS_BY_NAME.get("zone_outcome")
    count_specs = [spec for spec in fields if spec.type in _COUNT_TYPES]
    results: list[RecordResult] = []
    # One resolver for the whole pass: it memoizes each provenance key's
    # keyword, and the stratum repeats the same few hundred cards.
    resolver = KeywordResolver(batcher.sidecars)

    encoder.eval()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), BATCH_RECORDS):
            chunk = records[start : start + BATCH_RECORDS]
            batch, surfaces = batcher.build(chunk, encoder)
            outputs = model.per_entity(model(**batch))
            targets = [derive_targets(record) for record in chunk]
            gate, field_targets, mask, index = entity_target_tensors(
                surfaces, targets, fields,
            )
            index = index.to(outputs.device)
            gathered = outputs.gather(
                1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]),
            ).float().cpu()
            for row, record in enumerate(chunk):
                results.append(_record_result(
                    record, batcher, resolver,
                    _record_losses(row, gathered, gate, field_targets, mask, fields),
                    legality_mode_counts,
                ))

            real = mask.bool()
            gate_pred.append(
                (gathered[..., GATE_INDEX][real] > 0).numpy().astype(np.int8)
            )
            gate_true.append(gate[real].numpy().astype(np.int8))

            affected = real & gate.bool()
            if zone_spec is not None and "zone_outcome" in field_targets:
                start_i, end_i = FIELD_SLICES["zone_outcome"]
                logits = gathered[..., start_i:end_i][affected]
                if logits.numel():
                    zone_pred.append(logits.argmax(-1).numpy())
                    zone_true.append(
                        field_targets["zone_outcome"][affected].numpy()
                    )
            for spec in count_specs:
                target = field_targets.get(spec.name)
                if target is None:
                    continue
                start_i, end_i = FIELD_SLICES[spec.name]
                slice_ = gathered[..., start_i:end_i][affected]
                if not slice_.numel():
                    continue
                # A signed delta predicts direction then magnitude; the
                # magnitude is the last column either way.
                log_rate = slice_[..., -1]
                count_pred.append(torch.exp(log_rate).numpy())
                count_true.append(np.abs(target[affected].numpy()))

    return GateOneMetrics(
        affected_gate_f1=_f1(gate_pred, gate_true),
        zone_outcome_accuracy=_accuracy(zone_pred, zone_true),
        mean_poisson_deviance=_deviance(count_pred, count_true),
        records=tuple(results),
    )


def _record_losses(
    row: int, gathered: torch.Tensor, gate: torch.Tensor,
    field_targets: dict[str, torch.Tensor], mask: torch.Tensor, fields,
) -> dict[str, float]:
    """One record's per-entity loss by field, as ``per_entity_loss`` sums it.

    The same terms the trainer's loss adds for this record alone: the gate over
    every real entity, then each field over the entities the gate's *target*
    marks affected. A field with no affected entity is not supervised here and
    is left out rather than reported as zero. Read on the host, where the batch
    already is, so scoring a record costs no device synchronization.
    """
    real = mask[row].bool()
    gate_loss = functional.binary_cross_entropy_with_logits(
        gathered[row, :, GATE_INDEX][real], gate[row][real].float(), reduction="sum",
    )
    losses = {"gate": float(gate_loss)}
    affected = real & gate[row].bool()
    if bool(affected.any()):
        for spec in fields:
            target = field_targets.get(spec.name)
            if target is None:
                continue
            start, end = FIELD_SLICES[spec.name]
            losses[spec.name] = float(field_loss(
                spec, gathered[row, :, start:end][affected], target[row][affected],
            ))
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
        random_seat=record.random_seat,
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
    for key in record.ability or ():
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
    from effects.application.evaluate_effect_model import f1_score

    predicted, observed = _cat(pred), _cat(true)
    return f1_score(predicted, observed) if predicted.size else float("nan")


def _accuracy(pred: list[np.ndarray], true: list[np.ndarray]) -> float:
    predicted, observed = _cat(pred), _cat(true)
    if not predicted.size:
        return float("nan")
    return float(np.mean(predicted == observed))


def _deviance(pred: list[np.ndarray], true: list[np.ndarray]) -> float:
    from effects.application.evaluate_effect_model import poisson_deviance

    predicted, observed = _cat(pred), _cat(true)
    return poisson_deviance(predicted, observed) if predicted.size else float("nan")
