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
import math
from dataclasses import dataclass

import numpy as np
import torch

from effects.application.surface_batching import SurfaceBatcher
from effects.domain.effect_model import (
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    FieldType,
    entity_target_tensors,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.records import RecordKind

logger = logging.getLogger(__name__)

#: Fields whose magnitude the Poisson deviance is read over — "how much damage",
#: "how many cards". A model not reading the text can only answer at the corpus
#: average.
_COUNT_TYPES = (FieldType.COUNT, FieldType.SIGNED_DELTA)

BATCH_RECORDS = 32


@dataclass(frozen=True, slots=True)
class GateOneMetrics:
    """One model's scores on the card-disjoint split's unique-text stratum."""

    affected_gate_f1: float
    zone_outcome_accuracy: float
    mean_poisson_deviance: float


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


def recency_cutoff(
    first_printing: dict[str, str], *, fraction: float = 0.08,
) -> str | None:
    """The first-printing date that separates "newest sets" from the rest.

    A date rather than a set list, and derived the way a recency-ordered holdout
    would have chosen its cards: take them newest-first until they cover
    ``fraction`` of the corpus, and the last one's date is the boundary. The
    breakdown then reports on the same population that holdout would have
    tested, without anyone maintaining a second holdout.

    A card with no printing date sorts as oldest and never sets the boundary.
    """
    dated = sorted(
        (date for date in first_printing.values() if date), reverse=True,
    )
    if not dated:
        return None
    target = max(1, math.ceil(len(first_printing) * fraction))
    return dated[min(target, len(dated)) - 1]


def partition_by_recency(
    items,
    first_printing: dict[str, str],
    *,
    card_of,
    since: str,
) -> tuple[list, list]:
    """Split a stratum into recently-printed and older, by first printing.

    A text hash does not prefer novel mechanics, and novel mechanics are the
    harder generalization: new keywords, new templating, parameter combinations
    nothing older uses. Reporting the gate-1 margins on both halves recovers
    what a recency-ordered holdout tested, at the cost of one column rather than
    a second holdout and a second depleted corpus (FR-088c).

    A card with no printing date counts as older. Dates are ISO 8601, so the
    comparison is a string comparison.
    """
    recent, older = [], []
    for item in items:
        card = card_of(item)
        printed = first_printing.get(card) if card else None
        (recent if printed and printed >= since else older).append(item)
    return recent, older


def measure(
    records, encoder, model, batcher: SurfaceBatcher, *, fields,
) -> GateOneMetrics:
    """Run the model over ``records`` and score the three margins.

    ``fields`` is the active-field tuple the run trains with, so the gate reads
    the same heads the loss shaped.
    """
    gate_pred: list[np.ndarray] = []
    gate_true: list[np.ndarray] = []
    zone_pred: list[np.ndarray] = []
    zone_true: list[np.ndarray] = []
    count_pred: list[np.ndarray] = []
    count_true: list[np.ndarray] = []

    zone_spec = FIELDS_BY_NAME.get("zone_outcome")
    count_specs = [spec for spec in fields if spec.type in _COUNT_TYPES]

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
            ).cpu()

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
    )


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
