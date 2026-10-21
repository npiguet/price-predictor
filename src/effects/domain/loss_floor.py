"""The constant-predictor floor (FR-127c): the loss the best constant reaches.

A field's loss is a number in nats with no scale of its own; scaled against
the floor it says whether the head learned the field or only how often it fires.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    EntityFieldTargets,
    FieldSpec,
    FieldType,
)
from effects.domain.effect_losses import capped, counted, per_entity_loss

#: A guard on the division, not the detector: a field whose targets never vary
#: is given a floor of exactly zero by :func:`constant_predictor_floor`, and
#: this is what keeps the ratio off a floor too small to divide by anyway.
FLOOR_EPSILON = 1e-9

#: Rates are clamped away from zero and one before a log: an outcome the sample
#: never showed would otherwise put the constant predictor at minus infinity and
#: the floor at nan.
_RATE_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class EntityTargetBatch:
    """One scored batch's per-entity targets, held apart from the model.

    The floor is a property of the targets alone, so a batch is scored twice
    from one set of tensors: once against what the head predicted, once against
    the best constant. Keeping the targets rather than re-deriving them is what
    guarantees the two numbers were measured over the same entities.
    """

    gate: torch.Tensor
    fields: EntityFieldTargets
    mask: torch.Tensor


def _mean(values: torch.Tensor) -> torch.Tensor:
    """The mean, or zero where nothing was selected — no entity, no statistic."""
    if values.numel() == 0:
        return torch.zeros(())
    return values.mean()


def _logit(rate: torch.Tensor) -> torch.Tensor:
    """The logit a BCE-with-logits loss reads back as ``rate``."""
    rate = rate.clamp(_RATE_FLOOR, 1.0 - _RATE_FLOOR)
    return torch.log(rate / (1.0 - rate))


def _log_frequencies(counts: torch.Tensor) -> torch.Tensor:
    """Log class frequencies, which cross-entropy reads back as frequencies."""
    return torch.log((counts / counts.sum()).clamp(min=_RATE_FLOOR))


def _supervised(
    batches: Sequence[EntityTargetBatch], name: str,
) -> torch.Tensor | None:
    """One field's targets over every supervised entity of every batch.

    Supervised is where the gate's *target* fires and the field is present,
    which is the same selection :func:`per_entity_loss` scores it on.
    """
    parts = [
        batch.fields.values[name][
            batch.mask.bool() & batch.gate.bool() & batch.fields.present[name].bool()
        ]
        for batch in batches if name in batch.fields
    ]
    if not parts:
        return None
    selected = torch.cat(parts)
    return selected if selected.numel() else None


def _degenerate(batches: Sequence[EntityTargetBatch], name: str) -> bool:
    """Whether a field's supervised targets never vary.

    Such a field has no deviance for a head to explain, and the number the rate
    clamp leaves behind is not a floor: an all-zero count floors at about a
    millionth of a nat, so a head idling at a log-rate of -5 would report
    hundreds of thousands of percent worse than constant. The sparse fields
    this happens to are the ones the breakdown exists to make legible, so they
    report no deviance rather than a ratio against nothing.

    Whole rows are compared, so a multi-binary field one of whose bits never
    fires is not degenerate — that bit rides the clamp inside a field the rest
    of which still varies.
    """
    target = _supervised(batches, name)
    return target is None or bool((target == target[0]).all())


def constant_predictor_outputs(
    batches: Sequence[EntityTargetBatch], *, fields: tuple[FieldSpec, ...],
) -> torch.Tensor:
    """The best constant prediction for the gate and each active field.

    One ``(PER_ENTITY_WIDTH,)`` vector to broadcast over every entity of every
    batch. Each slice holds the maximum-likelihood constant under that field's
    own loss — the logit of a positive rate under BCE, the log of a mean under
    Poisson, log class frequencies under cross-entropy — which is what makes the
    loss it scores a *floor* rather than one arbitrary constant's number.
    """
    vector = torch.zeros(PER_ENTITY_WIDTH)
    real = [batch.gate[batch.mask.bool()] for batch in batches]
    vector[GATE_INDEX] = _logit(_mean(torch.cat(real or [torch.zeros(0)])))

    for spec in fields:
        target = _supervised(batches, spec.name)
        if target is None:
            continue
        start, end = FIELD_SLICES[spec.name]
        match spec.type:
            case FieldType.BINARY:
                vector[start:end] = _logit(_mean(target))
            case FieldType.MULTI_BINARY:
                vector[start:end] = _logit(target.mean(dim=0))
            case FieldType.COUNT:
                vector[start:end] = torch.log(
                    _mean(counted(target)).clamp(min=_RATE_FLOOR)
                )
            case FieldType.SIGNED_DELTA:
                target = capped(target)
                direction = torch.sign(target).long().reshape(-1) + 1
                vector[start:start + 3] = _log_frequencies(
                    torch.bincount(direction, minlength=3).float()
                )
                vector[start + 3] = torch.log(
                    _mean(target.abs()).clamp(min=_RATE_FLOOR)
                )
            case FieldType.CATEGORICAL:
                vector[start:end] = _log_frequencies(
                    torch.bincount(
                        target.long().reshape(-1), minlength=spec.arity,
                    ).float()
                )
    return vector


def constant_predictor_floor(
    batches: Sequence[EntityTargetBatch], *, fields: tuple[FieldSpec, ...],
) -> dict[str, float]:
    """The loss the best constant prediction reaches, by field.

    Scored through :func:`per_entity_loss` itself rather than from closed-form
    deviance formulas, so the floor carries the same masking, the same Stirling
    terms and the same per-record normalization as the number it is the floor
    for. A floor derived a second way would differ from the loss by whichever of
    those three it got wrong, and the difference would read as a head having
    learned something.

    Averaged over the batches that carried each field, the way the validation
    pass averages the losses it scales. A field whose targets never vary is
    reported as exactly zero rather than at whatever the rate clamp left it, so
    the caller reads it as the absence of a floor. The gate is never treated
    that way: it is supervised on every entity and always has a base rate.
    """
    vector = constant_predictor_outputs(batches, fields=fields)
    degenerate = {
        spec.name for spec in fields if _degenerate(batches, spec.name)
    }
    means = PartsMean()
    for batch in batches:
        _, parts = per_entity_loss(
            vector.expand(*batch.gate.shape, PER_ENTITY_WIDTH),
            batch.gate, batch.fields, batch.mask,
            fields=fields, report_parts=True,
        )
        means.add(parts)
    return {
        name: 0.0 if name in degenerate else mean
        for name, mean in means.means().items()
    }


class PartsMean:
    """Each field's loss averaged over the batches that carried it.

    Over those batches rather than every batch: a field a batch did not
    supervise contributes no zero, which would read as the model having got
    it right.
    """

    def __init__(self) -> None:
        self._summed: dict[str, float] = defaultdict(float)
        self._counted: dict[str, int] = defaultdict(int)

    def add(self, parts: Mapping[str, float]) -> None:
        for name, value in parts.items():
            self._summed[name] += value
            self._counted[name] += 1

    def means(self) -> dict[str, float]:
        return {name: self._summed[name] / self._counted[name] for name in self._summed}
