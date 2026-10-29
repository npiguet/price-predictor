"""The constant-predictor floor (FR-127c): the loss the best constant reaches.

A field's loss is a number in nats with no scale of its own; scaled against
the floor it says whether the head learned the field or only how often it fires.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch

from effects.domain.effect_head_layout import (
    DIRECTIONS,
    FIELD_SLICES,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    SIGNED_DIRECTIONS,
    SIGNED_MAGNITUDE,
    FieldSpec,
    FieldType,
)
from effects.domain.effect_losses import (
    LossParts,
    capped,
    counted,
    delta_direction,
    per_entity_loss,
)
from effects.domain.entity_field_targets import EntityTargetBatch

#: A guard on dividing by a floor, not the degenerate-field detector.
FLOOR_EPSILON = 1e-9

#: Rates are clamped away from zero and one before a log, or an outcome the
#: sample never showed would make the floor nan.
_RATE_FLOOR = 1e-6


def _mean(values: torch.Tensor) -> torch.Tensor:
    """The mean, or zero where nothing was selected."""
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
    """One field's targets over every supervised entity of every batch, the
    selection :func:`per_entity_loss` scores."""
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
    """Whether a field's supervised targets never vary, whole rows compared.

    Such a field has no deviance to explain, and the clamp leaves a near-zero
    floor any head would read as enormously worse than constant.
    """
    target = _supervised(batches, name)
    return target is None or bool((target == target[0]).all())


def constant_predictor_outputs(
    batches: Sequence[EntityTargetBatch], *, fields: tuple[FieldSpec, ...],
) -> torch.Tensor:
    """The best constant prediction for the gate and each active field: one
    ``(PER_ENTITY_WIDTH,)`` vector, each slice the maximum-likelihood constant
    under that field's own loss."""
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
                direction = delta_direction(target).reshape(-1)
                columns = vector[start:end]
                columns[SIGNED_DIRECTIONS] = _log_frequencies(
                    torch.bincount(direction, minlength=len(DIRECTIONS)).float()
                )
                columns[SIGNED_MAGNITUDE] = torch.log(
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

    Scored through :func:`per_entity_loss` itself, so the floor carries the
    loss's own masking and normalization. A field whose targets never vary
    reports exactly zero: no floor.
    """
    vector = constant_predictor_outputs(batches, fields=fields)
    degenerate = {
        spec.name for spec in fields if _degenerate(batches, spec.name)
    }
    means = PartsMean()
    for batch in batches:
        _, parts = per_entity_loss(
            vector.expand(*batch.gate.shape, PER_ENTITY_WIDTH), batch,
            fields=fields, report_parts=True,
        )
        means.add(parts)
    return {
        name: 0.0 if name in degenerate else mean
        for name, mean in means.means().items()
    }


class PartsMean:
    """Each field's loss averaged over the batches that carried it, summed on
    the device and read back once in :meth:`means`."""

    def __init__(self) -> None:
        self._summed: dict[str, torch.Tensor] = {}
        self._counted: dict[str, int] = defaultdict(int)

    def add(self, parts: LossParts) -> None:
        if parts.values is None:
            return
        for name, value in zip(parts.names, parts.values.unbind()):
            held = self._summed.get(name)
            self._summed[name] = value if held is None else held + value
            self._counted[name] += 1

    def means(self) -> dict[str, float]:
        if not self._summed:
            return {}
        sums = torch.stack(list(self._summed.values())).tolist()
        return {
            name: summed / self._counted[name]
            for name, summed in zip(self._summed, sums)
        }
