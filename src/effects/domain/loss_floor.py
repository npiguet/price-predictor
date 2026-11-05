"""The constant-predictor floor (FR-127c, FR-062, FR-068): the loss the best
constant prediction reaches, by head and term.

A field's loss is a number in nats with no scale of its own; scaled against
the floor it says whether the head learned the field or only how often it
fires. One function, :func:`constant_predictor_losses`, gives it for every
shipped head over any sample of targets: a training sample's for the loss
denominators (FR-062), a validation stratum's for the explained deviance
(FR-068). Each term is reported in nats per record over every record of the
sample, the unit a step's loss term is in, so a term the sample rarely
supervises has a small floor rather than its loss where it fires.

The constant is each term's maximum-likelihood one: a rate, class
frequencies, or a mean, and for a count a negative-binomial dispersion fitted
around that mean by moments and then Newton steps (research T2). The value,
script-API and MLM heads are training-only auxiliaries whose scale their own
weight flags set, and have no floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from effects.domain.aux_heads import Head
from effects.domain.effect_head_input import MAGNITUDE_CAP
from effects.domain.effect_head_layout import (
    CREATED_DISPERSED,
    CREATED_MAGNITUDES,
    CREATED_OBJECT_SLOTS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    CREATED_TERMS,
    DISPERSED_FIELDS,
    DISPERSION_INDEX,
    FIELD_SLICES,
    GATE_INDEX,
    GLOBAL_LAYOUT,
    PER_ENTITY_WIDTH,
    VERDICT_LAYOUT,
    FieldSpec,
    FieldType,
    RecordHeadLayout,
)
from effects.domain.effect_losses import (
    LossParts,
    created_objects_terms,
    per_entity_loss,
    record_head_terms,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import (
    FIELD_KINDS,
    LOG_DISPERSION_MAX,
    counted,
    fit_log_dispersion,
)

#: A guard on dividing by a floor, not the degenerate-field detector.
FLOOR_EPSILON = 1e-9

#: The per-entity head's key among the heads a floor is reported for; the
#: others are :class:`Head` values.
PER_ENTITY = "per_entity"
#: Every head with a floor, in the order a run reports them.
FLOOR_HEADS: tuple[str, ...] = (
    PER_ENTITY, Head.VERDICT.value, Head.CREATED_OBJECTS.value, Head.GLOBAL.value,
)
#: The per-record heads' layouts.
RECORD_LAYOUTS: dict[str, RecordHeadLayout] = {
    Head.VERDICT.value: VERDICT_LAYOUT, Head.GLOBAL.value: GLOBAL_LAYOUT,
}

#: ``head -> term -> loss``, nats per record.
HeadLosses = dict[str, dict[str, float]]


@dataclass
class TargetSample:
    """A sample's host targets, batch by batch: each batch's per-entity
    targets and its shipped heads' ``(target, mask[, known])``, the tuples the
    step preparation builds; a head no record of a batch supervises is absent
    from that batch."""

    entity: list[EntityTargetBatch] = field(default_factory=list)
    heads: list[Mapping[str, tuple]] = field(default_factory=list)

    def add(self, entity: EntityTargetBatch, heads: Mapping[str, tuple] | None = None) -> None:
        self.entity.append(entity)
        self.heads.append({str(head): value for head, value in (heads or {}).items()})

    @classmethod
    def of(cls, batches: Sequence[EntityTargetBatch]) -> TargetSample:
        sample = cls()
        for batch in batches:
            sample.add(batch)
        return sample

    @property
    def records(self) -> int:
        return sum(int(batch.gate.shape[0]) for batch in self.entity)

    def head_batches(self, head: str) -> list[tuple[int, tuple]]:
        """``(records, targets)`` of every batch carrying ``head``."""
        return [
            (int(entity.gate.shape[0]), heads[head])
            for entity, heads in zip(self.entity, self.heads) if head in heads
        ]


def constant_predictor_losses(
    sample: TargetSample, *, fields: tuple[FieldSpec, ...],
) -> HeadLosses:
    """The loss the best constant prediction reaches on ``sample``, by head
    and term, in nats per record over all its records.

    Scored through the heads' own loss functions, so each floor carries the
    loss's masking and normalization. A term whose supervised targets never
    vary reports exactly zero: it has no floor. A term the sample never
    supervises is absent. ``fields`` are the per-entity fields to cover.
    """
    losses: HeadLosses = {PER_ENTITY: _per_entity_losses(sample, fields)}
    for head, layout in RECORD_LAYOUTS.items():
        found = _record_head_losses(sample, head, layout)
        if found:
            losses[head] = found
    created = _created_objects_losses(sample)
    if created:
        losses[Head.CREATED_OBJECTS.value] = created
    return {head: losses[head] for head in FLOOR_HEADS if head in losses}


# ── the per-entity head ──


def _supervised(
    batches: Sequence[EntityTargetBatch], name: str,
) -> torch.Tensor | None:
    """One field's targets over every supervised entity of every batch, the
    selection :func:`per_entity_loss` scores."""
    parts = [
        batch.fields.values[name][batch.scored(name)]
        for batch in batches if name in batch.fields
    ]
    if not parts:
        return None
    selected = torch.cat(parts)
    return selected if selected.numel() else None


def _degenerate(target: torch.Tensor | None) -> bool:
    """Whether a term's supervised targets never vary, whole rows compared.

    Such a term has no deviance to explain, and the clamp leaves a near-zero
    floor any head would read as enormously worse than constant.
    """
    return target is None or target.numel() == 0 or bool((target == target[0]).all())


def constant_predictor_outputs(
    batches: Sequence[EntityTargetBatch], *, fields: tuple[FieldSpec, ...],
    supervised: Mapping[str, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """The best constant prediction for the gate and each active field: one
    ``(PER_ENTITY_WIDTH,)`` vector, each slice the maximum-likelihood constant
    under that field's own loss. ``supervised`` holds each field's
    :func:`_supervised` targets where the caller selected them already."""
    if supervised is None:
        supervised = {spec.name: _supervised(batches, spec.name) for spec in fields}
    vector = torch.zeros(PER_ENTITY_WIDTH)
    # The gate is one binary column over every entity it trains on.
    real = [batch.gate[batch.gate_scored] for batch in batches]
    vector[GATE_INDEX] = FIELD_KINDS[FieldType.BINARY].best_constant(
        torch.cat(real or [torch.zeros(0)]), arity=1, cap=MAGNITUDE_CAP,
    )

    for spec in fields:
        target = supervised[spec.name]
        if target is None:
            continue
        start, end = FIELD_SLICES[spec.name]
        vector[start:end] = FIELD_KINDS[spec.type].best_constant(
            target, arity=spec.arity, cap=MAGNITUDE_CAP,
        )
    return vector


def constant_predictor_dispersion(
    batches: Sequence[EntityTargetBatch], *, fields: tuple[FieldSpec, ...],
    supervised: Mapping[str, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """The constant predictor's log-dispersions, in :data:`DISPERSED_FIELDS`
    order: each count's fitted around its constant mean (FR-063). A field the
    sample does not score keeps the upper bound, which nothing reads."""
    if supervised is None:
        supervised = {spec.name: _supervised(batches, spec.name) for spec in fields}
    log_dispersion = torch.full((len(DISPERSED_FIELDS),), LOG_DISPERSION_MAX)
    for spec in fields:
        target = supervised[spec.name]
        if target is None or not spec.type.has_magnitude:
            continue
        magnitude = FIELD_KINDS[spec.type].magnitude(target, cap=MAGNITUDE_CAP)
        log_dispersion[DISPERSION_INDEX[spec.name]] = fit_log_dispersion(
            magnitude, magnitude.mean(),
        )
    return log_dispersion


def _per_entity_losses(sample: TargetSample, fields: tuple[FieldSpec, ...]) -> dict[str, float]:
    batches = sample.entity
    # Selected once: each field's targets are a concatenation over every batch.
    supervised = {spec.name: _supervised(batches, spec.name) for spec in fields}
    vector = constant_predictor_outputs(batches, fields=fields, supervised=supervised)
    log_dispersion = constant_predictor_dispersion(
        batches, fields=fields, supervised=supervised,
    )
    degenerate = {name for name, target in supervised.items() if _degenerate(target)}
    means = PartsMean()
    for batch in batches:
        _, parts = per_entity_loss(
            vector.expand(*batch.gate.shape, PER_ENTITY_WIDTH), batch,
            fields=fields, log_dispersion=log_dispersion, report_parts=True,
        )
        means.add(parts, rows=int(batch.gate.shape[0]))
    return {
        name: 0.0 if name in degenerate else mean
        for name, mean in means.means().items()
    }


def constant_predictor_floor(
    batches: Sequence[EntityTargetBatch], *, fields: tuple[FieldSpec, ...],
) -> dict[str, float]:
    """The per-entity head's floor by term: :func:`constant_predictor_losses`
    over per-entity targets alone."""
    return constant_predictor_losses(TargetSample.of(batches), fields=fields)[PER_ENTITY]


# ── the per-record heads ──


def _record_head_losses(
    sample: TargetSample, head: str, layout: RecordHeadLayout,
) -> dict[str, float]:
    batches = sample.head_batches(head)
    if not batches:
        return {}
    masks = [_field_mask(targets[1], layout) for _, targets in batches]
    outputs = torch.zeros(layout.width)
    log_dispersion = torch.full((len(layout.dispersed),), LOG_DISPERSION_MAX)
    degenerate: set[str] = set()
    supervised: set[str] = set()
    for position, spec in enumerate(layout.fields):
        first, last = layout.target_slices[spec.name]
        rows = [
            targets[0][mask[:, position]][:, first:last].float()
            for (_, targets), mask in zip(batches, masks)
        ]
        wanted = torch.cat(rows) if rows else torch.zeros(0, last - first)
        if wanted.shape[0] == 0:
            continue
        supervised.add(str(spec.name))
        if _degenerate(wanted):
            degenerate.add(str(spec.name))
        kind = FIELD_KINDS[spec.type]
        member_width = spec.type.width(spec.arity)
        start, _ = layout.slices[spec.name]
        for member in range(spec.count):
            offset = start + member * member_width
            outputs[offset:offset + member_width] = kind.best_constant(
                wanted[:, member], arity=spec.arity, cap=MAGNITUDE_CAP,
            )
        if spec.type.has_magnitude:
            magnitude = kind.magnitude(wanted, cap=MAGNITUDE_CAP)
            log_dispersion[layout.dispersion_index[str(spec.name)]] = fit_log_dispersion(
                magnitude, magnitude.mean(0),
            )
    summed = torch.zeros(len(layout.fields), dtype=torch.float64)
    for (_, targets), mask in zip(batches, masks):
        summed += record_head_terms(
            outputs.expand(int(targets[0].shape[0]), layout.width), targets[0], mask,
            layout, log_dispersion=log_dispersion,
        ).double()
    records = max(sample.records, 1)
    return {
        name: 0.0 if name in degenerate else float(value) / records
        for name, value in zip(layout.terms, summed.tolist())
        if name in supervised
    }


def _field_mask(mask: torch.Tensor, layout: RecordHeadLayout) -> torch.Tensor:
    """A per-record head's mask as ``(records, fields)`` bools."""
    mask = mask.detach().bool().cpu()
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1).expand(-1, len(layout.fields))
    return mask


# ── the created-objects head ──


def _created_objects_losses(sample: TargetSample) -> dict[str, float]:
    batches = [
        (records, _created_rows(targets))
        for records, targets in sample.head_batches(Head.CREATED_OBJECTS.value)
    ]
    batches = [(records, rows) for records, rows in batches if rows is not None]
    if not batches:
        return {}
    target = torch.cat([rows[0] for _, rows in batches])
    known = torch.cat([rows[1] for _, rows in batches])
    outputs, log_dispersion = _created_constant(target, known)
    summed = torch.zeros(len(CREATED_TERMS), dtype=torch.float64)
    for _, (wanted, said) in batches:
        summed += created_objects_terms(
            outputs.expand(int(wanted.shape[0]), CREATED_OBJECTS_WIDTH), wanted,
            known=said, log_dispersion=log_dispersion,
        ).double()
    records = max(sample.records, 1)
    slot_known = known[:, :-1].reshape(-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    slot_target = target[:, :-1].reshape(-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    losses = {}
    for column, name in enumerate(CREATED_TERMS):
        if column < CREATED_SLOT_WIDTH:
            said = slot_known[..., column] > 0
            values = slot_target[..., column][said]
        else:
            said = known[:, -1] > 0
            values = target[:, -1][said]
        if values.numel() == 0:
            continue
        losses[name] = 0.0 if _degenerate(values) else float(summed[column]) / records
    return losses


def _created_rows(targets: tuple) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The supervising records' ``(target, known)``, or None for none."""
    target, mask = targets[0], targets[1]
    known = targets[2] if len(targets) > 2 and targets[2] is not None else torch.ones_like(target)
    rows = mask.detach().bool().cpu()
    if not rows.any():
        return None
    return target[rows].float(), known[rows].float()


def _created_constant(
    target: torch.Tensor, known: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The created-objects head's best constant: each slot column's rate or
    mean over the records that say it, the overflow flag's rate, and a
    log-dispersion per magnitude column, shared by the slots."""
    slots = (-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    wanted = target[:, :-1].reshape(slots)
    said = known[:, :-1].reshape(slots)
    magnitudes = CREATED_MAGNITUDES
    wanted = torch.cat([
        wanted[..., :magnitudes.start],
        counted(wanted[..., magnitudes], MAGNITUDE_CAP),
        wanted[..., magnitudes.stop:],
    ], dim=-1)
    mean = (wanted * said).sum(0) / said.sum(0).clamp(min=1.0)
    binary = FIELD_KINDS[FieldType.BINARY]
    count = FIELD_KINDS[FieldType.COUNT]
    columns = torch.empty(CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    for slot in range(CREATED_OBJECT_SLOTS):
        for column in range(CREATED_SLOT_WIDTH):
            rate = mean[slot, column].reshape(1)
            kind = count if magnitudes.start <= column < magnitudes.stop else binary
            columns[slot, column] = kind.best_constant(rate, arity=1, cap=None)
    overflow = target[:, -1][known[:, -1] > 0]
    outputs = torch.cat([
        columns.reshape(-1),
        binary.best_constant(overflow, arity=1, cap=None).reshape(1),
    ])
    log_dispersion = torch.full((len(CREATED_DISPERSED),), LOG_DISPERSION_MAX)
    for index, column in enumerate(range(magnitudes.start, magnitudes.stop)):
        present = said[..., column] > 0
        if not present.any():
            continue
        values = wanted[..., column]
        means = mean[:, column].expand_as(values)
        log_dispersion[index] = fit_log_dispersion(values[present], means[present])
    return outputs, log_dispersion


class PartsMean:
    """Each term's loss per record over every record of the batches added,
    a term a batch does not carry counting zero there: the unit of
    :func:`constant_predictor_losses`. Summed on the device and read back once
    in :meth:`means`."""

    def __init__(self) -> None:
        self._summed: dict[str, torch.Tensor] = {}
        self._rows = 0

    def add(self, parts: LossParts, *, rows: int) -> None:
        """Fold in one batch of ``rows`` records, its ``parts`` per record."""
        self._rows += int(rows)
        if parts.values is None:
            return
        for name, value in zip(parts.names, parts.values.unbind()):
            weighted = value * int(rows)
            held = self._summed.get(name)
            self._summed[name] = weighted if held is None else held + weighted

    def means(self) -> dict[str, float]:
        if not self._summed:
            return {}
        sums = torch.stack(list(self._summed.values())).tolist()
        return {
            name: summed / max(self._rows, 1)
            for name, summed in zip(self._summed, sums)
        }
