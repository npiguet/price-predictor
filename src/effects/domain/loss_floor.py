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

A sample prepared under ``--rule-mask`` (FR-061) carries each target's
weight, and every term's loss is weighted as the run weights it (research
T3): a target the mask drops is not in the sample, a fractional one counts
at its weight. The constant itself is fitted over the targets the sample
holds, unweighted; under the soft mask that puts the denominator a little
above the weighted optimum's loss, never below it.

The constant is each term's maximum-likelihood one: a rate, class
frequencies, or a mean, and for a count a negative-binomial dispersion fitted
around that mean by moments and then Newton steps (research T2). The value,
script-API and MLM heads are training-only auxiliaries whose scale their own
weight flags set, and have no floor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
    GATE_TERM,
    GLOBAL_LAYOUT,
    PER_ENTITY_WIDTH,
    VERDICT_LAYOUT,
    FieldSpec,
    FieldType,
    RecordHeadLayout,
)
from effects.domain.effect_losses import (
    LossParts,
    created_objects_row_terms,
    created_objects_terms,
    record_head_row_terms,
    record_head_terms,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import (
    FIELD_KINDS,
    LOG_DISPERSION_MAX,
    counted,
    fit_log_dispersion,
)
from effects.domain.selection_set import PER_ENTITY_HEAD

#: A guard on dividing by a floor, not the degenerate-field detector.
FLOOR_EPSILON = 1e-9

#: The per-entity head's key among the heads a floor is reported for; the
#: others are :class:`Head` values.
PER_ENTITY = PER_ENTITY_HEAD
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


class TargetSample:
    """A sample's supervised targets, gathered batch by batch and kept
    compact: the gate's targets where it trains, each per-entity field's where
    it is scored, and each shipped head's ``(target, mask[, known])`` rows that
    supervise anything; beside the per-entity targets, their rule-mask
    weights (FR-061). Compact because the floor needs nothing else (the
    constant is the same at every entity) and a loss-denominator sample holds
    thousands of records."""

    def __init__(self) -> None:
        self.records = 0
        self._gate: list[torch.Tensor] = []
        self._gate_weights: list[torch.Tensor] = []
        self._fields: dict[str, list[torch.Tensor]] = {}
        self._field_weights: dict[str, list[torch.Tensor]] = {}
        self._heads: dict[str, list[tuple]] = {}

    def add(self, entity: EntityTargetBatch, heads: Mapping | None = None) -> None:
        """Fold in one batch: its per-entity targets and its shipped heads'
        tuples, keyed by :class:`Head` or its value; a head a batch lacks
        supervises nothing there."""
        self.records += int(entity.gate.shape[0])
        gate_scored = entity.gate_scored
        self._gate.append(entity.gate.detach().cpu()[gate_scored])
        self._gate_weights.append(_weights_at(entity.gate_weights(), gate_scored))
        for name, values in entity.fields.values.items():
            scored = entity.scored(name)
            selected = values.detach().cpu()[scored]
            if selected.numel():
                self._fields.setdefault(name, []).append(selected)
                self._field_weights.setdefault(name, []).append(
                    _weights_at(entity.field_weights(name), scored),
                )
        for head, targets in (heads or {}).items():
            compact = _supervising_rows(str(head), targets)
            if compact is not None:
                self._heads.setdefault(str(head), []).append(compact)

    @classmethod
    def of(cls, batches: Sequence[EntityTargetBatch]) -> TargetSample:
        sample = cls()
        for batch in batches:
            sample.add(batch)
        return sample

    @property
    def gate(self) -> torch.Tensor:
        """The gate's targets over every entity it trains on."""
        return torch.cat(self._gate) if self._gate else torch.zeros(0)

    @property
    def gate_weights(self) -> torch.Tensor | None:
        """The rule mask's weight on each of :attr:`gate`'s targets; None
        where every one weighs one."""
        return _unless_unit(self._gate_weights)

    def weights(self, name: str) -> torch.Tensor | None:
        """The rule mask's weight on each of :meth:`supervised`'s targets;
        None where every one weighs one."""
        return _unless_unit(self._field_weights.get(name, []))

    def supervised(self, name: str) -> torch.Tensor | None:
        """One per-entity field's targets over every entity it is scored at,
        the selection :func:`per_entity_loss` scores; None where it is never
        scored."""
        parts = self._fields.get(name)
        return torch.cat(parts) if parts else None

    def head_batches(self, head: str) -> list[tuple]:
        """The supervising rows of each batch carrying ``head``."""
        return self._heads.get(head, [])


def _weights_at(weights: torch.Tensor | None, scored: torch.Tensor) -> torch.Tensor:
    """``weights`` at the entities ``scored`` selects, one where None."""
    if weights is None:
        return torch.ones(int(scored.sum()))
    return weights[scored]


def _unless_unit(parts: list[torch.Tensor]) -> torch.Tensor | None:
    """``parts`` joined, or None where none weighs other than one, so an
    unmasked sample sums exactly as it always has."""
    if not parts:
        return None
    joined = torch.cat(parts)
    return None if bool((joined == 1).all()) else joined


def _weighted_sum(terms: torch.Tensor, weights: torch.Tensor | None) -> float:
    """``terms`` summed, each at its weight."""
    if weights is None:
        return float(terms.sum())
    return float((terms.double() * weights.double().reshape(-1)).sum())


def _supervising_rows(head: str, targets: tuple) -> tuple | None:
    """A head's ``(target, mask[, known])`` cut to the records that supervise
    any of its terms; None for none."""
    mask = targets[1].detach().bool().cpu()
    rows = mask if mask.dim() == 1 else mask.any(-1)
    if not rows.any():
        return None
    cut = tuple(
        None if part is None else part.detach().cpu()[rows] for part in targets
    )
    if head == Head.CREATED_OBJECTS.value and (len(cut) < 3 or cut[2] is None):
        cut = (cut[0], cut[1], torch.ones_like(cut[0]))
    return cut


def constant_predictor_losses(
    sample: TargetSample, *, fields: tuple[FieldSpec, ...],
) -> HeadLosses:
    """The loss the best constant prediction reaches on ``sample``, by head
    and term, in nats per record over all its records.

    Scored through the heads' own loss terms over the entities and records
    they supervise, so each floor carries the loss's masking. A term whose
    supervised targets never vary reports exactly zero: it has no floor. A
    term the sample never supervises is absent, but for the per-entity gate.
    ``fields`` are the per-entity fields to cover.
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


def _degenerate(target: torch.Tensor | None) -> bool:
    """Whether a term's supervised targets never vary, whole rows compared.

    Such a term has no deviance to explain, and the clamp leaves a near-zero
    floor any head would read as enormously worse than constant.
    """
    return target is None or target.numel() == 0 or bool((target == target[0]).all())


def _as_sample(batches: Sequence[EntityTargetBatch] | TargetSample) -> TargetSample:
    return batches if isinstance(batches, TargetSample) else TargetSample.of(batches)


def constant_predictor_outputs(
    batches: Sequence[EntityTargetBatch] | TargetSample, *, fields: tuple[FieldSpec, ...],
) -> torch.Tensor:
    """The best constant prediction for the gate and each active field: one
    ``(PER_ENTITY_WIDTH,)`` vector, each slice the maximum-likelihood constant
    under that field's own loss."""
    sample = _as_sample(batches)
    vector = torch.zeros(PER_ENTITY_WIDTH)
    # The gate is one binary column over every entity it trains on.
    vector[GATE_INDEX] = FIELD_KINDS[FieldType.BINARY].best_constant(
        sample.gate, arity=1, cap=MAGNITUDE_CAP,
    )
    for spec in fields:
        target = sample.supervised(spec.name)
        if target is None:
            continue
        start, end = FIELD_SLICES[spec.name]
        vector[start:end] = FIELD_KINDS[spec.type].best_constant(
            target, arity=spec.arity, cap=MAGNITUDE_CAP,
        )
    return vector


def constant_predictor_dispersion(
    batches: Sequence[EntityTargetBatch] | TargetSample, *, fields: tuple[FieldSpec, ...],
) -> torch.Tensor:
    """The constant predictor's log-dispersions, in :data:`DISPERSED_FIELDS`
    order: each count's fitted around its constant mean (FR-063). A field the
    sample does not score keeps the upper bound, which nothing reads."""
    sample = _as_sample(batches)
    log_dispersion = torch.full((len(DISPERSED_FIELDS),), LOG_DISPERSION_MAX)
    for spec in fields:
        target = sample.supervised(spec.name)
        if target is None or not spec.type.has_magnitude:
            continue
        magnitude = FIELD_KINDS[spec.type].magnitude(target, cap=MAGNITUDE_CAP)
        log_dispersion[DISPERSION_INDEX[spec.name]] = fit_log_dispersion(
            magnitude, magnitude.mean(),
        )
    return log_dispersion


def _per_entity_losses(sample: TargetSample, fields: tuple[FieldSpec, ...]) -> dict[str, float]:
    vector = constant_predictor_outputs(sample, fields=fields)
    log_dispersion = constant_predictor_dispersion(sample, fields=fields)
    records = max(sample.records, 1)
    gate = sample.gate
    losses = {GATE_TERM: _weighted_sum(FIELD_KINDS[FieldType.BINARY].terms(
        vector[GATE_INDEX].expand(gate.shape[0], 1), gate, arity=1, cap=MAGNITUDE_CAP,
    ), sample.gate_weights) / records}
    for spec in fields:
        target = sample.supervised(spec.name)
        if target is None:
            continue
        if _degenerate(target):
            losses[spec.name] = 0.0
            continue
        start, end = FIELD_SLICES[spec.name]
        dispersion = (
            log_dispersion[DISPERSION_INDEX[spec.name]] if spec.type.has_magnitude else None
        )
        losses[spec.name] = _weighted_sum(FIELD_KINDS[spec.type].terms(
            vector[start:end].expand(target.shape[0], end - start), target,
            arity=spec.arity, cap=MAGNITUDE_CAP, log_dispersion=dispersion,
        ), sample.weights(spec.name)) / records
    return losses


def constant_predictor_floor(
    batches: Sequence[EntityTargetBatch] | TargetSample, *, fields: tuple[FieldSpec, ...],
) -> dict[str, float]:
    """The per-entity head's floor by term: :func:`constant_predictor_losses`
    over per-entity targets alone."""
    return constant_predictor_losses(_as_sample(batches), fields=fields)[PER_ENTITY]


# ── the per-record heads ──


@dataclass(frozen=True)
class RecordHeadConstant:
    """A per-record head's best constant over a sample: its outputs and
    log-dispersions, the fields the sample supervises and those of them whose
    targets never vary."""

    outputs: torch.Tensor
    log_dispersion: torch.Tensor
    supervised: frozenset[str]
    degenerate: frozenset[str]


def record_head_constant(batches: Sequence[tuple], layout: RecordHeadLayout) -> RecordHeadConstant:
    """The maximum-likelihood constant of a per-record head over ``batches``,
    each a ``(target, mask[, …])`` of the head's: each member's rate, class
    frequencies or mean over the records setting its field, and each count's
    dispersion fitted around that mean."""
    masks = [_field_mask(targets[1], layout) for targets in batches]
    outputs = torch.zeros(layout.width)
    log_dispersion = torch.full((len(layout.dispersed),), LOG_DISPERSION_MAX)
    degenerate: set[str] = set()
    supervised: set[str] = set()
    for position, spec in enumerate(layout.fields):
        first, last = layout.target_slices[spec.name]
        rows = [
            targets[0][mask[:, position]][:, first:last].float()
            for targets, mask in zip(batches, masks)
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
    return RecordHeadConstant(outputs, log_dispersion, frozenset(supervised),
                              frozenset(degenerate))


def _record_head_losses(
    sample: TargetSample, head: str, layout: RecordHeadLayout,
) -> dict[str, float]:
    batches = sample.head_batches(head)
    if not batches:
        return {}
    constant = record_head_constant(batches, layout)
    summed = torch.zeros(len(layout.fields), dtype=torch.float64)
    for targets in batches:
        # The head's own mask, a weight per record under the rule mask.
        summed += record_head_terms(
            constant.outputs.expand(int(targets[0].shape[0]), layout.width), targets[0],
            targets[1], layout, log_dispersion=constant.log_dispersion,
        ).double()
    records = max(sample.records, 1)
    return {
        name: 0.0 if name in constant.degenerate else float(value) / records
        for name, value in zip(layout.terms, summed.tolist())
        if name in constant.supervised
    }


def record_head_record_losses(
    target: torch.Tensor, mask: torch.Tensor, layout: RecordHeadLayout,
) -> dict[str, torch.Tensor]:
    """Each field's constant-prediction loss on each record, unweighted: the
    head's best constant over these records (:func:`record_head_constant`),
    scored on every record that sets the field and zero elsewhere. A field
    no record sets is absent; one whose targets never vary is all zero, as
    its floor is (research E4's ``ℓ_i``)."""
    flags = _field_mask(mask, layout)
    if not flags.any():
        return {}
    constant = record_head_constant([(target, flags)], layout)
    rows = record_head_row_terms(
        constant.outputs.expand(int(target.shape[0]), layout.width), target, flags, layout,
        log_dispersion=constant.log_dispersion,
    ).double()
    return {
        name: torch.zeros_like(rows[:, position]) if name in constant.degenerate
        else rows[:, position]
        for position, name in enumerate(layout.terms) if name in constant.supervised
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
        _created_rows(targets) for targets in sample.head_batches(Head.CREATED_OBJECTS.value)
    ]
    batches = [rows for rows in batches if rows is not None]
    if not batches:
        return {}
    target = torch.cat([rows[0] for rows in batches])
    known = torch.cat([rows[1] for rows in batches])
    outputs, log_dispersion = _created_constant(target, known)
    summed = torch.zeros(len(CREATED_TERMS), dtype=torch.float64)
    for wanted, said in batches:
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


def created_objects_record_losses(
    target: torch.Tensor, mask: torch.Tensor, known: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Each created-objects term's constant-prediction loss on each record,
    unweighted: the head's best constant over the records ``mask`` selects,
    scored on each of them and zero elsewhere, a term summed over the slots
    as the head's loss sums it. A term no record says is absent; one whose
    targets never vary is all zero, as its floor is (research E4's ``ℓ_i``)."""
    rows = _created_rows((target, mask, known))
    if rows is None:
        return {}
    wanted, said = rows
    outputs, log_dispersion = _created_constant(wanted, said)
    terms = created_objects_row_terms(
        outputs.expand(int(wanted.shape[0]), CREATED_OBJECTS_WIDTH), wanted,
        known=said, log_dispersion=log_dispersion,
    ).double()
    selected = mask.detach().bool().cpu().nonzero()[:, 0]
    slot_known = said[:, :-1].reshape(-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    slot_target = wanted[:, :-1].reshape(-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    out = {}
    for column, name in enumerate(CREATED_TERMS):
        if column < CREATED_SLOT_WIDTH:
            values = slot_target[..., column][slot_known[..., column] > 0]
        else:
            values = wanted[:, -1][said[:, -1] > 0]
        if values.numel() == 0:
            continue
        per_record = torch.zeros(int(target.shape[0]), dtype=torch.float64)
        if not _degenerate(values):
            per_record.index_copy_(0, selected, terms[:, column])
        out[name] = per_record
    return out


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
