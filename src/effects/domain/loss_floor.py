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
above the weighted optimum's loss, never below it. ``weighted`` fits it at
the targets' weights instead: the explained-deviance sample's, whose rows
carry their class weight (FR-059a), so a row of class weight k fits as k
repeated rows did.

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
from effects.domain.effect_losses import LossParts
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import (
    FIELD_KINDS,
    LOG_DISPERSION_MAX,
    FieldScoring,
    counted,
    fit_log_dispersion,
)
from effects.domain.input_layout import MAGNITUDE_CAP
from effects.domain.record_head_losses import (
    CreatedTargets,
    HeadScoring,
    RecordHeadTargets,
    created_objects_row_terms,
    created_objects_terms,
    record_head_row_terms,
    record_head_terms,
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
            self._add_field(entity, name, values)
        for head, targets in (heads or {}).items():
            compact = _supervising_rows(str(head), targets)
            if compact is not None:
                self._heads.setdefault(str(head), []).append(compact)

    def _add_field(self, entity: EntityTargetBatch, name: str, values: torch.Tensor) -> None:
        scored = entity.scored(name)
        selected = values.detach().cpu()[scored]
        if selected.numel():
            self._fields.setdefault(name, []).append(selected)
            self._field_weights.setdefault(name, []).append(
                _weights_at(entity.field_weights(name), scored),
            )

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
    if head == Head.CREATED_OBJECTS.value and _lacks_known(cut):
        cut = (cut[0], cut[1], torch.ones_like(cut[0]))
    return cut


def _lacks_known(cut: tuple) -> bool:
    """Whether a created-objects tuple carries no ``known`` part."""
    return len(cut) < 3 or cut[2] is None


def constant_predictor_losses(
    sample: TargetSample, *, fields: tuple[FieldSpec, ...], weighted: bool = False,
) -> HeadLosses:
    """The loss the best constant prediction reaches on ``sample``, by head
    and term, in nats per record over all its records.

    Scored through the heads' own loss terms over the entities and records
    they supervise, so each floor carries the loss's masking. A term whose
    supervised targets never vary reports exactly zero: it has no floor. A
    term the sample never supervises is absent, but for the per-entity gate.
    ``fields`` are the per-entity fields to cover; ``weighted`` fits each
    constant at the targets' weights.
    """
    losses: HeadLosses = {PER_ENTITY: _per_entity_losses(sample, fields, weighted)}
    for head, layout in RECORD_LAYOUTS.items():
        found = _record_head_losses(sample, head, layout, weighted)
        if found:
            losses[head] = found
    created = _created_objects_losses(sample, weighted)
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
    weighted: bool = False,
) -> torch.Tensor:
    """The best constant prediction for the gate and each active field: one
    ``(PER_ENTITY_WIDTH,)`` vector, each slice the maximum-likelihood constant
    under that field's own loss, at the targets' weights where ``weighted``."""
    sample = _as_sample(batches)
    vector = torch.zeros(PER_ENTITY_WIDTH)
    # The gate is one binary column over every entity it trains on.
    vector[GATE_INDEX] = FIELD_KINDS[FieldType.BINARY].best_constant(
        sample.gate, arity=1, cap=MAGNITUDE_CAP,
        weights=sample.gate_weights if weighted else None,
    )
    for spec in fields:
        target = sample.supervised(spec.name)
        if target is None:
            continue
        start, end = FIELD_SLICES[spec.name]
        vector[start:end] = FIELD_KINDS[spec.type].best_constant(
            target, arity=spec.arity, cap=MAGNITUDE_CAP,
            weights=sample.weights(spec.name) if weighted else None,
        )
    return vector


def constant_predictor_dispersion(
    batches: Sequence[EntityTargetBatch] | TargetSample, *, fields: tuple[FieldSpec, ...],
    weighted: bool = False,
) -> torch.Tensor:
    """The constant predictor's log-dispersions, in :data:`DISPERSED_FIELDS`
    order: each count's fitted around its constant mean (FR-063), at the
    targets' weights where ``weighted``. A field the sample does not score
    keeps the upper bound, which nothing reads."""
    sample = _as_sample(batches)
    log_dispersion = torch.full((len(DISPERSED_FIELDS),), LOG_DISPERSION_MAX)
    for spec in fields:
        target = sample.supervised(spec.name)
        if target is None or not spec.type.has_magnitude:
            continue
        magnitude = FIELD_KINDS[spec.type].magnitude(target, cap=MAGNITUDE_CAP)
        weights = sample.weights(spec.name) if weighted else None
        centre = magnitude.mean() if weights is None else weighted_mean(magnitude, weights)
        log_dispersion[DISPERSION_INDEX[spec.name]] = fit_log_dispersion(
            magnitude, centre, weights,
        )
    return log_dispersion


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The mean of ``values`` over its rows, each row at its weight (one a row)."""
    rows = weights.double().reshape(-1, *([1] * (values.dim() - 1)))
    return ((values.double() * rows).sum(0) / rows.sum(0).clamp(min=FLOOR_EPSILON)).float()


def _per_entity_losses(sample: TargetSample, fields: tuple[FieldSpec, ...],
                       weighted: bool = False) -> dict[str, float]:
    vector = constant_predictor_outputs(sample, fields=fields, weighted=weighted)
    log_dispersion = constant_predictor_dispersion(sample, fields=fields, weighted=weighted)
    records = max(sample.records, 1)
    gate = sample.gate
    losses = {GATE_TERM: _weighted_sum(FIELD_KINDS[FieldType.BINARY].terms(
        vector[GATE_INDEX].expand(gate.shape[0], 1), gate, FieldScoring(1, MAGNITUDE_CAP),
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
            FieldScoring(spec.arity, MAGNITUDE_CAP, dispersion),
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


def record_head_constant(batches: Sequence[tuple], layout: RecordHeadLayout, *,
                         weighted: bool = False) -> RecordHeadConstant:
    """The maximum-likelihood constant of a per-record head over ``batches``,
    each a ``(target, mask[, …])`` of the head's: each member's rate, class
    frequencies or mean over the records setting its field, and each count's
    dispersion fitted around that mean; where ``weighted``, each record at
    its weight on the field, the floating mask's value."""
    held = _HeadRows.of(batches, layout, weighted=weighted)
    outputs = torch.zeros(layout.width)
    log_dispersion = torch.full((len(layout.dispersed),), LOG_DISPERSION_MAX)
    degenerate: set[str] = set()
    supervised: set[str] = set()
    for position, spec in enumerate(layout.fields):
        wanted, weights = held.field(position)
        if wanted.shape[0] == 0:
            continue
        supervised.add(str(spec.name))
        if _degenerate(wanted):
            degenerate.add(str(spec.name))
        _fit_field(_FieldFit(outputs, log_dispersion, layout), spec, wanted, weights)
    return RecordHeadConstant(outputs, log_dispersion, frozenset(supervised),
                              frozenset(degenerate))


@dataclass(frozen=True)
class _HeadRows:
    """A per-record head's batches, each one's ``(records, fields)`` mask as
    bools and, where weighted, as weights."""

    batches: Sequence[tuple]
    layout: RecordHeadLayout
    masks: list[torch.Tensor]
    weights: list[torch.Tensor] | None

    @classmethod
    def of(cls, batches: Sequence[tuple], layout: RecordHeadLayout, *,
           weighted: bool) -> _HeadRows:
        masks = [_field_mask(targets[1], layout) for targets in batches]
        weights = [_mask_weights(targets[1], layout) for targets in batches] if weighted else None
        return cls(batches, layout, masks, weights)

    def field(self, position: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One field's targets over the records setting it, and their weights
        (None unweighted)."""
        first, last = self.layout.target_slices[self.layout.fields[position].name]
        rows = [targets[0][mask[:, position]][:, first:last].float()
                for targets, mask in zip(self.batches, self.masks)]
        wanted = torch.cat(rows) if rows else torch.zeros(0, last - first)
        if self.weights is None:
            return wanted, None
        return wanted, torch.cat([held[mask[:, position], position]
                                  for held, mask in zip(self.weights, self.masks)])


@dataclass(frozen=True)
class _FieldFit:
    """Where a per-record head's constant is written: its outputs and its
    log-dispersions, in its layout."""

    outputs: torch.Tensor
    log_dispersion: torch.Tensor
    layout: RecordHeadLayout


def _fit_field(fit: _FieldFit, spec: FieldSpec, wanted: torch.Tensor,
               weights: torch.Tensor | None) -> None:
    """One per-record field's best constant into ``fit.outputs`` and, for a
    count, its dispersion into ``fit.log_dispersion``."""
    kind = FIELD_KINDS[spec.type]
    member_width = spec.type.width(spec.arity)
    start, _ = fit.layout.slices[spec.name]
    for member in range(spec.count):
        offset = start + member * member_width
        fit.outputs[offset:offset + member_width] = kind.best_constant(
            wanted[:, member], arity=spec.arity, cap=MAGNITUDE_CAP, weights=weights,
        )
    if spec.type.has_magnitude:
        magnitude = kind.magnitude(wanted, cap=MAGNITUDE_CAP)
        centre = magnitude.mean(0) if weights is None else weighted_mean(magnitude, weights)
        fit.log_dispersion[fit.layout.dispersion_index[str(spec.name)]] = fit_log_dispersion(
            magnitude, centre, weights,
        )


def _mask_weights(mask: torch.Tensor, layout: RecordHeadLayout) -> torch.Tensor:
    """A per-record head's mask as ``(records, fields)`` weights: a floating
    mask's values, one where a boolean mask is set."""
    weights = mask.detach().cpu().float()
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1).expand(-1, len(layout.fields))
    return weights


def _record_head_losses(
    sample: TargetSample, head: str, layout: RecordHeadLayout, weighted: bool = False,
) -> dict[str, float]:
    batches = sample.head_batches(head)
    if not batches:
        return {}
    constant = record_head_constant(batches, layout, weighted=weighted)
    summed = torch.zeros(len(layout.fields), dtype=torch.float64)
    for targets in batches:
        # The head's own mask, a weight per record under the rule mask.
        summed += record_head_terms(
            constant.outputs.expand(int(targets[0].shape[0]), layout.width),
            RecordHeadTargets(targets[0], targets[1]), layout, HeadScoring(constant.log_dispersion),
        ).double()
    records = max(sample.records, 1)
    return {
        name: 0.0 if name in constant.degenerate else float(value) / records
        for name, value in zip(layout.terms, summed.tolist())
        if name in constant.supervised
    }


def record_head_record_losses(
    target: torch.Tensor, mask: torch.Tensor, layout: RecordHeadLayout,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Each field's constant-prediction loss on each record, unweighted: the
    head's best constant over these records (:func:`record_head_constant`),
    each at its weight where ``weights`` (one a record) is given, scored on
    every record that sets the field and zero elsewhere. A field no record
    sets is absent; one whose targets never vary is all zero, as its floor
    is (research E4's ``ℓ_i``)."""
    flags = _field_mask(mask, layout)
    if not flags.any():
        return {}
    fitted = flags if weights is None else flags.float() * weights.float().unsqueeze(-1)
    constant = record_head_constant([(target, fitted)], layout, weighted=weights is not None)
    rows = record_head_row_terms(
        constant.outputs.expand(int(target.shape[0]), layout.width),
        RecordHeadTargets(target, flags), layout, HeadScoring(constant.log_dispersion),
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


def _created_objects_losses(sample: TargetSample, weighted: bool = False) -> dict[str, float]:
    batches = _created_batches(sample)
    if not batches:
        return {}
    target = torch.cat([rows[0] for rows in batches])
    known = torch.cat([rows[1] for rows in batches])
    outputs, log_dispersion = _created_constant(target, known, weighted=weighted)
    summed = torch.zeros(len(CREATED_TERMS), dtype=torch.float64)
    for wanted, said in batches:
        summed += created_objects_terms(
            outputs.expand(int(wanted.shape[0]), CREATED_OBJECTS_WIDTH),
            CreatedTargets(wanted, known=said), HeadScoring(log_dispersion),
        ).double()
    records = max(sample.records, 1)
    losses = {}
    for column, name in enumerate(CREATED_TERMS):
        values = _column_values(target, known, column)
        if values.numel() == 0:
            continue
        losses[name] = 0.0 if _degenerate(values) else float(summed[column]) / records
    return losses


def _created_batches(sample: TargetSample) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Each created-objects batch's supervising ``(target, known)``, the
    batches supervising none left out."""
    batches = [
        _created_rows(targets) for targets in sample.head_batches(Head.CREATED_OBJECTS.value)
    ]
    return [rows for rows in batches if rows is not None]


def _column_values(target: torch.Tensor, known: torch.Tensor, column: int) -> torch.Tensor:
    """The targets one created-objects term is said at: a slot column over
    every slot of every record, or the overflow flag."""
    if column < CREATED_SLOT_WIDTH:
        slots = (-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
        said = known[:, :-1].reshape(slots)[..., column] > 0
        return target[:, :-1].reshape(slots)[..., column][said]
    return target[:, -1][known[:, -1] > 0]


def created_objects_record_losses(
    target: torch.Tensor, mask: torch.Tensor, known: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Each created-objects term's constant-prediction loss on each record,
    unweighted: the head's best constant over the records ``mask`` selects,
    each at its weight where ``weights`` (one a record) is given, scored on
    each of them and zero elsewhere, a term summed over the slots as the
    head's loss sums it. A term no record says is absent; one whose targets
    never vary is all zero, as its floor is (research E4's ``ℓ_i``)."""
    rows = _created_rows((target, mask, known))
    if rows is None:
        return {}
    wanted, said = rows
    outputs, log_dispersion = _created_record_constant(wanted, said, mask, weights)
    terms = created_objects_row_terms(
        outputs.expand(int(wanted.shape[0]), CREATED_OBJECTS_WIDTH), wanted,
        known=said, log_dispersion=log_dispersion,
    ).double()
    selected = mask.detach().bool().cpu().nonzero()[:, 0]
    out = {}
    for column, name in enumerate(CREATED_TERMS):
        values = _column_values(wanted, said, column)
        if values.numel() == 0:
            continue
        per_record = torch.zeros(int(target.shape[0]), dtype=torch.float64)
        if not _degenerate(values):
            per_record.index_copy_(0, selected, terms[:, column])
        out[name] = per_record
    return out


def _created_record_constant(
    wanted: torch.Tensor, said: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The created-objects head's best constant over the records ``mask``
    selects (``wanted`` and ``said`` are theirs), each at its weight where
    ``weights`` (one a record of ``mask``) is given."""
    if weights is None:
        return _created_constant(wanted, said)
    chosen = weights.float()[mask.detach().bool().cpu()].unsqueeze(-1)
    return _created_constant(wanted, said * chosen, weighted=True)


def _created_rows(targets: tuple) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The supervising records' ``(target, known)``, or None for none."""
    target, mask = targets[0], targets[1]
    known = targets[2] if len(targets) > 2 and targets[2] is not None else torch.ones_like(target)
    rows = mask.detach().bool().cpu()
    if not rows.any():
        return None
    return target[rows].float(), known[rows].float()


def _created_constant(
    target: torch.Tensor, known: torch.Tensor, *, weighted: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The created-objects head's best constant: each slot column's rate or
    mean over the records that say it, each at the weight ``known`` gives
    it, the overflow flag's rate, and a log-dispersion per magnitude column,
    shared by the slots; where ``weighted``, the overflow rate and the
    dispersions are fitted at ``known``'s weights too."""
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
    overflow = known[:, -1] > 0
    outputs = torch.cat([
        columns.reshape(-1),
        binary.best_constant(target[:, -1][overflow], arity=1, cap=None,
                             weights=known[:, -1][overflow] if weighted else None).reshape(1),
    ])
    return outputs, _created_dispersion(wanted, said, mean, weighted=weighted)


def _created_dispersion(wanted: torch.Tensor, said: torch.Tensor, mean: torch.Tensor, *,
                        weighted: bool) -> torch.Tensor:
    """A log-dispersion per created-objects magnitude column, shared by the
    slots, fitted around each slot's mean."""
    log_dispersion = torch.full((len(CREATED_DISPERSED),), LOG_DISPERSION_MAX)
    for index, column in enumerate(range(CREATED_MAGNITUDES.start, CREATED_MAGNITUDES.stop)):
        present = said[..., column] > 0
        if not present.any():
            continue
        values = wanted[..., column]
        means = mean[:, column].expand_as(values)
        log_dispersion[index] = fit_log_dispersion(
            values[present], means[present],
            said[..., column][present] if weighted else None)
    return log_dispersion


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
