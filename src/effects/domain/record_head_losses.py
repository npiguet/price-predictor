"""The per-record heads' losses: the verdict, global and created-objects heads
(FR-078, FR-079).

A per-record head's loss by field or term is summed over the records that set
it; its loss is those terms scaled by their loss denominators (FR-062) and
summed, per record of the batch. A floating mask carries the rule mask's weight
on each field (FR-061).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from effects.domain.effect_head_layout import (
    CREATED_MAGNITUDES,
    CREATED_OBJECT_SLOTS,
    CREATED_SLOT_WIDTH,
    CREATED_TERMS,
    GLOBAL_LAYOUT,
    VERDICT_LAYOUT,
    CreatedSlot,
    RecordHeadLayout,
)
from effects.domain.field_kinds import FIELD_KINDS, FieldScoring, counted, negative_binomial
from effects.domain.input_layout import MAGNITUDE_CAP
from effects.domain.loss_rows import Rows, per_row, scaled


@dataclass(frozen=True, slots=True)
class HeadScoring:
    """How a head's terms are read: its learned ``log_dispersion`` and, for
    a loss, each term's reciprocal loss denominator ``scale`` (FR-062); None
    leaves the terms unscaled."""

    log_dispersion: torch.Tensor | None = None
    scale: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class RecordHeadTargets:
    """A per-record head's targets: ``target`` ``(batch, layout.target_width)``,
    a categorical's column a class index, and ``mask`` ``(batch, fields)``, or
    ``(batch,)`` for a record that sets every field; a floating ``mask`` is
    each record's weight on the field under the rule mask (FR-061), zero
    where the record does not set it."""

    target: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class CreatedTargets:
    """The created-objects head's targets: ``mask`` selects the records that
    supervise the head, ``known`` ``(records, width)`` the columns each
    record says, a column it leaves unsaid (a token's controller) training
    nothing."""

    target: torch.Tensor
    mask: torch.Tensor | None = None
    known: torch.Tensor | None = None


def created_objects_terms(
    prediction: torch.Tensor, targets: CreatedTargets, scoring: HeadScoring,
) -> torch.Tensor:
    """The created-objects head's loss by term, summed over records, in
    :data:`CREATED_TERMS` order: each slot column over every slot, then the
    overflow flag.

    The K slots are in canonical order (FR-078), so no assignment problem
    arises. ``targets.mask`` selects the records that supervise the head;
    ``targets.known`` the columns each record says. ``scoring.log_dispersion``
    is the head's, in :data:`CREATED_DISPERSED` order, shared by the slots.
    """
    if scoring.log_dispersion is None:
        raise ValueError(
            "the created-objects head scores its counts by a negative binomial "
            "and needs its log-dispersions"
        )
    target, known = targets.target, targets.known
    if targets.mask is not None:
        rows = Rows.of(targets.mask, prediction.device)
        if rows.empty:
            return prediction.new_zeros(len(CREATED_TERMS), dtype=torch.float32)
        prediction = rows.select(prediction)
        target = rows.select(target)
        if known is not None:
            known = rows.select(known)
    elementwise, overflow = _created_elementwise(prediction, target, known, scoring.log_dispersion)
    columns = elementwise.sum((0, 1))
    return torch.cat([columns, overflow.sum().reshape(1)])


def created_objects_row_terms(
    prediction: torch.Tensor, target: torch.Tensor, known: torch.Tensor | None = None,
    *, log_dispersion: torch.Tensor,
) -> torch.Tensor:
    """:func:`created_objects_terms` before the sum over records: ``(records,
    len(CREATED_TERMS))``, each slot column summed over the slots, every
    record scored (the deviance sample's per-record losses, research E4)."""
    elementwise, overflow = _created_elementwise(prediction, target, known, log_dispersion)
    return torch.cat([elementwise.sum(1), overflow.reshape(-1, 1)], dim=-1)


def _created_elementwise(
    prediction: torch.Tensor, target: torch.Tensor, known: torch.Tensor | None,
    log_dispersion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Each slot column's loss ``(records, slots, slot width)`` and the
    overflow flag's ``(records,)``, each at the weight ``known`` gives it."""
    prediction = prediction.float()
    target = target.to(prediction.device).float()
    weights = (
        torch.ones_like(target) if known is None
        else known.to(prediction.device, non_blocking=True).to(target.dtype)
    )
    # Every slot at once, as ``(records, slots, slot width)``.
    slots = (-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    predicted = prediction[:, :-1].reshape(slots)
    wanted = target[:, :-1].reshape(slots)
    weight = weights[:, :-1].reshape(slots)
    flags = slice(CreatedSlot.PRESENT, CreatedSlot.COUNT)
    binaries = slice(CreatedSlot.OTHER_CONTROLLER, CREATED_SLOT_WIDTH)
    elementwise = torch.cat([
        functional.binary_cross_entropy_with_logits(
            predicted[..., flags], wanted[..., flags], reduction="none",
        ),
        negative_binomial(
            predicted[..., CREATED_MAGNITUDES],
            counted(wanted[..., CREATED_MAGNITUDES], MAGNITUDE_CAP),
            log_dispersion,
        ),
        functional.binary_cross_entropy_with_logits(
            predicted[..., binaries], wanted[..., binaries], reduction="none",
        ),
    ], dim=-1)
    overflow = functional.binary_cross_entropy_with_logits(
        prediction[:, -1], target[:, -1], reduction="none",
    ) * weights[:, -1]
    return elementwise * weight, overflow


def created_objects_loss(
    prediction: torch.Tensor, targets: CreatedTargets, scoring: HeadScoring,
) -> torch.Tensor:
    """:func:`created_objects_terms`, scaled by ``scoring.scale`` (each term's
    reciprocal denominator in :data:`CREATED_TERMS` order) and summed, per
    record of the batch."""
    return _per_record(prediction, created_objects_terms(prediction, targets, scoring), scoring)


def _per_record(
    prediction: torch.Tensor, terms: torch.Tensor, scoring: HeadScoring,
) -> torch.Tensor:
    """A head's ``terms`` scaled and summed, per record of ``prediction``'s batch."""
    return per_row(scaled(terms, scoring.scale), int(prediction.shape[0]))


@dataclass(frozen=True, slots=True)
class _RecordSource:
    """A per-record head's prediction and targets, on the device the loss runs
    on, beside its layout and log-dispersions: what one field's rows are cut
    from and scored with."""

    prediction: torch.Tensor
    target: torch.Tensor
    layout: RecordHeadLayout
    log_dispersion: torch.Tensor | None

    def view(self, rows: Rows, spec) -> tuple[torch.Tensor, torch.Tensor]:
        """``spec``'s prediction ``(rows, members, width)`` and target at ``rows``."""
        start, end = self.layout.slices[spec.name]
        first, last = self.layout.target_slices[spec.name]
        predicted = rows.select(self.prediction)[:, start:end].float().reshape(
            -1, spec.count, spec.type.width(spec.arity),
        )
        return predicted, rows.select(self.target)[:, first:last]

    def scoring(self, spec) -> FieldScoring:
        """How ``spec`` is scored: its own log-dispersion when a negative
        binomial scores it."""
        dispersion = None
        if spec.type.has_magnitude:
            if self.log_dispersion is None:
                raise ValueError(
                    f"{spec.name} is scored by a negative binomial and needs "
                    "its head's log-dispersions"
                )
            dispersion = self.log_dispersion[self.layout.dispersion_index[spec.name]]
        return FieldScoring(spec.arity, MAGNITUDE_CAP, dispersion)

    def field_sum(self, rows: Rows, spec, weight: torch.Tensor | None) -> torch.Tensor:
        """``spec``'s loss summed over ``rows``, each at its ``weight`` (a
        host column) where one is given."""
        predicted, wanted = self.view(rows, spec)
        scoring = self.scoring(spec)
        kind = FIELD_KINDS[spec.type]
        if weight is None:
            return kind.loss(predicted, wanted, scoring)
        terms = kind.terms(predicted, wanted, scoring)
        weight = rows.select(weight).to(self.prediction.device, non_blocking=True)
        return (terms * weight.reshape(-1, *([1] * (terms.dim() - 1)))).sum()

    def field_terms(self, rows: Rows, spec) -> torch.Tensor:
        """``spec``'s loss terms on each of ``rows``."""
        predicted, wanted = self.view(rows, spec)
        return FIELD_KINDS[spec.type].terms(predicted, wanted, self.scoring(spec))


def _set_flags(mask: torch.Tensor, layout: RecordHeadLayout) -> torch.Tensor:
    """``(records, fields)`` bools on the host: any non-zero ``mask`` entry sets
    the field; a ``(records,)`` mask sets every field of its record."""
    flags = mask.detach().cpu() != 0
    if flags.dim() == 1:
        flags = flags.unsqueeze(-1).expand(-1, len(layout.fields))
    return flags


def _field_weights(mask: torch.Tensor, layout: RecordHeadLayout) -> torch.Tensor | None:
    """``(records, fields)`` floats on the host: a floating ``mask`` as each
    record's weight on each field; None for a mask that only sets."""
    if not mask.is_floating_point():
        return None
    weights = mask.detach().cpu().float()
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1).expand(-1, len(layout.fields))
    return weights


def record_head_terms(
    prediction: torch.Tensor, targets: RecordHeadTargets, layout: RecordHeadLayout,
    scoring: HeadScoring,
) -> torch.Tensor:
    """A per-record head's loss by field, summed over the records that set
    it, in ``layout.terms`` order; a field's members are scored together
    (FR-079), under the field's one log-dispersion (``scoring.log_dispersion``,
    in ``layout.dispersed`` order).

    A floating ``targets.mask`` is each record's weight on the field under
    the rule mask (FR-061), zero where the record does not set it.
    """
    weights = _field_weights(targets.mask, layout)
    flags = _set_flags(targets.mask, layout)
    sums = [prediction.new_zeros((), dtype=torch.float32)] * len(layout.fields)
    if not flags.any():
        return torch.stack(sums)
    source = _RecordSource(
        prediction, targets.target.to(prediction.device, non_blocking=True), layout,
        scoring.log_dispersion,
    )
    for position, spec in enumerate(layout.fields):
        rows = Rows.of(flags[:, position], prediction.device)
        if rows.empty:
            continue
        sums[position] = source.field_sum(
            rows, spec, None if weights is None else weights[:, position],
        )
    return torch.stack(sums)


def record_head_row_terms(
    prediction: torch.Tensor, targets: RecordHeadTargets, layout: RecordHeadLayout,
    scoring: HeadScoring,
) -> torch.Tensor:
    """:func:`record_head_terms` before the sum over records, unweighted:
    ``(records, fields)``, each field's loss on each record that sets it and
    zero on the others (the deviance sample's per-record losses, research
    E4). ``targets.mask`` is ``(records, fields)`` or ``(records,)``; any
    non-zero entry sets the field."""
    flags = _set_flags(targets.mask, layout)
    out = torch.zeros(int(prediction.shape[0]), len(layout.fields), dtype=torch.float32,
                      device=prediction.device)
    source = _RecordSource(
        prediction, targets.target.to(prediction.device, non_blocking=True), layout,
        scoring.log_dispersion,
    )
    for position, spec in enumerate(layout.fields):
        rows = Rows.of(flags[:, position], prediction.device)
        if rows.empty:
            continue
        terms = source.field_terms(rows, spec)
        out[rows.device, position] = terms.reshape(terms.shape[0], -1).sum(-1).float()
    return out


def record_head_loss(
    prediction: torch.Tensor, targets: RecordHeadTargets, layout: RecordHeadLayout,
    scoring: HeadScoring,
) -> torch.Tensor:
    """A per-record head's loss: :func:`record_head_terms`, scaled by
    ``scoring.scale`` (each field's reciprocal denominator in ``layout.terms``
    order) and summed, per record of the batch."""
    return _per_record(prediction, record_head_terms(prediction, targets, layout, scoring), scoring)


def verdict_loss(
    prediction: torch.Tensor, targets: RecordHeadTargets, scoring: HeadScoring,
) -> torch.Tensor:
    """The verdict head's loss at ``[ACT]``: each record kind supervises only
    its own fields (:data:`VERDICT_LAYOUT`)."""
    return record_head_loss(prediction, targets, VERDICT_LAYOUT, scoring)


def global_loss(
    prediction: torch.Tensor, targets: RecordHeadTargets, scoring: HeadScoring,
) -> torch.Tensor:
    """The global head's loss at ``[GLOBAL]`` (:data:`GLOBAL_LAYOUT`)."""
    return record_head_loss(prediction, targets, GLOBAL_LAYOUT, scoring)
