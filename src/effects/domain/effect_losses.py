"""The effect head's losses (FR-080, FR-081, FR-082, FR-085).

One reduction for every head: a row's loss is summed over its targets, then
averaged over every row of the batch, scored or not (:func:`per_row`). A row is
a record, or an encoded text for the value, script-API and MLM heads.

Every mask is tested and indexed on the host, where it was built, and rows are
gathered by index: a device mask would stall the step until the GPU drains.

A shipped head's loss takes ``scale``, each term's reciprocal loss denominator
(FR-062), and sums its terms scaled; its reported parts stay in nats.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from effects.domain.effect_head_input import MAGNITUDE_CAP
from effects.domain.effect_head_layout import (
    AT_LEAST_SLICES,
    AT_LEAST_STEPS,
    CREATED_MAGNITUDES,
    CREATED_OBJECT_SLOTS,
    CREATED_SLOT_WIDTH,
    CREATED_TERMS,
    DISPERSION_INDEX,
    FIELD_SLICES,
    GATE_INDEX,
    GATE_TERM,
    GLOBAL_LAYOUT,
    PER_ENTITY_TERMS,
    VALUE_DISPERSED,
    VALUE_SLICES,
    VERDICT_LAYOUT,
    CreatedSlot,
    FieldSpec,
    RecordHeadLayout,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import FIELD_KINDS, counted, negative_binomial
from effects.domain.field_types import FieldType
from effects.domain.value_targets import VALUE_TARGETS


@dataclass(frozen=True, slots=True)
class LossParts:
    """The per-entity loss by term, still on the device: names beside one
    float64 tensor, read back in one synchronization."""

    names: tuple[str, ...] = ()
    values: torch.Tensor | None = None

    def __bool__(self) -> bool:
        return bool(self.names)

    def as_dict(self) -> dict[str, float]:
        if self.values is None:
            return {}
        return dict(zip(self.names, self.values.tolist()))


@dataclass(frozen=True, slots=True)
class _Rows:
    """The rows a host mask selects: tested and indexed on the host, where the
    mask was built, and copied to the device once for gathering there."""

    host: torch.Tensor
    device: torch.Tensor

    @classmethod
    def of(cls, mask: torch.Tensor, device: torch.device) -> _Rows:
        host = mask.detach().bool().cpu().flatten().nonzero().squeeze(1)
        return cls(host, host.to(device, non_blocking=True))

    @property
    def empty(self) -> bool:
        return self.host.numel() == 0

    def select(self, tensor: torch.Tensor) -> torch.Tensor:
        """``tensor``'s selected rows, gathered wherever ``tensor`` lives."""
        return tensor.index_select(0, self.host if tensor.device.type == "cpu" else self.device)


def per_row(total: torch.Tensor, rows: int) -> torch.Tensor:
    return total / max(int(rows), 1)


def at_least_targets(amounts: torch.Tensor) -> torch.Tensor:
    """``(n,)`` amounts as ``(n, AT_LEAST_STEPS)`` bits: column k-1 is ``|amount| >= k``."""
    steps = torch.arange(
        1, AT_LEAST_STEPS + 1, device=amounts.device, dtype=amounts.dtype,
    )
    return (amounts.abs().unsqueeze(-1) >= steps).to(amounts.dtype)


def magnitude_target(spec: FieldSpec, target: torch.Tensor) -> torch.Tensor:
    """The magnitude a count or signed-delta field's count term trains on,
    at the per-entity head's cap, so anything scoring that term reads the
    target the same way."""
    return FIELD_KINDS[spec.type].magnitude(target, cap=MAGNITUDE_CAP)


def field_loss(
    spec: FieldSpec, prediction: torch.Tensor, target: torch.Tensor,
    *, log_dispersion: torch.Tensor | None = None,
) -> torch.Tensor:
    """The loss for one field, chosen by what the field is (FR-080), every
    magnitude scored at :data:`MAGNITUDE_CAP`; ``log_dispersion`` is the
    field's own, which a count or signed delta needs."""
    return FIELD_KINDS[spec.type].loss(
        prediction, target, arity=spec.arity, cap=MAGNITUDE_CAP,
        log_dispersion=log_dispersion,
    )


def _scaled(sums: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """Per-term sums divided by their loss denominators (FR-062): ``scale``
    holds each term's reciprocal, in the same order; None leaves them as they
    are."""
    return sums.sum() if scale is None else (sums * scale).sum()


def per_entity_loss(
    outputs: torch.Tensor,
    targets: EntityTargetBatch,
    *,
    fields: tuple[FieldSpec, ...],
    log_dispersion: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
    report_parts: bool = False,
) -> tuple[torch.Tensor, LossParts]:
    """The per-entity head's loss, per record, and its terms when
    ``report_parts``.

    Args:
        outputs: ``(batch, entities, PER_ENTITY_WIDTH)``, already gathered
            column for column with ``targets``.
        fields: the fields active at this step.
        log_dispersion: the head's log-dispersions in
            :data:`DISPERSED_FIELDS` order, which a count or signed-delta
            field needs.
        scale: each term's reciprocal loss denominator in
            :data:`PER_ENTITY_TERMS` order (FR-062), on the outputs' device;
            None scores every term unscaled.

    The gate trains wherever :attr:`~EntityTargetBatch.gate_scored` puts it,
    a field wherever :meth:`~EntityTargetBatch.scored` does. The parts are
    each term's loss in nats per record, before any scaling, so they compare
    with a constant predictor's; the loss is their scaled sum.
    """
    flat_outputs = outputs.reshape(-1, outputs.shape[-1])
    gate = _gate_term(flat_outputs, targets)
    names, sums = _field_sums(flat_outputs, targets, fields, log_dispersion)
    terms = torch.cat([gate.reshape(1), sums])
    term_scale = None
    if scale is not None:
        term_scale = scale.index_select(0, _term_index((GATE_TERM, *names), scale.device))
    total = _scaled(terms, term_scale)

    records = int(outputs.shape[0])
    parts = LossParts()
    if report_parts:
        parts = LossParts((GATE_TERM, *names), terms.detach().double() / max(records, 1))
    return per_row(total, records), parts


_PER_ENTITY_TERM_INDEX: dict[str, int] = {name: i for i, name in enumerate(PER_ENTITY_TERMS)}


def _term_index(names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor([_PER_ENTITY_TERM_INDEX[name] for name in names]).to(
        device, non_blocking=True,
    )


def _gate_term(flat_outputs: torch.Tensor, targets: EntityTargetBatch) -> torch.Tensor:
    """The affected gate's BCE over every real entity of a record that trains
    it, summed (a playability record trains none, root spec § 11.4)."""
    device = flat_outputs.device
    scored = _Rows.of(targets.gate_scored, device)
    if scored.empty:
        return flat_outputs.new_zeros((), dtype=torch.float32)
    affected = scored.select(targets.gate.detach().cpu().flatten()).float()
    return functional.binary_cross_entropy_with_logits(
        scored.select(flat_outputs)[:, GATE_INDEX].float(),
        affected.to(device, non_blocking=True),
        reduction="sum",
    )


def _same_kind(fields) -> list[list[FieldSpec]]:
    """``fields`` grouped by type and width, each group in field order: one
    group's terms are computed in one call."""
    groups: dict[tuple[FieldType, int], list[FieldSpec]] = {}
    for spec in fields:
        groups.setdefault((spec.type, spec.arity), []).append(spec)
    return list(groups.values())


def _field_sums(
    flat_outputs: torch.Tensor, targets: EntityTargetBatch,
    fields: tuple[FieldSpec, ...], log_dispersion: torch.Tensor | None,
) -> tuple[tuple[str, ...], torch.Tensor]:
    """Each field's loss summed over the entities it is scored at, as one
    vector, beside the names of the fields it holds, in ``fields`` order: a
    field the batch carries no target for, or scores nowhere, is left out.

    Fields of one type and width are scored together, over every entity any
    of them is scored at, each weighted 1 where it is scored and 0 elsewhere:
    the sums the field-by-field loop takes, at one call per group.
    """
    named: list[str] = []
    sums: list[torch.Tensor] = []
    carried = [spec for spec in fields if spec.name in targets.fields.values]
    for group in _same_kind(carried):
        found = _group_sums(flat_outputs, targets, group, log_dispersion)
        if found is not None:
            named.extend(found[0])
            sums.append(found[1])
    if not sums:
        return (), flat_outputs.new_zeros(0, dtype=torch.float32)
    order = {spec.name: position for position, spec in enumerate(fields)}
    permutation = sorted(range(len(named)), key=lambda i: order[named[i]])
    stacked = torch.cat(sums)
    if permutation != list(range(len(named))):
        stacked = stacked.index_select(
            0, torch.tensor(permutation).to(stacked.device, non_blocking=True),
        )
    return tuple(named[i] for i in permutation), stacked


def _group_sums(
    flat_outputs: torch.Tensor, targets: EntityTargetBatch,
    group: list[FieldSpec], log_dispersion: torch.Tensor | None,
) -> tuple[list[str], torch.Tensor] | None:
    """One same-kind group's per-field sums; None where it scores nothing."""
    device = flat_outputs.device
    masks = torch.stack([targets.scored(spec.name).reshape(-1) for spec in group], dim=1)
    scored = masks.any(0)
    if not scored.any():
        return None
    if not scored.all():
        keep = scored.nonzero().squeeze(1)
        group = [group[int(i)] for i in keep]
        masks = masks.index_select(1, keep)
    rows = _Rows.of(masks.any(1), device)
    weights = rows.select(masks).to(device, torch.float32, non_blocking=True)

    first = group[0]
    columns = torch.cat([
        torch.arange(*FIELD_SLICES[spec.name]) for spec in group
    ]).to(device, non_blocking=True)
    predicted = rows.select(flat_outputs).index_select(1, columns).reshape(
        -1, len(group), first.width,
    )
    member = (first.arity,) if first.type.per_member else ()
    target = torch.stack([
        targets.fields.values[spec.name].reshape(masks.shape[0], *member)
        for spec in group
    ], dim=1)
    target = rows.select(target).to(device, non_blocking=True)
    dispersion = None
    if first.type.has_magnitude:
        if log_dispersion is None:
            raise ValueError(
                f"{first.name} is scored by a negative binomial and needs the "
                "per-entity head's log-dispersions"
            )
        index = torch.tensor([DISPERSION_INDEX[spec.name] for spec in group])
        dispersion = log_dispersion.index_select(
            0, index.to(log_dispersion.device, non_blocking=True),
        )
    terms = FIELD_KINDS[first.type].terms(
        predicted.float(), target, arity=first.arity, cap=MAGNITUDE_CAP,
        log_dispersion=dispersion,
    )
    return [spec.name for spec in group], (terms * weights).sum(0)


def created_objects_terms(
    prediction: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None, known: torch.Tensor | None = None,
    *, log_dispersion: torch.Tensor | None = None,
) -> torch.Tensor:
    """The created-objects head's loss by term, summed over records, in
    :data:`CREATED_TERMS` order: each slot column over every slot, then the
    overflow flag.

    The K slots are in canonical order (FR-078), so no assignment problem
    arises. ``mask`` selects the records that supervise the head; ``known``
    ``(records, width)`` the columns each record says, a column it leaves
    unsaid (a token's controller) training nothing. ``log_dispersion`` is
    the head's, in :data:`CREATED_DISPERSED` order, shared by the slots.
    """
    if log_dispersion is None:
        raise ValueError(
            "the created-objects head scores its counts by a negative binomial "
            "and needs its log-dispersions"
        )
    if mask is not None:
        rows = _Rows.of(mask, prediction.device)
        if rows.empty:
            return prediction.new_zeros(len(CREATED_TERMS), dtype=torch.float32)
        prediction = rows.select(prediction)
        target = rows.select(target)
        if known is not None:
            known = rows.select(known)
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
    columns = (elementwise * weight).sum((0, 1))
    overflow = (functional.binary_cross_entropy_with_logits(
        prediction[:, -1], target[:, -1], reduction="none",
    ) * weights[:, -1]).sum()
    return torch.cat([columns, overflow.reshape(1)])


def created_objects_loss(
    prediction: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None, known: torch.Tensor | None = None,
    *, log_dispersion: torch.Tensor | None = None, scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """:func:`created_objects_terms`, scaled by ``scale`` (each term's
    reciprocal denominator in :data:`CREATED_TERMS` order) and summed, per
    record of the batch."""
    terms = created_objects_terms(
        prediction, target, mask, known, log_dispersion=log_dispersion,
    )
    return per_row(_scaled(terms, scale), int(prediction.shape[0]))


def record_head_terms(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    layout: RecordHeadLayout, *, log_dispersion: torch.Tensor | None = None,
) -> torch.Tensor:
    """A per-record head's loss by field, summed over the records that set
    it, in ``layout.terms`` order; a field's members are scored together
    (FR-079), under the field's one log-dispersion.

    ``target`` is ``(batch, layout.target_width)``, a categorical's column a
    class index; ``mask`` ``(batch, fields)``, or ``(batch,)`` for a record
    that sets every field; ``log_dispersion`` in ``layout.dispersed`` order.
    """
    mask = mask.detach().bool().cpu()
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1).expand(-1, len(layout.fields))
    sums = [prediction.new_zeros((), dtype=torch.float32)] * len(layout.fields)
    if not mask.any():
        return torch.stack(sums)
    target = target.to(prediction.device, non_blocking=True)
    for position, spec in enumerate(layout.fields):
        rows = _Rows.of(mask[:, position], prediction.device)
        if rows.empty:
            continue
        start, end = layout.slices[spec.name]
        first, last = layout.target_slices[spec.name]
        predicted = rows.select(prediction)[:, start:end].float().reshape(
            -1, spec.count, spec.type.width(spec.arity),
        )
        wanted = rows.select(target)[:, first:last]
        dispersion = None
        if spec.type.has_magnitude:
            if log_dispersion is None:
                raise ValueError(
                    f"{spec.name} is scored by a negative binomial and needs "
                    "its head's log-dispersions"
                )
            dispersion = log_dispersion[layout.dispersion_index[spec.name]]
        sums[position] = FIELD_KINDS[spec.type].loss(
            predicted, wanted, arity=spec.arity, cap=MAGNITUDE_CAP,
            log_dispersion=dispersion,
        )
    return torch.stack(sums)


def record_head_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    layout: RecordHeadLayout, *, log_dispersion: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """A per-record head's loss: :func:`record_head_terms`, scaled by
    ``scale`` (each field's reciprocal denominator in ``layout.terms``
    order) and summed, per record of the batch."""
    terms = record_head_terms(
        prediction, target, mask, layout, log_dispersion=log_dispersion,
    )
    return per_row(_scaled(terms, scale), int(prediction.shape[0]))


def verdict_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, log_dispersion: torch.Tensor | None = None, scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """The verdict head's loss at ``[ACT]``: each record kind supervises only
    its own fields (:data:`VERDICT_LAYOUT`)."""
    return record_head_loss(
        prediction, target, mask, VERDICT_LAYOUT,
        log_dispersion=log_dispersion, scale=scale,
    )


def global_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, log_dispersion: torch.Tensor | None = None, scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """The global head's loss at ``[GLOBAL]`` (:data:`GLOBAL_LAYOUT`)."""
    return record_head_loss(
        prediction, target, mask, GLOBAL_LAYOUT,
        log_dispersion=log_dispersion, scale=scale,
    )


_VALUE_DISPERSION_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(VALUE_DISPERSED)
}


def value_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
    *, log_dispersion: torch.Tensor | None = None,
) -> torch.Tensor:
    """The value head's loss, one row per unique text (FR-058, FR-058b): the
    per-entity fields' losses by type, plus each amount's at-least bits;
    ``log_dispersion`` in :data:`VALUE_DISPERSED` order."""
    rows = int(prediction.shape[0])
    mask = mask.detach().bool().cpu()
    target = target.to(prediction.device)
    total = prediction.new_zeros((), dtype=torch.float32)
    for index, (name, field_type) in enumerate(VALUE_TARGETS):
        selected = _Rows.of(mask[:, index], prediction.device)
        if selected.empty:
            continue
        predicted = selected.select(prediction).float()
        start, end = VALUE_SLICES[name]
        wanted = selected.select(target)[:, index]
        dispersion = None
        if field_type.has_magnitude:
            if log_dispersion is None:
                raise ValueError(
                    f"the value head's {name} is scored by a negative binomial "
                    "and needs the head's log-dispersions"
                )
            dispersion = log_dispersion[_VALUE_DISPERSION_INDEX[name]]
        # Uncapped: a stated amount is the text's own, which no game compounds.
        total = total + FIELD_KINDS[field_type].loss(
            predicted[:, start:end], wanted, arity=1, cap=None,
            log_dispersion=dispersion,
        )
        if name in AT_LEAST_SLICES:
            # Averaged over the steps, so forty bits weigh what one binary
            # target does rather than dominating the encoder's gradient.
            start, end = AT_LEAST_SLICES[name]
            total = total + functional.binary_cross_entropy_with_logits(
                predicted[:, start:end],
                at_least_targets(wanted.float()), reduction="sum",
            ) / AT_LEAST_STEPS
    return per_row(total, rows)


def mlm_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, rows: int,
) -> torch.Tensor:
    """Cross-entropy at the masked positions, per text of the pass."""
    return per_row(functional.cross_entropy(logits, targets, reduction="sum"), rows)


def api_loss(
    type_logits: torch.Tensor,
    type_targets: torch.Tensor,
    key_logits: torch.Tensor | None,
    key_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Script-API classification from ``e``, one row per text: the API type's
    cross-entropy plus every parameter key's binary cross-entropy. The keys
    are None for a run whose sidecars name no parameter key, which builds no
    key head."""
    rows = int(type_logits.shape[0])
    total = None
    # A line with no API type carries -1 and is skipped.
    known = _Rows.of(type_targets.detach().cpu() >= 0, type_logits.device)
    if not known.empty:
        total = functional.cross_entropy(
            known.select(type_logits),
            known.select(type_targets).to(type_logits.device),
            reduction="sum",
        )
    if key_logits is not None and key_targets is not None:
        keys = functional.binary_cross_entropy_with_logits(
            key_logits, key_targets.float(), reduction="sum",
        )
        total = keys if total is None else total + keys
    if total is None:
        return type_logits.new_zeros(())
    return per_row(total, rows)
