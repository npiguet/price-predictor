"""The effect head's losses (FR-080, FR-081, FR-082, FR-085).

One reduction for every head: a row's loss is summed over its targets, then
averaged over every row of the batch, scored or not (:func:`per_row`). A row is
a record, or an encoded text for the value, script-API and MLM heads.

Every mask is tested and indexed on the host, where it was built, and rows are
gathered by index: a device mask would stall the step until the GPU drains.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from effects.domain.effect_head_input import MAGNITUDE_CAP
from effects.domain.effect_head_layout import (
    AT_LEAST_SLICES,
    AT_LEAST_STEPS,
    CREATED_OBJECT_SLOTS,
    CREATED_SLOT_WIDTH,
    FIELD_SLICES,
    GATE_INDEX,
    VALUE_SLICES,
    VERDICT_BITS,
    CreatedSlot,
    FieldSpec,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import FIELD_KINDS, capped, poisson
from effects.domain.state_snapshot import COLORS
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
    """The magnitude a count or signed-delta field's Poisson term trains on,
    at the per-entity head's cap, so anything scoring that term reads the
    target the same way."""
    return FIELD_KINDS[spec.type].magnitude(target, cap=MAGNITUDE_CAP)


def field_loss(
    spec: FieldSpec, prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """The loss for one field, chosen by what the field is (FR-080), every
    magnitude scored at :data:`MAGNITUDE_CAP`."""
    return FIELD_KINDS[spec.type].loss(
        prediction, target, arity=spec.arity, cap=MAGNITUDE_CAP,
    )


def per_entity_loss(
    outputs: torch.Tensor,
    targets: EntityTargetBatch,
    *,
    fields: tuple[FieldSpec, ...],
    report_parts: bool = False,
) -> tuple[torch.Tensor, LossParts]:
    """The per-entity head's loss, per record, and its terms when
    ``report_parts`` (scaled alike, so they sum to the loss).

    Args:
        outputs: ``(batch, entities, PER_ENTITY_WIDTH)``, already gathered
            column for column with ``targets``.
        fields: the fields active at this step.

    The gate trains on every :attr:`~EntityTargetBatch.real` entity, a field
    wherever :meth:`~EntityTargetBatch.scored` puts it.
    """
    names: list[str] = []
    terms: list[torch.Tensor] = []
    flat_outputs = outputs.reshape(-1, outputs.shape[-1])
    total = _gate_term(flat_outputs, targets)
    if report_parts:
        names.append("gate")
        terms.append(total.detach())

    for spec in fields:
        loss = _field_term(spec, flat_outputs, targets)
        if loss is None:
            continue
        if report_parts:
            names.append(spec.name)
            terms.append(loss.detach())
        total = total + loss

    records = int(outputs.shape[0])
    parts = LossParts()
    if report_parts:
        parts = LossParts(
            tuple(names),
            torch.stack([term.double() for term in terms]) / max(records, 1),
        )
    return per_row(total, records), parts


def _gate_term(flat_outputs: torch.Tensor, targets: EntityTargetBatch) -> torch.Tensor:
    """The affected gate's BCE over every real entity, summed."""
    device = flat_outputs.device
    real = _Rows.of(targets.real, device)
    affected = real.select(targets.gate.detach().cpu().flatten()).float()
    return functional.binary_cross_entropy_with_logits(
        real.select(flat_outputs)[:, GATE_INDEX],
        affected.to(device, non_blocking=True),
        reduction="sum",
    )


def _field_term(
    spec: FieldSpec, flat_outputs: torch.Tensor, targets: EntityTargetBatch,
) -> torch.Tensor | None:
    """One field's loss over the entities it is scored at, summed; None where
    the batch carries no target for it or scores it nowhere."""
    target = targets.fields.values.get(spec.name)
    if target is None:
        return None
    device = flat_outputs.device
    rows = _Rows.of(targets.scored(spec.name), device)
    if rows.empty:
        return None
    start, end = FIELD_SLICES[spec.name]
    flat_target = target.reshape(-1, *target.shape[2:])
    return field_loss(
        spec, rows.select(flat_outputs)[:, start:end],
        rows.select(flat_target).to(device, non_blocking=True),
    )


def created_objects_loss(
    prediction: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bits plus counts over the K slots, which are in canonical order
    (FR-078) so no assignment problem arises. ``mask`` selects the records
    that supervise the head."""
    records = int(prediction.shape[0])
    if mask is not None:
        rows = _Rows.of(mask, prediction.device)
        if rows.empty:
            return prediction.new_zeros(())
        prediction = rows.select(prediction)
        target = rows.select(target).to(prediction.device)
    # Every slot at once, as ``(records, slots, slot width)``.
    slots = (-1, CREATED_OBJECT_SLOTS, CREATED_SLOT_WIDTH)
    predicted = prediction[:, :-1].reshape(slots)
    wanted = target[:, :-1].reshape(slots)
    flags = slice(CreatedSlot.PRESENT, CreatedSlot.COUNT)
    magnitudes = slice(CreatedSlot.COUNT, CreatedSlot.TYPES)
    rest = slice(CreatedSlot.TYPES, CREATED_SLOT_WIDTH)
    total = functional.binary_cross_entropy_with_logits(
        predicted[..., flags], wanted[..., flags], reduction="sum",
    )
    total = total + poisson(
        predicted[..., magnitudes], capped(wanted[..., magnitudes], MAGNITUDE_CAP),
    )
    total = total + functional.binary_cross_entropy_with_logits(
        predicted[..., rest], wanted[..., rest], reduction="sum",
    )
    total = total + functional.binary_cross_entropy_with_logits(
        prediction[:, -1], target[:, -1], reduction="sum",
    )
    return per_row(total, records)


def verdict_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Verdict bits and the trigger bit binary; cost paid Poisson per colour.

    ``mask`` is per record ``(batch,)`` or per output ``(batch,
    VERDICT_WIDTH)``, as training passes it: each record kind supervises only
    its own columns.
    """
    records = int(prediction.shape[0])
    mask = mask.detach().bool().cpu()
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1).expand(-1, prediction.shape[-1])
    if not mask.any():
        return prediction.new_zeros(())
    bits = len(VERDICT_BITS)
    cost = slice(bits, bits + len(COLORS))
    weights = mask.to(prediction.dtype).to(prediction.device, non_blocking=True)
    binary = functional.binary_cross_entropy_with_logits(
        prediction, target, reduction="none",
    )
    counts = functional.poisson_nll_loss(
        prediction[:, cost], target[:, cost], log_input=True, full=True,
        reduction="none",
    )
    total = (binary[:, :bits] * weights[:, :bits]).sum()
    total = total + (counts * weights[:, cost]).sum()
    total = total + (binary[:, -1] * weights[:, -1]).sum()
    return per_row(total, records)


def value_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """The value head's loss, one row per unique text (FR-058, FR-058b): the
    per-entity fields' losses by type, plus each amount's at-least bits."""
    rows = int(prediction.shape[0])
    mask = mask.detach().bool().cpu()
    target = target.to(prediction.device)
    total = prediction.new_zeros(())
    for index, (name, field_type) in enumerate(VALUE_TARGETS):
        selected = _Rows.of(mask[:, index], prediction.device)
        if selected.empty:
            continue
        predicted = selected.select(prediction)
        start, end = VALUE_SLICES[name]
        wanted = selected.select(target)[:, index]
        # Uncapped: a stated amount is the text's own, which no game compounds.
        total = total + FIELD_KINDS[field_type].loss(
            predicted[:, start:end], wanted, arity=1, cap=None,
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
