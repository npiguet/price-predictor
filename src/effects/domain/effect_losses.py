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
    DIRECTIONS,
    FIELD_SLICES,
    GATE_INDEX,
    SIGNED_DIRECTIONS,
    SIGNED_MAGNITUDE,
    VALUE_SLICES,
    VERDICT_BITS,
    CreatedSlot,
    FieldSpec,
    FieldType,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.state_snapshot import COLORS
from effects.domain.value_targets import VALUE_TARGETS, ValueKind


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


def per_row(total: torch.Tensor, rows: int) -> torch.Tensor:
    return total / max(int(rows), 1)


def at_least_targets(amounts: torch.Tensor) -> torch.Tensor:
    """``(n,)`` amounts as ``(n, AT_LEAST_STEPS)`` bits: column k-1 is ``|amount| >= k``."""
    steps = torch.arange(
        1, AT_LEAST_STEPS + 1, device=amounts.device, dtype=amounts.dtype,
    )
    return (amounts.abs().unsqueeze(-1) >= steps).to(amounts.dtype)


def capped(target: torch.Tensor) -> torch.Tensor:
    """``target`` with every magnitude above :data:`MAGNITUDE_CAP` at the cap, sign kept."""
    return target.clamp(min=-MAGNITUDE_CAP, max=MAGNITUDE_CAP)


def counted(target: torch.Tensor) -> torch.Tensor:
    """A count target clamped to ``[0, cap]``: a negative target (a heal
    netted into ``damage_taken``) makes the Poisson NLL unbounded below."""
    return target.clamp(min=0.0, max=MAGNITUDE_CAP)


def magnitude_target(spec: FieldSpec, target: torch.Tensor) -> torch.Tensor:
    """The magnitude a count or signed-delta field's Poisson term trains on,
    so anything scoring that term reads the target the same way."""
    if spec.type is FieldType.COUNT:
        return counted(target.float())
    return capped(target.float()).abs()


def _poisson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Poisson NLL on a log-rate prediction, summed. ``full=True`` changes no
    gradient but puts every term's minimum near zero, so the loss does not
    track batch composition."""
    return functional.poisson_nll_loss(
        prediction, target, log_input=True, full=True, reduction="sum",
    )


def _signed_delta_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Direction (categorical) plus magnitude (count), summed, so a gain and
    a loss are different predictions rather than a hedge."""
    direction_logits = prediction[..., SIGNED_DIRECTIONS]
    log_magnitude = prediction[..., SIGNED_MAGNITUDE]
    direction = torch.sign(target).long() + 1  # -1,0,1 -> 0,1,2
    loss = functional.cross_entropy(
        direction_logits.reshape(-1, len(DIRECTIONS)), direction.reshape(-1),
        reduction="sum",
    )
    return loss + _poisson(log_magnitude, target.abs())


def field_loss(
    spec: FieldSpec, prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """The loss for one field, chosen by what the field is (FR-080)."""
    match spec.type:
        case FieldType.BINARY:
            return functional.binary_cross_entropy_with_logits(
                prediction.squeeze(-1), target.float(), reduction="sum",
            )
        case FieldType.MULTI_BINARY:
            return functional.binary_cross_entropy_with_logits(
                prediction, target.float(), reduction="sum",
            )
        case FieldType.COUNT:
            return _poisson(prediction.squeeze(-1), magnitude_target(spec, target))
        case FieldType.SIGNED_DELTA:
            return _signed_delta_loss(prediction, capped(target.float()))
        case FieldType.CATEGORICAL:
            return functional.cross_entropy(
                prediction.reshape(-1, spec.arity), target.reshape(-1).long(),
                reduction="sum",
            )
    raise ValueError(f"unhandled field type {spec.type}")


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

    The gate trains on every real entity, players included. A field trains
    where the gate's *target* fires and the field is present, never where the
    prediction fires: a head that gates everything off would escape the field
    loss.
    """
    names: list[str] = []
    terms: list[torch.Tensor] = []
    device = outputs.device
    field_targets = targets.fields
    real_mask = targets.mask.detach().bool().cpu()
    gate_host = targets.gate.detach().cpu()
    real = real_mask.flatten().nonzero().squeeze(1)
    hit_mask = real_mask & gate_host.bool()
    flat_outputs = outputs.reshape(-1, outputs.shape[-1])
    gate = functional.binary_cross_entropy_with_logits(
        flat_outputs.index_select(0, real.to(device, non_blocking=True))[:, GATE_INDEX],
        gate_host.flatten().index_select(0, real).float().to(device, non_blocking=True),
        reduction="sum",
    )
    if report_parts:
        names.append("gate")
        terms.append(gate.detach())
    total = gate

    for spec in fields:
        target = field_targets.values.get(spec.name)
        if target is None:
            continue
        present = field_targets.present[spec.name].detach().bool().cpu()
        rows = (hit_mask & present).flatten().nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        rows_device = rows.to(device, non_blocking=True)
        start, end = FIELD_SLICES[spec.name]
        flat_target = target.reshape(-1, *target.shape[2:])
        loss = field_loss(
            spec, flat_outputs.index_select(0, rows_device)[:, start:end],
            flat_target.index_select(
                0, rows if flat_target.device.type == "cpu" else rows_device,
            ).to(device, non_blocking=True),
        )
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


def created_objects_loss(
    prediction: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bits plus counts over the K slots, which are in canonical order
    (FR-078) so no assignment problem arises. ``mask`` selects the records
    that supervise the head."""
    records = int(prediction.shape[0])
    if mask is not None:
        rows = mask.detach().bool().cpu().nonzero().squeeze(1)
        if rows.numel() == 0:
            return prediction.new_zeros(())
        rows = rows.to(prediction.device, non_blocking=True)
        prediction = prediction.index_select(0, rows)
        target = target.to(prediction.device).index_select(0, rows)
    total = prediction.new_zeros(())
    for slot in range(CREATED_OBJECT_SLOTS):
        base = slot * CREATED_SLOT_WIDTH
        flags = slice(base + CreatedSlot.PRESENT, base + CreatedSlot.COUNT)
        total = total + functional.binary_cross_entropy_with_logits(
            prediction[:, flags], target[:, flags], reduction="sum",
        )
        magnitudes = slice(base + CreatedSlot.COUNT, base + CreatedSlot.TYPES)
        total = total + _poisson(
            prediction[:, magnitudes], capped(target[:, magnitudes]),
        )
        rest = slice(base + CreatedSlot.TYPES, base + CREATED_SLOT_WIDTH)
        total = total + functional.binary_cross_entropy_with_logits(
            prediction[:, rest], target[:, rest], reduction="sum",
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
    per-entity fields' losses by kind, plus each amount's at-least bits."""
    rows = int(prediction.shape[0])
    mask = mask.detach().bool().cpu()
    total = prediction.new_zeros(())
    for index, (name, kind) in enumerate(VALUE_TARGETS):
        selected = mask[:, index].nonzero().squeeze(1)
        if selected.numel() == 0:
            continue
        selected = selected.to(prediction.device, non_blocking=True)
        start, end = VALUE_SLICES[name]
        output = prediction.index_select(0, selected)[:, start:end]
        wanted = target.to(prediction.device).index_select(0, selected)[:, index]
        match kind:
            case ValueKind.COUNT:
                total = total + _poisson(output.squeeze(-1), wanted)
            case ValueKind.SIGNED:
                total = total + _signed_delta_loss(output, wanted)
            case ValueKind.BINARY:
                total = total + functional.binary_cross_entropy_with_logits(
                    output.squeeze(-1), wanted, reduction="sum",
                )
        if name in AT_LEAST_SLICES:
            # Averaged over the steps, so forty bits weigh what one binary
            # target does rather than dominating the encoder's gradient.
            start, end = AT_LEAST_SLICES[name]
            total = total + functional.binary_cross_entropy_with_logits(
                prediction.index_select(0, selected)[:, start:end],
                at_least_targets(wanted.float()), reduction="sum",
            ) / AT_LEAST_STEPS
    return per_row(total, rows)


def mlm_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, rows: int,
) -> torch.Tensor:
    """Cross-entropy at the masked positions, per text of the pass."""
    return per_row(functional.cross_entropy(logits, targets, reduction="sum"), rows)


def api_loss(
    type_logits: torch.Tensor | None,
    type_targets: torch.Tensor | None,
    key_logits: torch.Tensor | None,
    key_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Script-API classification from ``e``, one row per text: the API type's
    cross-entropy plus every parameter key's binary cross-entropy."""
    given = type_logits if type_logits is not None else key_logits
    rows = int(given.shape[0]) if given is not None else 0
    total = None
    if type_logits is not None and type_targets is not None:
        # A line with no API type carries -1 and is skipped.
        known = (type_targets.detach().cpu() >= 0).nonzero().squeeze(1)
        if known.numel():
            known = known.to(type_logits.device, non_blocking=True)
            total = functional.cross_entropy(
                type_logits.index_select(0, known),
                type_targets.to(type_logits.device).index_select(0, known),
                reduction="sum",
            )
    if key_logits is not None and key_targets is not None:
        keys = functional.binary_cross_entropy_with_logits(
            key_logits, key_targets.float(), reduction="sum",
        )
        total = keys if total is None else total + keys
    if total is None:
        if type_targets is None and key_targets is None:
            raise ValueError("api_loss needs at least one of its two targets")
        return given.new_zeros(())
    return per_row(total, rows)
