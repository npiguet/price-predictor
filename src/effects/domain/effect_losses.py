"""The effect head's losses (FR-080, FR-081, FR-082, FR-085).

Each field's loss follows what the field is, as :mod:`effect_head_layout`
declares it, and each head's term is normalized per record so a large board or
a batch with few supervising records weighs no differently from any other.
"""

from __future__ import annotations

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
    EntityFieldTargets,
    FieldSpec,
    FieldType,
)
from effects.domain.state_snapshot import COLORS
from effects.domain.value_targets import VALUE_TARGETS, ValueKind


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
    """A count target as the Poisson loss scores it: from zero to the cap.

    A count below zero has no likelihood, and a negative target makes the
    Poisson NLL unbounded below. ``damage_taken`` nets healing against damage,
    so a heal on an undamaged creature reaches here as one; the creature
    ended with no damage marked, which is what zero says.
    """
    return target.clamp(min=0.0, max=MAGNITUDE_CAP)


def _poisson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Poisson NLL on a log-rate prediction, summed.

    ``full=True`` adds the Stirling approximation of ``log(k!)``, which is
    constant in the prediction and so changes no gradient — but it moves the
    minimum of every term to about zero. Without it a term's floor is
    ``k - k·ln k``, and a batch holding one 88-card draw reads as -306 at a
    perfect prediction, which is what made the first run's training loss
    track batch composition rather than the model.
    """
    return functional.poisson_nll_loss(
        prediction, target, log_input=True, full=True, reduction="sum",
    )


def _signed_delta_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Direction (categorical) plus magnitude (count), summed.

    Splitting them is what makes "gains 2 life" and "loses 2 life" different
    predictions rather than a regression that can hedge between them.
    """
    direction_logits, log_magnitude = prediction[..., :3], prediction[..., 3]
    direction = torch.sign(target).long() + 1  # -1,0,1 -> 0,1,2
    loss = functional.cross_entropy(
        direction_logits.reshape(-1, 3), direction.reshape(-1), reduction="sum",
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
            return _poisson(prediction.squeeze(-1), counted(target.float()))
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
    gate_target: torch.Tensor,
    field_targets: EntityFieldTargets,
    entity_mask: torch.Tensor,
    *,
    fields: tuple[FieldSpec, ...],
    report_parts: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The per-entity head's loss, normalized per record over entity count.

    Args:
        outputs: ``(batch, entities, PER_ENTITY_WIDTH)``.
        gate_target: ``(batch, entities)``, 1 where the entity was affected.
        field_targets: per-field ``(batch, entities, …)`` targets and where
            each is present.
        entity_mask: ``(batch, entities)``, 1 for a real entity.
        fields: the fields active at this step, from :func:`active_fields`.
        report_parts: read each term back as a Python float for logging. Off by
            default: every read is a device synchronization, and with fifteen
            active fields that is fifteen stalls on a path the training loop
            walks once per batch and whose result it discards. Each reported
            term is divided by the same record count the total is, so the
            breakdown a log line prints sums to the loss printed beside it.

    The gate trains on every entity — players included, since the head is mapped
    over ``[PLAYER]`` outputs too. A conditional field trains only where the
    gate's *target* fires and the field is present — where the record said
    something about it — and not where the gate's prediction fires: supervising
    on the prediction would let a head that gates everything off escape the
    field loss entirely.

    Normalizing per record rather than per entity keeps a twelve-permanent board
    from outweighing a two-permanent one; each record is one observation of an
    ability, whatever the board size.
    """
    parts: dict[str, float] = {}
    device = outputs.device
    # The rows are picked on the host, where the masks were built, and gathered
    # by index: a device mask tested or indexed by stalls the step until the GPU
    # drains, which is what kept the CPU from preparing the next batch while
    # the GPU worked. A mask already on the device is read back once here.
    real_mask = entity_mask.detach().bool().cpu()
    gate_host = gate_target.detach().cpu()
    real = real_mask.flatten().nonzero().squeeze(1)
    hit_mask = real_mask & gate_host.bool()
    flat_outputs = outputs.reshape(-1, outputs.shape[-1])
    gate = functional.binary_cross_entropy_with_logits(
        flat_outputs.index_select(0, real.to(device, non_blocking=True))[:, GATE_INDEX],
        gate_host.flatten().index_select(0, real).float().to(device, non_blocking=True),
        reduction="sum",
    )
    if report_parts:
        parts["gate"] = float(gate.detach())
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
            parts[spec.name] = float(loss.detach())
        total = total + loss

    records = max(int(outputs.shape[0]), 1)
    # The parts were summed over the batch like the total; dividing them by the
    # same record count is what makes the printed breakdown add up to the
    # printed loss rather than to `records` times it.
    if report_parts:
        parts = {name: value / records for name, value in parts.items()}
    return total / records, parts


def created_objects_loss(
    prediction: torch.Tensor, target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary presence/flags plus count regression over the K slots.

    Slots are in canonical order (FR-078), so slot *i* means the same group on
    both sides and the head is not asked to solve an assignment problem. Count,
    power and toughness are counts (FR-080); presence, the scripted flag and the
    type and keyword flags are bits.

    ``mask`` selects the records that supervise the head — effect halves and
    rewrites with an outgoing event. The sum is normalized over every record of
    the batch, like the per-entity loss, so a batch with few supervising
    records does not weigh them up.
    """
    records = max(int(prediction.shape[0]), 1)
    if mask is not None:
        # Decided on the host, where the mask was built: no device read.
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
    return total / records


def verdict_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Verdict bits and the trigger bit binary; cost paid Poisson per colour.

    ``mask`` is per record ``(batch,)`` — every output of a selected record is
    scored — or per output ``(batch, VERDICT_WIDTH)``, which is what training
    passes: a ``decision`` record supervises the three verdict bits, a cost half
    the cost paid and a ``trigger`` record the fired bit, and none of them says
    anything about the others' columns. Normalized over every record of the
    batch, like the per-entity loss.
    """
    records = max(int(prediction.shape[0]), 1)
    # Tested on the host, where the mask was built: no device read.
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
    return total / records


def value_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """The value head's loss over the batch's encoded texts (FR-058, FR-058b).

    One row per unique text. A count is Poisson, a signed amount the
    direction-and-magnitude split, a cost flag binary — the same losses the
    per-entity fields of those types use. Every amount also scores its
    at-least bits by binary cross-entropy, averaged over the bits. A masked
    target contributes nothing, and the sum is normalized over the texts.
    """
    rows = max(int(prediction.shape[0]), 1)
    # The mask is decided on the host, where the targets were built: testing a
    # device mask and indexing by it would stall the step once per target.
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
            # Averaged over the steps, so an amount's forty bits weigh what
            # one binary target does: summed, they would outweigh the rest of
            # the head's loss many times over and set the encoder's gradient
            # alone.
            start, end = AT_LEAST_SLICES[name]
            total = total + functional.binary_cross_entropy_with_logits(
                prediction.index_select(0, selected)[:, start:end],
                at_least_targets(wanted.float()), reduction="sum",
            ) / AT_LEAST_STEPS
    return total / rows


def mlm_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the masked positions, which are the rows given.

    Training-only: it teaches the encoder the language of card text, which the
    effect objective alone would only reach through whichever words happened to
    change an outcome.
    """
    return functional.cross_entropy(logits, targets)


def api_loss(
    type_logits: torch.Tensor | None,
    type_targets: torch.Tensor | None,
    key_logits: torch.Tensor | None,
    key_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Script-API classification from ``e``: the API type and its parameter keys.

    It teaches ``e`` the script structure the prose surface does not spell
    out; on the script surface it reinforces what the encoder reads directly.
    """
    total = None
    if type_logits is not None and type_targets is not None:
        # A line with no API type (a replacement, say) carries -1 and is
        # skipped rather than scored against an invented class. The test runs
        # on the host, where the targets were built, so it costs no sync.
        known = (type_targets.detach().cpu() >= 0).nonzero().squeeze(1)
        if known.numel():
            known = known.to(type_logits.device, non_blocking=True)
            total = functional.cross_entropy(
                type_logits.index_select(0, known),
                type_targets.to(type_logits.device).index_select(0, known),
            )
    if key_logits is not None and key_targets is not None:
        keys = functional.binary_cross_entropy_with_logits(
            key_logits, key_targets.float(),
        )
        total = keys if total is None else total + keys
    if total is None:
        if type_targets is None and key_targets is None:
            raise ValueError("api_loss needs at least one of its two targets")
        logits = type_logits if type_logits is not None else key_logits
        return logits.new_zeros(())
    return total
