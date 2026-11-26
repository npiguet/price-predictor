"""The per-entity effect head's loss (FR-080, FR-081, FR-082, FR-085).

A row's loss is summed over its targets, then averaged over every row of the
batch, scored or not (:func:`~effects.domain.loss_rows.per_row`): the rows are
the batch's records. The other heads' losses are in
:mod:`~effects.domain.record_head_losses` and
:mod:`~effects.domain.text_head_losses`.

Every mask is tested and indexed on the host, where it was built, and rows are
gathered by index: a device mask would stall the step until the GPU drains.

A shipped head's loss takes ``scale``, each term's reciprocal loss denominator
(FR-062), and sums its terms scaled; its reported parts stay in nats.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from effects.domain.effect_head_layout import (
    DISPERSION_INDEX,
    FIELD_SLICES,
    GATE_INDEX,
    GATE_TERM,
    PER_ENTITY_TERMS,
    FieldSpec,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import FIELD_KINDS, FieldScoring
from effects.domain.field_types import FieldType
from effects.domain.input_layout import MAGNITUDE_CAP
from effects.domain.loss_rows import Rows, per_row, scaled


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

    def joined(self, other: LossParts) -> LossParts:
        """These parts followed by ``other``'s, one tensor still on the device."""
        if other.values is None:
            return self
        if self.values is None:
            return other
        return LossParts((*self.names, *other.names),
                         torch.cat([self.values, other.values.to(self.values.device)]))


@dataclass(frozen=True, slots=True)
class EntityLossScoring:
    """How the per-entity head is scored: the ``fields`` active at the step,
    the head's log-dispersions in :data:`DISPERSED_FIELDS` order (which a
    count or signed-delta field needs), each term's reciprocal loss
    denominator in :data:`PER_ENTITY_TERMS` order (FR-062; None scores every
    term unscaled), and whether the terms are reported."""

    fields: tuple[FieldSpec, ...]
    log_dispersion: torch.Tensor | None = None
    scale: torch.Tensor | None = None
    report_parts: bool = False


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
        prediction, target, FieldScoring(spec.arity, MAGNITUDE_CAP, log_dispersion),
    )


def per_entity_loss(
    outputs: torch.Tensor, targets: EntityTargetBatch, scoring: EntityLossScoring,
) -> tuple[torch.Tensor, LossParts]:
    """The per-entity head's loss, per record, and its terms when
    ``scoring.report_parts``.

    Args:
        outputs: ``(batch, entities, PER_ENTITY_WIDTH)``, already gathered
            column for column with ``targets``.
        scoring: the fields active at this step, the head's log-dispersions
            and each term's reciprocal loss denominator (:class:`EntityLossScoring`).

    The gate trains wherever :attr:`~EntityTargetBatch.gate_scored` puts it,
    a field wherever :meth:`~EntityTargetBatch.scored` does. The parts are
    each term's loss in nats per record, before any scaling, so they compare
    with a constant predictor's; the loss is their scaled sum.
    """
    flat_outputs = outputs.reshape(-1, outputs.shape[-1])
    gate = _gate_term(flat_outputs, targets)
    names, sums = _field_sums(flat_outputs, targets, scoring.fields, scoring.log_dispersion)
    terms = torch.cat([gate.reshape(1), sums])
    total = scaled(terms, _term_scale(scoring.scale, (GATE_TERM, *names)))

    records = int(outputs.shape[0])
    parts = LossParts()
    if scoring.report_parts:
        parts = LossParts((GATE_TERM, *names), terms.detach().double() / max(records, 1))
    return per_row(total, records), parts


def _term_scale(scale: torch.Tensor | None, names: tuple[str, ...]) -> torch.Tensor | None:
    """``scale`` (every per-entity term's) cut to the terms ``names``."""
    if scale is None:
        return None
    return scale.index_select(0, _term_index(names, scale.device))


_PER_ENTITY_TERM_INDEX: dict[str, int] = {name: i for i, name in enumerate(PER_ENTITY_TERMS)}


def _term_index(names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor([_PER_ENTITY_TERM_INDEX[name] for name in names]).to(
        device, non_blocking=True,
    )


def _gate_term(flat_outputs: torch.Tensor, targets: EntityTargetBatch) -> torch.Tensor:
    """The affected gate's BCE over every real entity of a record that trains
    it, summed (a playability record trains none, root spec § 11.4), each
    entity at its rule-mask weight (FR-061): under the hard mask an added
    copy trains the gate on its drawn rule's entities alone."""
    device = flat_outputs.device
    scored = Rows.of(targets.gate_scored, device)
    if scored.empty:
        return flat_outputs.new_zeros((), dtype=torch.float32)
    affected = scored.select(targets.gate.detach().cpu().flatten()).float()
    weights = targets.gate_weights()
    if weights is not None:
        weights = scored.select(weights.flatten()).to(device, non_blocking=True)
    return functional.binary_cross_entropy_with_logits(
        scored.select(flat_outputs)[:, GATE_INDEX].float(),
        affected.to(device, non_blocking=True),
        weight=weights,
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
    the sums the field-by-field loop takes, at one call per group. A field
    the rule mask weighs (FR-061) takes its weight where it is scored.
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
    kept = _scored_members(targets, group)
    if kept is None:
        return None
    group, masks = kept
    device = flat_outputs.device
    rows = Rows.of(masks.any(1), device)
    weights = rows.select(_weighed(masks, targets, group)).to(
        device, torch.float32, non_blocking=True,
    )
    first = group[0]
    predicted = _group_predicted(flat_outputs, rows, group)
    target = _group_target(targets, rows, group, masks.shape[0]).to(device, non_blocking=True)
    dispersion = _group_dispersion(group, log_dispersion)
    terms = FIELD_KINDS[first.type].terms(
        predicted.float(), target, FieldScoring(first.arity, MAGNITUDE_CAP, dispersion),
    )
    return [spec.name for spec in group], (terms * weights).sum(0)


def _scored_members(
    targets: EntityTargetBatch, group: list[FieldSpec],
) -> tuple[list[FieldSpec], torch.Tensor] | None:
    """The members of ``group`` scored anywhere, beside their ``(entities,
    members)`` masks; None where none is."""
    masks = torch.stack([targets.scored(spec.name).reshape(-1) for spec in group], dim=1)
    scored = masks.any(0)
    if not scored.any():
        return None
    if scored.all():
        return group, masks
    keep = scored.nonzero().squeeze(1)
    return [group[int(i)] for i in keep], masks.index_select(1, keep)


def _weighed(
    masks: torch.Tensor, targets: EntityTargetBatch, group: list[FieldSpec],
) -> torch.Tensor:
    """``masks`` as each member's weight where scored: the rule mask's (FR-061)
    for a field it weighs, 1 for the others; the masks as they are where none
    is weighed."""
    weighed = [targets.field_weights(spec.name) for spec in group]
    if all(weight is None for weight in weighed):
        return masks
    return masks.float() * torch.stack([
        torch.ones(masks.shape[0]) if weight is None else weight.reshape(-1)
        for weight in weighed
    ], dim=1)


def _group_predicted(
    flat_outputs: torch.Tensor, rows: Rows, group: list[FieldSpec],
) -> torch.Tensor:
    """``(rows, members, width)``: the group's columns at the selected rows."""
    columns = torch.cat([
        torch.arange(*FIELD_SLICES[spec.name]) for spec in group
    ]).to(flat_outputs.device, non_blocking=True)
    return rows.select(flat_outputs).index_select(1, columns).reshape(
        -1, len(group), group[0].width,
    )


def _group_target(
    targets: EntityTargetBatch, rows: Rows, group: list[FieldSpec], entities: int,
) -> torch.Tensor:
    """``(rows, members[, arity])``: the group's targets at the selected rows."""
    first = group[0]
    member = (first.arity,) if first.type.per_member else ()
    target = torch.stack([
        targets.fields.values[spec.name].reshape(entities, *member)
        for spec in group
    ], dim=1)
    return rows.select(target)


def _group_dispersion(
    group: list[FieldSpec], log_dispersion: torch.Tensor | None,
) -> torch.Tensor | None:
    """The members' log-dispersions, None for a group no negative binomial scores."""
    first = group[0]
    if not first.type.has_magnitude:
        return None
    if log_dispersion is None:
        raise ValueError(
            f"{first.name} is scored by a negative binomial and needs the "
            "per-entity head's log-dispersions"
        )
    index = torch.tensor([DISPERSION_INDEX[spec.name] for spec in group])
    return log_dispersion.index_select(0, index.to(log_dispersion.device, non_blocking=True))
