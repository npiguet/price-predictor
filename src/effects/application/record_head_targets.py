"""The shipped record heads' targets, on the host: the verdict, global and
created-objects heads' rows over a step's records, each field weighed by the
rule mask (FR-060a, FR-061). Their derivation from one record is
``effects.domain.aux_head_targets``; these functions batch and weigh it."""

from __future__ import annotations

import torch

from effects.application.mlm_masking import MaskedTokens
from effects.domain.aux_head_targets import (
    created_objects_targets,
    global_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.aux_heads import Head
from effects.domain.effect_head_layout import (
    CREATED_OBJECTS_WIDTH,
    GLOBAL_LAYOUT,
    VERDICT_LAYOUT,
    RecordHeadLayout,
)
from effects.domain.head_targets import HeadTargetParts
from effects.domain.record_head_losses import CreatedTargets, RecordHeadTargets
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT, CopyWeights


class HeadTargets(dict[Head, HeadTargetParts | MaskedTokens]):
    """The targets of every loss term beside the per-entity one, on the host,
    for only the heads computed for this batch."""

    def pin(self) -> None:
        for head, targets in self.items():
            if isinstance(targets, MaskedTokens):
                targets.pin()
            else:
                self[head] = targets.pinned()


def shipped_head_targets(records, weights: list[CopyWeights]) -> HeadTargets:
    """The shipped record heads' targets (FR-060a), each record's fields
    weighed by its rule mask (FR-061)."""
    targets = HeadTargets()
    for head, layout, derive in (
        (Head.VERDICT, VERDICT_LAYOUT, verdict_targets),
        (Head.GLOBAL, GLOBAL_LAYOUT, global_targets),
    ):
        rows = record_head_rows(records, weights, layout, derive)
        if rows is not None:
            targets[head] = rows
    made = created_objects_rows(records, weights)
    if made is not None:
        targets[Head.CREATED_OBJECTS] = made
    return targets


def record_head_rows(
    records, weights: list[CopyWeights], layout: RecordHeadLayout, derive,
) -> RecordHeadTargets | None:
    """A per-record head's targets over ``records``, or None where
    no record sets any of its fields. Under the rule mask (FR-061) a field a
    copy weighs zero is unset, and the mask holds each field's weight where
    any is fractional."""
    rows = [derive(record) for record in records]
    if all(row is None for row in rows):
        return None
    values, mask = _filled(rows, layout)
    if not all(copy.unmasked for copy in weights):
        mask = _weighed_mask(mask, weights, layout)
    return RecordHeadTargets(values, mask)


def _filled(rows: list, layout: RecordHeadLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """The rows' target values and masks as tensors, a blank row where a
    record sets none of the head's fields."""
    values, masks = zip(*(row if row is not None else layout.blank() for row in rows))
    return torch.tensor(list(values)), torch.tensor(list(masks))


def _weighed_mask(mask: torch.Tensor, weights: list[CopyWeights], layout: RecordHeadLayout):
    """``mask`` scaled by each copy's weight on each field; a boolean mask
    again where every weight is zero or one."""
    weighed = _field_weights(weights, layout)
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1).expand(-1, len(layout.fields))
    mask = mask.float() * weighed
    if bool(((mask == 0) | (mask == 1)).all()):
        mask = mask.bool()
    return mask


def _field_weights(weights: list[CopyWeights], layout: RecordHeadLayout) -> torch.Tensor:
    """Each copy's weight on each of the head's fields, one row per copy."""
    return torch.tensor([
        [copy.weight(str(spec.name)) for spec in layout.fields] for copy in weights
    ])


def created_objects_rows(records, weights: list[CopyWeights]) -> CreatedTargets | None:
    """The created-objects head's targets over ``records``, or None where none
    supervises it. The rule mask (FR-061) weighs the head
    whole: a copy weighing it zero does not supervise it, and any other
    weight but one (a soft share, a class weight) scales every column the
    record says."""
    weight = [copy.weight(CREATED_OBJECTS_OUTPUT) for copy in weights]
    created = [_supervised(record, share) for record, share in zip(records, weight)]
    if not any(created):
        return None
    targets, said = zip(*_created_rows(records, created))
    return CreatedTargets(
        torch.tensor(list(targets)), torch.tensor(created),
        _scaled_known(torch.tensor(list(said)), weight),
    )


def _supervised(record, share: float) -> bool:
    """Whether the created-objects head trains on ``record`` at weight ``share``."""
    return supervises_created_objects(record) and share > 0


def _created_rows(records, created: list[bool]) -> list:
    """Each record's created-objects target and the columns it says, a blank
    row where the record does not supervise the head."""
    blank = ([0.0] * CREATED_OBJECTS_WIDTH, [True] * CREATED_OBJECTS_WIDTH)
    return [
        created_objects_targets(record) if wanted else blank
        for record, wanted in zip(records, created)
    ]


def _scaled_known(known: torch.Tensor, weight: list[float]) -> torch.Tensor:
    """``known`` scaled by each record's weight where any is not zero or one."""
    if any(share not in (0.0, 1.0) for share in weight):
        known = known.float() * torch.tensor(weight).unsqueeze(-1)
    return known
