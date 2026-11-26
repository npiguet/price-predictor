"""A baseline variant's masks applied to a batch's collated trunk inputs (FR-094)."""

from __future__ import annotations

import dataclasses

import torch

from effects.domain.baselines import BaselineMasks
from effects.domain.input_layout import ACT_SLOT, KIND_FLAG_WIDTH, SlotKind
from effects.domain.surface_collation import TrunkInputs

#: A card's features beyond its slot's, zeroed with it: its subtypes and counters.
_CARD_IDS = (
    "subtype_ids", "chosen_type_ids", "real_face_subtype_ids",
    "counter_ids", "counter_counts",
)


def apply_masks(inputs: TrunkInputs, masks: BaselineMasks) -> TrunkInputs:
    """``inputs`` with the variant's masks applied (FR-094), zeroed rather than
    dropped so the per-entity targets keep their shape."""
    if masks.zero_state:
        inputs = _zero_state(inputs)
    if masks.zero_act_e or masks.zero_context_e:
        _zero_e(inputs, masks)
    return inputs


def _zero_state(inputs: TrunkInputs) -> TrunkInputs:
    """No-state: the player, card and ability features, the global ones but
    the record-kind flags, a card's subtypes and counters, and the board's
    links."""
    features = inputs.slot_features
    for kind in (SlotKind.PLAYER, SlotKind.CARD, SlotKind.ABILITY):
        if kind in features:
            features[kind] = torch.zeros_like(features[kind])
    features[SlotKind.GLOBAL][..., :-KIND_FLAG_WIDTH] = 0.0
    inputs = dataclasses.replace(inputs, **{
        name: torch.zeros_like(getattr(inputs, name))
        for name in _CARD_IDS
        if getattr(inputs, name) is not None
    })
    # The board's links are state too: no-state reads none.
    if inputs.relations is not None:
        inputs = dataclasses.replace(inputs, relations=inputs.relations[:0])
    return inputs


def _zero_e(inputs: TrunkInputs, masks: BaselineMasks) -> None:
    """The ``e`` of the slots the masks name: every context slot, and ``[ACT]``
    as ``zero_act_e`` says."""
    zeroed = torch.full(
        (inputs.e_rows.shape[1],), masks.zero_context_e, dtype=torch.bool,
    )
    if zeroed.numel() > ACT_SLOT:
        zeroed[ACT_SLOT] = masks.zero_act_e
    inputs.e_rows[:, zeroed] = -1
    inputs.e_vectors[:, zeroed] = 0.0
