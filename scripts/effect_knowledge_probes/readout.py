"""The model's own answer to each probe question, read one way everywhere.

The ladder's rung 3, the board sweeps and the one-off grids all ask the full
model the same few questions — will this creature die, may it block, is this
ability affordable — and each answer lives in one output: a per-entity field
at the entity's ``[CARD]`` slot, or a verdict bit of the record. Reading every
answer through :data:`READOUTS` keeps a grid's "predicted death" the very
number the ladder scores.

Field answers are gated as gate 2 reads them (:class:`HeadReading`): a field
head trains only where an entity was affected, so its answer is scaled by the
gate's. ``damage_taken`` is the exception: every one of its items took damage,
so the gate's doubt would scale the amount by noise, and the amount is read
alone.

Nothing here does work at import time.
"""

from __future__ import annotations

from collections.abc import Callable

#: ``readout(outputs, verdict, row, slot)``: one answer, a scalar tensor.
Readout = Callable[[object, object, int, int], object]


def _reading(outputs, row: int, slot: int):
    from effects.domain.head_decoding import HeadReading

    return HeadReading(outputs[row, slot])


def _affected(outputs, verdict, row: int, slot: int):
    return _reading(outputs, row, slot).p_affected


def _dies(outputs, verdict, row: int, slot: int):
    from effects.domain.effect_head_layout import ZoneOutcome

    return _reading(outputs, row, slot).p_zone(ZoneOutcome.DIED)


def _damage_taken(outputs, verdict, row: int, slot: int):
    from effects.domain.effect_head_layout import HeadField

    return _reading(outputs, row, slot).expected(HeadField.DAMAGE_TAKEN, conditional=True)


def _pt_until_end_of_turn(outputs, verdict, row: int, slot: int):
    from effects.domain.effect_head_layout import HeadField

    return _reading(outputs, row, slot).p_class(HeadField.PT_DURATION, "end_of_turn")


def _may_block(outputs, verdict, row: int, slot: int):
    import torch

    from effects.domain.effect_head_layout import FIELD_SLICES, HeadField

    return torch.sigmoid(outputs[row, slot, FIELD_SLICES[HeadField.BLOCKER_LEGAL][0]])


def _affordable(outputs, verdict, row: int, slot: int):
    import torch

    from effects.domain.effect_head_layout import VERDICT_BITS

    return torch.sigmoid(verdict[row, VERDICT_BITS.index("affordable")])


def _fires(outputs, verdict, row: int, slot: int):
    import torch

    return torch.sigmoid(verdict[row, -1])


READOUTS: dict[str, Readout] = {
    "affected": _affected,
    "dies": _dies,
    "damage_taken": _damage_taken,
    "pt_until_end_of_turn": _pt_until_end_of_turn,
    "may_block": _may_block,
    # The gate, not the `target_legal` field: training sets that field only on
    # legal targets, always to 1, so on an illegal entity its output was never
    # supervised and ranks nothing. Legality is learned through the gate, which
    # a decision record raises exactly on its legal targets.
    "legal_target": _affected,
    "legal_target_protected": _affected,
    "affordable": _affordable,
    "fires": _fires,
}


def read(target: str, outputs, verdict, row: int, slot: int) -> float:
    """``target``'s answer for batch row ``row`` at ``slot``; a verdict ignores the slot."""
    return float(READOUTS[target](outputs, verdict, row, slot))


def card_slots(surface) -> dict[str, int]:
    """Each card entity's ``[CARD]`` slot in one record's surface."""
    from effects.domain.effect_head_input import SlotKind

    slots: dict[str, int] = {}
    for index, slot in enumerate(surface.slots):
        if slot.kind is SlotKind.CARD and slot.entity_id is not None:
            slots.setdefault(slot.entity_id, index)
    return slots


def at_card(target: str, entity_id: str):
    """A ``common.predict`` read: ``target`` at one entity's card, NaN off the board."""

    def read_one(outputs, verdict, surface, row) -> float:
        slot = card_slots(surface).get(entity_id)
        return float("nan") if slot is None else read(target, outputs, verdict, row, slot)

    return read_one


def at_cards(target: str, entity_ids):
    """A ``common.predict`` read: row ``r`` of the batch at ``entity_ids[r]``'s card."""
    return lambda outputs, verdict, surface, row: at_card(target, entity_ids[row])(
        outputs, verdict, surface, row)


def at_every_card(target: str):
    """A ``common.predict`` read: ``{entity: answer}`` over every card on the board."""

    def read_all(outputs, verdict, surface, row) -> dict[str, float]:
        return {entity: read(target, outputs, verdict, row, slot)
                for entity, slot in card_slots(surface).items()}

    return read_all


def of_record(target: str):
    """A ``common.predict`` read: a verdict, which is the record's and no entity's."""
    return lambda outputs, verdict, surface, row: read(target, outputs, verdict, row, 0)
