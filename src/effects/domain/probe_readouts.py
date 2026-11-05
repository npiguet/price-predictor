"""The model's own answer to each probe question, read one way everywhere, and
how an item of answers is scored (research E1, E6).

The ladder's rung 3, the board sweeps, the grids, the rule tests and the
control score all ask the full model the same few questions — will this
creature die, may it block, is this ability affordable — and each answer lives
in one output: a per-entity field at the entity's ``[CARD]`` slot, or a verdict
bit of the record. Reading every answer through :data:`READOUTS` keeps a grid's
"predicted death" the very number the ladder scores, and the number training
selects on the number the probes report.

Field answers are gated as gate 2 reads them (``HeadReading``): a field head
trains only where an entity was affected, so its answer is scaled by the
gate's. ``damage_taken`` is the exception: every one of its items took damage,
so the gate's doubt would scale the amount by noise, and the amount is read
alone.

An item scores as an AUC for a yes/no target and an R² for an amount
(:func:`item_score`), both from ``score_scales``.

Torch-free at import: a readout imports torch when it is called, so the
selection-set freeze can name the readouts without loading it.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from effects.domain.score_scales import auc, r_squared

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


def _verdict_column(field: str, member: str | None = None) -> Readout:
    """A binary verdict output at ``[ACT]``: ``field``'s column, or one
    member's column of a field with members (``failed_checks``)."""

    def read(outputs, verdict, row: int, slot: int):
        import torch

        from effects.domain.effect_head_layout import VERDICT_LAYOUT

        start, _ = VERDICT_LAYOUT.slices[field]
        offset = 0 if member is None else VERDICT_LAYOUT.by_name[field].members.index(member)
        return torch.sigmoid(verdict[row, start + offset])

    return read


def _legality_field(field: str) -> Readout:
    """A legality field at an entity's slot, read alone: the legality fields
    train wherever present, gate or no gate (``FieldSpec.gated`` false)."""

    def read(outputs, verdict, row: int, slot: int):
        import torch

        from effects.domain.effect_head_layout import FIELD_SLICES

        return torch.sigmoid(outputs[row, slot, FIELD_SLICES[field][0]])

    return read


def _failed_check_readouts() -> dict[str, Readout]:
    from effects.domain.effect_head_layout import FAILED_CHECKS, VerdictField

    return {f"failed_{check}": _verdict_column(VerdictField.FAILED_CHECKS, check)
            for check in FAILED_CHECKS}


READOUTS: dict[str, Readout] = {
    "affected": _affected,
    "dies": _dies,
    "damage_taken": _damage_taken,
    "pt_until_end_of_turn": _pt_until_end_of_turn,
    "may_block": _may_block,
    # The `target_legal` field, not the gate (FR-074): gen-3 trains the field
    # on every card and player of a decision whose candidate targets, 1 at a
    # legal target and 0 elsewhere, and trains no gate on a decision record,
    # which observes what could be chosen rather than what changed.
    "legal_target": _legality_field("target_legal"),
    "legal_target_protected": _legality_field("target_legal"),
    "affordable": _affordable,
    "fires": _verdict_column("trigger_fired"),
    # Gen-3 trains the field itself on every card and player of a decision
    # whose candidate targets: 1 at a legal target, 0 elsewhere (FR-074).
    "target_legal": _legality_field("target_legal"),
    "may_attack": _legality_field("attacker_legal"),
    "can_play": _verdict_column("can_play"),
    "timing": _verdict_column("timing"),
    # One per check a candidate can fail: the probability it fails.
    **_failed_check_readouts(),
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


def entity_slots(surface) -> dict[str, int]:
    """Each card entity's ``[CARD]`` slot and each player's ``[PLAYER]`` slot
    in one record's surface: what a field scored at players as well as
    cards, such as ``target_legal``, is read at."""
    from effects.domain.effect_head_input import SlotKind

    slots = card_slots(surface)
    for index, slot in enumerate(surface.slots):
        if slot.kind is SlotKind.PLAYER and slot.player_id is not None:
            slots.setdefault(slot.player_id, index)
    return slots


def item_score(kind: str, y, prediction, weight=None) -> float:
    """One item's score: AUC for a yes/no target (``kind == "binary"``), R² for
    an amount; rows not finite in both ``y`` and ``prediction`` are dropped,
    and NaN where the score is undefined (fewer than two rows, one label)."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[ok], prediction[ok]
    weight = None if weight is None else np.asarray(weight, dtype=np.float64)[ok]
    if y.size < 2:
        return float("nan")
    if kind == "binary":
        if len(np.unique(y)) < 2:
            return float("nan")
        return auc(y, prediction, weight)
    return r_squared(y, prediction, weight)
