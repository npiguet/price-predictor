"""The probes' reads of the model's answers, as ``common.predict`` takes them.

The answers themselves — :data:`READOUTS`, one per probe question, gated as
gate 2 reads them — and :func:`card_slots` are the package's
(``effects.domain.probe_readouts``), so a probe reads the very number the
control score selects on. This module adapts them to ``common.predict``'s
``read(outputs, verdict, surface, row)``: at one entity's card, at each row's
entity, at every card, or of the record.

Nothing here does work at import time.
"""

from __future__ import annotations

from effects.domain.probe_readouts import (  # noqa: F401  (re-exported for the probes)
    READOUTS,
    Readout,
    card_slots,
    entity_slots,
    read,
)


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


def at_every_entity(target: str):
    """A ``common.predict`` read: ``{entity or player: answer}`` over every
    card and every player of the board."""

    def read_all(outputs, verdict, surface, row) -> dict[str, float]:
        return {entity: read(target, outputs, verdict, row, slot)
                for entity, slot in entity_slots(surface).items()}

    return read_all


def of_record(target: str):
    """A ``common.predict`` read: a verdict, which is the record's and no entity's."""
    return lambda outputs, verdict, surface, row: read(target, outputs, verdict, row, 0)
