"""The entities an ``outcome_event`` tag names in its two roles.

The condition entities are the acting ability's source, its targets and its
controller. The outcome entities are where the matching events' outcomes are
placed (``effect_targets.placed_subjects``), with the owner of a card a
``zone_change`` moves out of a hand or a library; an outcome that did not
happen names where it would have fallen.
"""

from __future__ import annotations

from collections.abc import Iterable

from effects.domain.effect_targets import placed_subjects
from effects.domain.event_schema import Event, EventType
from effects.domain.records import EffectRecord
from effects.domain.rule_detectors.acting_line import source_ids

_OWNED_ZONES = frozenset({"hand", "library"})


def condition_entities(record: EffectRecord) -> tuple[str, ...]:
    """The acting ability's source, its targets and its controller."""
    actor = (record.actor_player,) if record.actor_player else ()
    return tuple(dict.fromkeys((*source_ids(record), *record.state.refs.as_cast, *actor)))


def would_fall_on(record: EffectRecord) -> tuple[str, ...]:
    """Where an outcome that did not happen would have fallen: the acting
    clause's targets as the effect half's refs block holds them (those left
    after the illegal-target check, so a target gone since names no slot),
    else its controller."""
    targets = record.state.refs.targets
    if targets:
        return tuple(dict.fromkeys(targets))
    return (record.actor_player,) if record.actor_player else ()


def near_outcome(record: EffectRecord, events: Iterable[Event]) -> tuple[str, ...]:
    """A qualified near miss's outcome entities: where the events that
    missed the qualifiers fell, else where the outcome would have fallen."""
    return outcome_entities(record, events) or would_fall_on(record)


def _leaves_owned_zone(event: Event) -> bool:
    """Whether ``event`` moves a card out of a hand or a library."""
    return (event.type is EventType.ZONE_CHANGE
            and str(event.params.get("from_zone", "")).lower() in _OWNED_ZONES)


def _owner_of(record: EffectRecord, subject: str) -> str | None:
    entity = record.state.entity(subject)
    return None if entity is None else entity.owner or entity.controller


def outcome_entities(record: EffectRecord, events: Iterable[Event]) -> tuple[str, ...]:
    """Where ``events``' outcomes are placed, and the owner of a card a
    ``zone_change`` moves out of a hand or a library."""
    found: list[str] = []
    for event in events:
        placed = placed_subjects(record, event)
        found += placed
        if _leaves_owned_zone(event):
            owners = (_owner_of(record, subject) for subject in placed)
            found += [owner for owner in owners if owner]
    return tuple(dict.fromkeys(found))
