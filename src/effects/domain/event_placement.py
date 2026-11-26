"""Where an event's outcome lands: its subjects, or, for the events Forge
names no subject for (a copy, a retarget, a venture), the object they
happened to; and a move's zones as the targets read them.
"""

from __future__ import annotations

from collections.abc import Callable

from effects.domain.event_schema import Event, EventType
from effects.domain.records import EffectRecord, events_of
from effects.domain.state_snapshot import Zone


def from_zone(event: Event) -> str:
    """A move's ``from_zone``, lower case; empty where it names none."""
    return str(event.params.get("from_zone", "")).lower()


def to_zone(event: Event) -> str:
    """A move's ``to_zone``, lower case; empty where it names none."""
    return str(event.params.get("to_zone", "")).lower()


def placed_subjects(record: EffectRecord, event: Event,
                    events: tuple[Event, ...] | None = None) -> tuple[str, ...]:
    """Where an event's outcome is placed: its subjects, or, for the events
    Forge names no subject for, the object they happened to.

    A Clone's copy is placed on the Clone, which is the cause and enters
    from the stack in the same record; a spell copy or a retarget on the
    stack objects the acting ability targets, else on its cause when that is
    a stack object; venturing on the venturing ability's controller, the
    cause's where the snapshot holds it. Empty where none is found.
    ``events`` are the record's target events (``events_of``), read for a
    copy's entry."""
    if event.subjects:
        return tuple(event.subjects)
    place = _PLACERS.get(event.type)
    if place is None:
        return ()
    return place(record, event, events)


def _placed_copy(record: EffectRecord, event: Event,
                 events: tuple[Event, ...] | None) -> tuple[str, ...]:
    """A Clone's copy: on its cause, where the cause enters from the stack
    in the same record."""
    cause = event.params.get("cause")
    entered = _entered_from_stack(events if events is not None else events_of(record))
    return (cause,) if cause in entered else ()


def _entered_from_stack(events: tuple[Event, ...]) -> set[str]:
    return {
        subject for other in events if _enters_from_stack(other)
        for subject in other.subjects
    }


def _enters_from_stack(event: Event) -> bool:
    return (
        event.type is EventType.ZONE_CHANGE
        and from_zone(event) == Zone.STACK and to_zone(event) == Zone.BATTLEFIELD
    )


def _placed_on_stack(record: EffectRecord, event: Event,
                     _events: tuple[Event, ...] | None) -> tuple[str, ...]:
    """A spell copy or a retarget: on the stack objects the acting ability
    targets, else on its cause when that is a stack object."""
    entities = _entities_by_id(record)
    on_stack = tuple(
        ref for ref in record.state.refs.targets if _is_on_stack(entities, ref))
    if on_stack:
        return on_stack
    cause = event.params.get("cause")
    return (cause,) if _is_on_stack(entities, cause) else ()


def _is_on_stack(entities: dict, ref) -> bool:
    return ref in entities and entities[ref].zone == Zone.STACK


def _placed_venture(record: EffectRecord, event: Event,
                    _events: tuple[Event, ...] | None) -> tuple[str, ...]:
    """Venturing: on the venturing ability's controller, the cause's where
    the snapshot holds it, else the acting player; none where that is no
    player of the snapshot."""
    found = _entities_by_id(record).get(event.params.get("cause"))
    player = found.controller if found is not None else record.actor_player
    return (player,) if player in {p.id for p in record.state.players} else ()


def _entities_by_id(record: EffectRecord) -> dict:
    return {entity.id: entity for entity in record.state.entities}


#: ``event type -> where an event of it that names no subject is placed``.
_PLACERS: dict[
    EventType,
    Callable[[EffectRecord, Event, tuple[Event, ...] | None], tuple[str, ...]],
] = {
    EventType.PERMANENT_COPIED: _placed_copy,
    EventType.SPELL_COPIED: _placed_on_stack,
    EventType.TARGETS_CHANGED: _placed_on_stack,
    EventType.DUNGEON_VENTURED: _placed_venture,
}
