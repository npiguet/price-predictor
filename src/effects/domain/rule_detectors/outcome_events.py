"""How ``outcome_event`` reads a record's events: the declared equalities, the
event types a declaration names, and what a damage event's source did.

``rewrite_event`` reads events with the same equalities (:func:`event_filters`,
:func:`meets_filters`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EVENT_PARAMS, Event, EventType, damage_source
from effects.domain.records import EffectRecord
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.entity_facts import has_card_type
from effects.domain.state_snapshot import CombatSubstep, normalize_counter_type
from effects.domain.wire_closed_lists import DAMAGE_TYPES, GAME_LOSS_REASONS

#: Event parameters a declaration may not filter on: they name a card or an
#: object of one game (``cause``, ``card_name``, ``controller``), never a
#: condition of a rule, or hold a structure no equality reads.
UNFILTERABLE = frozenset({
    "cause", "source", "card_name", "token_script_id", "copy_source", "characteristics",
    "controller", "owner", "attached_to", "detached_from", "targets", "blockers", "blocked",
    "defender", "called", "options", "piles", "mana_by_color", "name", "mana_cost"})
EVENT_FILTERS = frozenset(
    name for names in EVENT_PARAMS.values() for name in names) - UNFILTERABLE
#: The checks an event filter's value passes (``declared_values``).
EVENT_FILTER_CHECKS = {
    "from_zone": dv.zone, "to_zone": dv.zone, "counter_type": dv.counter_type,
    "to_state": dv.face_state, "damage_type": dv.one_of(*DAMAGE_TYPES),
    "reason": dv.one_of(*GAME_LOSS_REASONS),
}
#: The qualifiers naming another event of the same record.
RECORD_QUALIFIERS = frozenset({"with_event", "without_event"})


def same_value(found: Any, wanted: Any, name: str | None = None) -> bool:
    """An event parameter's equality, text compared without case; a counter
    type (``name`` ``counter_type``) in ``normalize_counter_type``'s spelling."""
    if not isinstance(found, str) or not isinstance(wanted, str):
        return found == wanted
    if name == "counter_type":
        return normalize_counter_type(found) == normalize_counter_type(wanted)
    return found.lower() == wanted.lower()


def meets_filters(event: Event, filters: Mapping[str, Any]) -> bool:
    """Whether ``event`` meets every one of a declaration's equalities."""
    found = event.params or {}
    return all(same_value(found.get(name), value, name) for name, value in filters.items())


def event_filters(family: str, event: EventType, params: Mapping[str, Any],
                  own: frozenset[str]) -> dict[str, Any]:
    """The declaration's event-parameter equalities: every parameter not in
    ``own``, each one the event's schema lists and an equality can read."""
    filters = {name: value for name, value in params.items() if name not in own}
    unknown = set(filters) - set(EVENT_PARAMS.get(event, ())) | set(filters) & UNFILTERABLE
    if unknown:
        raise ValueError(f"{family}: {event.value} has no parameter {sorted(unknown)}")
    return filters


@dataclass(frozen=True, slots=True)
class EventSpec:
    """A ``with_event``/``without_event`` value: an event type and its filters."""

    type: EventType
    filters: Mapping[str, Any]

    @classmethod
    def of(cls, spec: Any) -> EventSpec:
        """A type, or a map of ``event`` and its filters; an unknown type or filter raises."""
        if isinstance(spec, Mapping):
            event = EventType(spec["event"])
            return cls(event, event_filters("outcome_event", event, spec, frozenset({"event"})))
        return cls(EventType(spec), {})

    def held_in(self, listed: tuple[Event, ...]) -> bool:
        """Whether one of ``listed`` is such an event."""
        return any(e.type is self.type and meets_filters(e, self.filters) for e in listed)


def declared_events(params: Mapping[str, Any]) -> frozenset[EventType]:
    """Every event type a declaration reads: its ``event`` and the types its
    ``with_event``/``without_event`` name."""
    found = {EventType(params["event"])}
    for name in RECORD_QUALIFIERS & set(params):
        found.add(EventSpec.of(params[name]).type)
    return frozenset(found)


def _subtypes(type_line: str) -> frozenset[str]:
    """The subtypes of a ``Creature - Dragon Horror`` type line, lower-cased."""
    _, dash, rest = type_line.partition(" - ")
    return frozenset(word.lower() for word in rest.split()) if dash else frozenset()


def token_kind_of(event: Event) -> frozenset[str]:
    """A ``token_created`` event's token subtypes, lower-cased."""
    characteristics = (event.params or {}).get("characteristics") or {}
    return _subtypes(str(characteristics.get("types") or ""))


def _to_graveyard(event: Event) -> bool:
    """Whether ``event`` moves its subjects from the battlefield to a graveyard."""
    found = event.params or {}
    return (same_value(found.get("from_zone"), "battlefield")
            and same_value(found.get("to_zone"), "graveyard"))


def dies(entity_id: str, listed: tuple[Event, ...]) -> bool:
    """Whether the record moves ``entity_id`` from the battlefield to a graveyard."""
    return any(e.type is EventType.ZONE_CHANGE and entity_id in e.subjects and _to_graveyard(e)
               for e in listed)


def source_dies_verdict(event: Event, listed: tuple[Event, ...],
                        source_dies: bool | None) -> bool | None:
    """Whether ``event``'s ``source`` dies in the record as a declared
    ``source_dies`` says (True where it declares none); None where the event
    names no source."""
    if source_dies is None:
        return True
    source = damage_source(event)
    if source is None:
        return None
    return dies(source, listed) == source_dies


def _dies_without_striking(record: EffectRecord, subject: str,
                           listed: tuple[Event, ...]) -> bool:
    """Whether ``subject`` is a creature with power above 0 that dies in the record."""
    entity = record.state.entity(subject)
    if entity is None or not has_card_type(entity, "creature"):
        return False
    return entity.power > 0 and dies(subject, listed)


def first_strike_victims(record: EffectRecord, wanted: list[Event],
                         listed: tuple[Event, ...]) -> tuple[str, ...]:
    """A first striker's victims: in the first-strike damage step, the
    creatures with power above 0 that ``wanted``'s events deal damage to and
    that die in the record having dealt none of those events themselves.
    None in any other step, or where the snapshot names no step."""
    if record.state.global_.combat_substep != CombatSubstep.FIRST_STRIKE:
        return ()
    sources = {damage_source(e) for e in wanted}
    found = [subject for event in wanted for subject in event.subjects
             if subject not in sources and _dies_without_striking(record, subject, listed)]
    return tuple(dict.fromkeys(found))
