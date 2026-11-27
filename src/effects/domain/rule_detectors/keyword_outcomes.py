"""Whether a damage-step keyword's effect happened in a combat record.

``keyword_combat`` tags a combat ``fires`` where gate 2's table says the
keyword's effect is observable; this reads the record's own events for the
effect itself (the opponent died, the controller gained life, the defending
player lost life or took poison, a creature took damage or -1/-1 counters), so
a combat whose outcome did not happen, a game-ending combat for one, is a tag
whose ``observed`` disagrees with its polarity and goes to review.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from effects.domain.damage_step_keywords import KeywordEffect
from effects.domain.effect_head_layout import HeadField, counter_field
from effects.domain.event_schema import Event, EventType

_DEATH_ZONES = frozenset({"graveyard", "exile"})
_M1M1 = counter_field("M1M1")


def _delta_has_sign(event: Event, direction: int) -> bool:
    return int((event.params or {}).get("delta") or 0) * direction > 0


def died(events: Iterable[Event], subject: str) -> bool:
    """Whether ``events`` move ``subject`` from the battlefield to a graveyard
    or exile: the death a zone-outcome effect reads."""
    return any(e.type is EventType.ZONE_CHANGE and subject in e.subjects
               and (e.params or {}).get("from_zone") == "battlefield"
               and (e.params or {}).get("to_zone") in _DEATH_ZONES for e in events)


def _died(events: Iterable[Event], subject: str, _direction: int) -> bool:
    return died(events, subject)


def _of_type(event_type: EventType, signed: bool) -> Callable[[Iterable[Event], str, int], bool]:
    def happened(events: Iterable[Event], subject: str, direction: int) -> bool:
        return any(e.type is event_type and subject in e.subjects
                   and (_delta_has_sign(e, direction) if signed else True) for e in events)
    return happened


def _m1m1_counters(events: Iterable[Event], subject: str, direction: int) -> bool:
    return any(e.type is EventType.COUNTER_CHANGE and subject in e.subjects
               and str((e.params or {}).get("counter_type", "")).upper() == "M1M1"
               and _delta_has_sign(e, 1) for e in events)


_READERS: dict[str, Callable[[Iterable[Event], str, int], bool]] = {
    HeadField.ZONE_OUTCOME: _died,
    HeadField.LIFE_DELTA: _of_type(EventType.LIFE_CHANGE, signed=True),
    HeadField.POISON_DELTA: _of_type(EventType.POISON_CHANGE, signed=True),
    HeadField.DAMAGE_TAKEN: _of_type(EventType.DAMAGE_DEALT, signed=False),
    _M1M1: _m1m1_counters,
}


def effect_happened(effect: KeywordEffect, subject: str, events: Iterable[Event]) -> bool:
    """Whether ``events`` show ``effect`` moving ``subject``'s field its way."""
    reader = _READERS.get(effect.field)
    return reader is not None and reader(tuple(events), subject, effect.direction)
