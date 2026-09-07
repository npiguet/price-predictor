"""What actually happened, as the per-entity head's targets.

A record stores *events* — "this creature moved to the graveyard", "this player
lost 3 life" — and the head predicts *fields*. This module is the bridge, and it
is the only place the event vocabulary meets the field layout, so a new event
type has exactly one place to say what it supervises.

Two rules run through it:

- **An entity is affected when any event names it.** The gate's target is that,
  and nothing else. It is deliberately not "some field changed": a spell that
  targets a creature and deals 0 damage to it still affected it, and a model
  that learns otherwise will predict "nothing happens" for every fizzle.
- **A field with no event is absent, not zero.** ``None`` means "this record
  says nothing about that field", and the loss skips it. Writing 0 instead would
  teach the head that every unmentioned counter type went to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from effects.domain.effect_head_input import COUNTER_TYPES
from effects.domain.effect_model import (
    LIBRARY_EVENTS,
    ZONE_OUTCOMES,
    FieldSpec,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    CombatPayload,
    EffectRecord,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import COLORS

#: Event types that move an entity out of where it was, and the zone outcome
#: each implies. A destroy and a sacrifice both end in the graveyard, but they
#: are different game actions with different replacement effects, so the record
#: keeps them apart and the target collapses them.
_ZONE_BY_EVENT: dict[EventType, str] = {
    EventType.DESTROYED: "died",
    EventType.SACRIFICED: "died",
    EventType.PHASED: "phased_out",
    EventType.FACE_CHANGE: "face_changed",
    EventType.SPELL_COUNTERED: "countered",
}

#: Destination zones a ``zone_change`` event maps to.
_ZONE_BY_DESTINATION: dict[str, str] = {
    "graveyard": "died",
    "exile": "exiled",
    "hand": "to_hand",
    "library": "library_top",
    "battlefield": "blinked",
}

_LIBRARY_EVENT_BY_TYPE: dict[EventType, str] = {
    EventType.CARD_LOOKED_AT: "scry",
    EventType.LIBRARY_REORDERED: "reorder",
    EventType.CARD_REVEALED: "reveal",
}


@dataclass
class EntityTargets:
    """One entity's targets for this record.

    ``affected`` is the gate. Every other field is ``None`` until an event says
    otherwise, which is what lets the loss skip what the record does not
    mention.
    """

    affected: bool = False
    fields: dict[str, float | int | list[float]] = field(default_factory=dict)

    def set(self, name: str, value) -> None:
        self.fields[name] = value

    def add(self, name: str, value: float) -> None:
        self.fields[name] = self.fields.get(name, 0) + value

    def flag(self, name: str, index: int, width: int) -> None:
        """Set one bit of a multi-binary field, allocating it on first use."""
        vector = self.fields.get(name)
        if not isinstance(vector, list):
            vector = [0.0] * width
            self.fields[name] = vector
        if 0 <= index < width:
            vector[index] = 1.0


def events_of(record: EffectRecord) -> tuple[Event, ...]:
    """Every event a record's payload carries, whatever its kind."""
    payload = record.payload
    if isinstance(payload, (ResolutionPayload, CombatPayload)):
        return payload.events
    if isinstance(payload, RewritePayload):
        return (payload.outgoing,)
    if isinstance(payload, TriggerPayload):
        return (payload.event,) if payload.fired else ()
    return ()


def derive_targets(record: EffectRecord) -> dict[str, EntityTargets]:
    """``entity or player id -> EntityTargets`` for one record."""
    targets: dict[str, EntityTargets] = {}

    def slot(subject: str) -> EntityTargets:
        return targets.setdefault(subject, EntityTargets())

    for event in events_of(record):
        for subject in event.subjects:
            entry = slot(subject)
            entry.affected = True
            _apply_event(entry, event)

    _apply_playability(record, slot)
    _apply_continuous(record, slot)
    return targets


def _apply_event(entry: EntityTargets, event: Event) -> None:
    """Fold one event into an entity's targets."""
    params = event.params
    match event.type:
        case EventType.ZONE_CHANGE:
            destination = str(params.get("to_zone", "")).lower()
            entry.set(
                "zone_outcome",
                ZONE_OUTCOMES.index(
                    _ZONE_BY_DESTINATION.get(destination, "stayed")
                ),
            )
        case (
            EventType.DESTROYED | EventType.SACRIFICED | EventType.PHASED
            | EventType.FACE_CHANGE | EventType.SPELL_COUNTERED
        ):
            entry.set("zone_outcome", ZONE_OUTCOMES.index(_ZONE_BY_EVENT[event.type]))
        case EventType.TAPPED:
            entry.set("tap_state", 1)
        case EventType.UNTAPPED:
            entry.set("tap_state", 0)
        case EventType.DAMAGE_DEALT:
            entry.add("damage_taken", float(params.get("amount", 0)))
        case EventType.DAMAGE_HEALED:
            entry.add("damage_taken", -float(params.get("amount", 0)))
        case EventType.COUNTER_CHANGE:
            counter = str(params.get("counter_type", "OTHER")).upper()
            name = (
                f"counters_delta_{counter.lower()}"
                if counter in COUNTER_TYPES
                else "counters_delta_other"
            )
            entry.add(name, float(params.get("delta", 0)))
        case EventType.PT_CHANGE:
            entry.add("power_delta", float(params.get("power_delta", 0)))
            entry.add("toughness_delta", float(params.get("toughness_delta", 0)))
        case EventType.KEYWORD_CHANGE:
            _apply_keywords(entry, params)
        case EventType.TYPE_CHANGE:
            _apply_multi(entry, "types_gained", params.get("types_added"), _TYPE_INDEX)
            _apply_multi(entry, "types_lost", params.get("types_removed"), _TYPE_INDEX)
        case EventType.COLOR_CHANGE:
            _apply_multi(entry, "colors_gained", params.get("colors"), _COLOR_INDEX)
        case EventType.CONTROL_CHANGE:
            entry.set("control_change", 1)
        case EventType.ATTACHED | EventType.UNATTACHED:
            entry.set("attached_to_change", 1)
        case EventType.LIFE_CHANGE:
            entry.add("life_delta", float(params.get("delta", 0)))
        case EventType.POISON_CHANGE:
            entry.add("poison_delta", float(params.get("delta", 0)))
        case EventType.CARD_DRAWN:
            entry.add("cards_drawn", float(params.get("count", 1)))
        case EventType.CARD_DISCARDED:
            entry.add("cards_discarded", float(params.get("count", 1)))
        case EventType.CARD_MILLED:
            entry.add("cards_milled", float(params.get("count", 1)))
        case EventType.MANA_PRODUCED:
            _apply_mana(entry, params.get("mana_by_color"), sign=1)
        case EventType.MANA_LOST:
            _apply_mana(entry, params.get("mana_by_color"), sign=-1)
        case (
            EventType.CARD_LOOKED_AT | EventType.LIBRARY_REORDERED
            | EventType.CARD_REVEALED
        ):
            entry.flag(
                "library_events",
                LIBRARY_EVENTS.index(_LIBRARY_EVENT_BY_TYPE[event.type]),
                len(LIBRARY_EVENTS),
            )


_TYPE_INDEX = "type"
_COLOR_INDEX = "color"


def _apply_multi(entry: EntityTargets, name: str, values, kind: str) -> None:
    from effects.domain.effect_head_input import CARD_TYPES

    vocabulary = CARD_TYPES if kind == _TYPE_INDEX else COLORS
    for value in values or ():
        normalized = str(value).strip().lower()
        for index, member in enumerate(vocabulary):
            if member.lower() == normalized:
                entry.flag(name, index, len(vocabulary))
                break


def _apply_keywords(entry: EntityTargets, params: dict) -> None:
    from effects.domain.effect_head_input import OVERLAY_KEYWORDS

    name = "keywords_lost" if params.get("removed") else "keywords_gained"
    for keyword in params.get("keywords") or ():
        normalized = str(keyword).strip().lower().replace(" ", "_")
        for index, member in enumerate(OVERLAY_KEYWORDS):
            if member == normalized:
                entry.flag(name, index, len(OVERLAY_KEYWORDS))
                break


def _apply_mana(entry: EntityTargets, mana, *, sign: int) -> None:
    for color, amount in (mana or {}).items():
        key = str(color).strip().upper()
        if key in COLORS:
            entry.add(f"mana_delta_{key.lower()}", sign * float(amount))


def _apply_playability(record: EffectRecord, slot) -> None:
    """Legality bits, which come from the payload rather than from events."""
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        for candidate in payload.candidates:
            for target in candidate.legal_targets:
                entry = slot(target)
                entry.affected = True
                entry.set("target_legal", 1)
    elif isinstance(payload, PlayabilityAttackersPayload):
        for entity in payload.legal_attackers:
            entry = slot(entity)
            entry.affected = True
            entry.set("attacker_legal", 1)
        for forbidden in payload.forbidden:
            entry = slot(forbidden.entity)
            entry.affected = True
            entry.set("attacker_legal", 0)
    elif isinstance(payload, PlayabilityBlockersPayload):
        for entity in payload.legal_blockers:
            entry = slot(entity)
            entry.affected = True
            entry.set("blocker_legal", 1)
        for forbidden in payload.forbidden:
            entry = slot(forbidden.entity)
            entry.affected = True
            entry.set("blocker_legal", 0)
        # min_blockers is read at the anchored attacker, not at the blockers.
        anchor = slot(payload.anchor_attacker)
        anchor.affected = True
        anchor.set("min_blockers", payload.min_blockers)


def _apply_continuous(record: EffectRecord, slot) -> None:
    """A continuous record's contributions are its targets directly."""
    payload = record.payload
    if not hasattr(payload, "contributions"):
        return
    from effects.domain.effect_head_input import OVERLAY_KEYWORDS

    for contribution in payload.contributions:
        entry = slot(contribution.entity)
        entry.affected = True
        if contribution.pt_boost != (0, 0):
            entry.add("power_delta", float(contribution.pt_boost[0]))
            entry.add("toughness_delta", float(contribution.pt_boost[1]))
        for keyword in contribution.keywords:
            normalized = keyword.strip().lower().replace(" ", "_")
            if normalized in OVERLAY_KEYWORDS:
                entry.flag(
                    "keywords_gained", OVERLAY_KEYWORDS.index(normalized),
                    len(OVERLAY_KEYWORDS),
                )
        _apply_multi(entry, "types_gained", contribution.types, _TYPE_INDEX)
        _apply_multi(entry, "colors_gained", contribution.colors, _COLOR_INDEX)


def target_for(
    targets: EntityTargets, spec: FieldSpec,
) -> float | int | list[float] | None:
    """One field's target, or None where the record says nothing about it."""
    return targets.fields.get(spec.name)
