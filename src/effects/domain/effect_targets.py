"""What actually happened, as the per-entity head's targets.

The one place the event vocabulary meets the field layout. An entity is
affected when any event names it, not when some field changed: a 0-damage hit
still affected its target. A field no event wrote is absent rather than zero
unless the layout gives it a silent value for this record's class and entity
kind (:meth:`EntityTargets.supervised`); zero everywhere would teach the head
that a destroyed creature was untapped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    OVERLAY_KEYWORDS,
    vocabulary_slot,
)
from effects.domain.effect_head_layout import (
    DURATIONS,
    LIBRARY_EVENTS,
    OTHER_COUNTER,
    ZONE_OUTCOMES,
    FieldScope,
    FieldSpec,
    HeadField,
    LibraryEvent,
    ZoneOutcome,
    counter_field,
    mana_field,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.keyword_names import overlay_keyword
from effects.domain.records import (
    ContinuousPayload,
    EffectRecord,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    TriggerPayload,
    events_of,
)
from effects.domain.sampling import SamplingClass, sampling_class
from effects.domain.state_snapshot import COLORS, Zone

#: The zone outcome each moving event implies; the record keeps destroy and
#: sacrifice apart, the target collapses them.
_ZONE_BY_EVENT: dict[EventType, ZoneOutcome] = {
    EventType.DESTROYED: ZoneOutcome.DIED,
    EventType.SACRIFICED: ZoneOutcome.DIED,
    EventType.PHASED: ZoneOutcome.PHASED_OUT,
    EventType.FACE_CHANGE: ZoneOutcome.FACE_CHANGED,
    EventType.SPELL_COUNTERED: ZoneOutcome.COUNTERED,
}

_ZONE_BY_DESTINATION: dict[Zone, ZoneOutcome] = {
    Zone.GRAVEYARD: ZoneOutcome.DIED,
    Zone.EXILE: ZoneOutcome.EXILED,
    Zone.HAND: ZoneOutcome.TO_HAND,
    Zone.LIBRARY: ZoneOutcome.LIBRARY_TOP,
    Zone.BATTLEFIELD: ZoneOutcome.BLINKED,
}
#: The outcome of a ``zone_change`` to a destination the table does not name.
_STAYED = ZoneOutcome.STAYED

_LIBRARY_EVENT_BY_TYPE: dict[EventType, LibraryEvent] = {
    EventType.CARD_LOOKED_AT: LibraryEvent.SCRY,
    EventType.LIBRARY_REORDERED: LibraryEvent.REORDER,
    EventType.CARD_REVEALED: LibraryEvent.REVEAL,
}

#: ``type -> (field, parameter, default when absent, sign)``.
_ADDITIVE: dict[EventType, tuple[str, str, float, float]] = {
    EventType.DAMAGE_DEALT: (HeadField.DAMAGE_TAKEN, "amount", 0, 1.0),
    EventType.DAMAGE_HEALED: (HeadField.DAMAGE_TAKEN, "amount", 0, -1.0),
    EventType.LIFE_CHANGE: (HeadField.LIFE_DELTA, "delta", 0, 1.0),
    EventType.POISON_CHANGE: (HeadField.POISON_DELTA, "delta", 0, 1.0),
    EventType.CARD_DRAWN: (HeadField.CARDS_DRAWN, "count", 1, 1.0),
    EventType.CARD_DISCARDED: (HeadField.CARDS_DISCARDED, "count", 1, 1.0),
    EventType.CARD_MILLED: (HeadField.CARDS_MILLED, "count", 1, 1.0),
}

_SETTERS: dict[EventType, tuple[str, int]] = {
    EventType.TAPPED: (HeadField.TAP_STATE, 1),
    EventType.UNTAPPED: (HeadField.TAP_STATE, 0),
    EventType.CONTROL_CHANGE: (HeadField.CONTROL_CHANGE, 1),
    EventType.ATTACHED: (HeadField.ATTACHED_TO_CHANGE, 1),
    EventType.UNATTACHED: (HeadField.ATTACHED_TO_CHANGE, 1),
}


def overlay_index(raw: str) -> int | None:
    """The overlay slot a keyword, in any spelling Forge writes, sets."""
    return vocabulary_slot(overlay_keyword(raw), OVERLAY_KEYWORDS)


@dataclass
class EntityTargets:
    """One entity's targets for this record.

    ``fields`` holds what the record wrote and nothing else, ``unknown`` what
    it declares it cannot say; ``record_class`` and ``scope`` decide which
    unwritten fields take their silent value.
    """

    record_class: SamplingClass
    scope: FieldScope
    affected: bool = False
    fields: dict[str, float | int | list[float]] = field(default_factory=dict)
    unknown: set[str] = field(default_factory=set)

    def supervised(self, spec: FieldSpec):
        """The target ``spec`` is scored on for this entity, or None where absent."""
        return spec.supervised_value(
            self.fields, self.unknown,
            record_class=self.record_class, scope=self.scope,
        )

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


def derive_targets(record: EffectRecord) -> dict[str, EntityTargets]:
    """``entity or player id -> EntityTargets`` for one record."""
    targets: dict[str, EntityTargets] = {}
    record_class = sampling_class(record)
    players = {player.id for player in record.state.players}

    def slot(subject: str) -> EntityTargets:
        entry = targets.get(subject)
        if entry is None:
            entry = targets[subject] = EntityTargets(
                record_class=record_class,
                scope=FieldScope.PLAYER if subject in players else FieldScope.PERMANENT,
            )
        return entry

    for event in _target_events(record):
        for subject in event.subjects:
            entry = slot(subject)
            entry.affected = True
            _apply_event(entry, event)

    _apply_playability(record, slot)
    _apply_continuous(record, slot)
    return targets


def _target_events(record: EffectRecord) -> tuple[Event, ...]:
    """The events the head trains on as this record's outcome: those that
    happened, except on a trigger record, whose event caused it and whose
    effect is its resolution's record."""
    if isinstance(record.payload, TriggerPayload):
        return ()
    return events_of(record)


def _apply_event(entry: EntityTargets, event: Event) -> None:
    """Fold one event into an entity's targets; an event no applier names
    writes nothing."""
    apply = _EVENT_APPLIERS.get(event.type)
    if apply is not None:
        apply(entry, event)


def _apply_additive(
    entry: EntityTargets, event: Event, *,
    name: str, parameter: str, default: float, sign: float,
) -> None:
    entry.add(name, sign * float(event.params.get(parameter, default)))


def _apply_setter(entry: EntityTargets, _event: Event, *, name: str, value: int) -> None:
    entry.set(name, value)


def _apply_zone_outcome(
    entry: EntityTargets, _event: Event, *, outcome: ZoneOutcome,
) -> None:
    entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(outcome))


def _apply_library_event(
    entry: EntityTargets, _event: Event, *, library_event: LibraryEvent,
) -> None:
    entry.flag(
        HeadField.LIBRARY_EVENTS, LIBRARY_EVENTS.index(library_event), len(LIBRARY_EVENTS),
    )


def _apply_zone_change(entry: EntityTargets, event: Event) -> None:
    destination = str(event.params.get("to_zone", "")).lower()
    entry.set(
        HeadField.ZONE_OUTCOME,
        ZONE_OUTCOMES.index(_ZONE_BY_DESTINATION.get(destination, _STAYED)),
    )


def _apply_counter_change(entry: EntityTargets, event: Event) -> None:
    counter = str(event.params.get("counter_type", OTHER_COUNTER)).upper()
    name = (
        counter_field(counter)
        if counter in COUNTER_TYPES
        else HeadField.COUNTERS_DELTA_OTHER
    )
    entry.add(name, float(event.params.get("delta", 0)))


def _apply_pt_change(entry: EntityTargets, event: Event) -> None:
    entry.add(HeadField.POWER_DELTA, float(event.params.get("power_delta", 0)))
    entry.add(HeadField.TOUGHNESS_DELTA, float(event.params.get("toughness_delta", 0)))
    _apply_duration(entry, HeadField.PT_DURATION, event.duration)


def _apply_type_change(entry: EntityTargets, event: Event) -> None:
    params = event.params
    _apply_multi(entry, HeadField.TYPES_GAINED, params.get("types_added"), CARD_TYPES)
    _apply_multi(entry, HeadField.TYPES_LOST, params.get("types_removed"), CARD_TYPES)
    _apply_duration(entry, HeadField.TYPE_COLOR_DURATION, event.duration)


def _apply_color_change(entry: EntityTargets, event: Event) -> None:
    params = event.params
    _apply_multi(entry, HeadField.COLORS_GAINED, params.get("colors"), COLORS)
    _apply_multi(entry, HeadField.COLORS_LOST, params.get("colors_removed"), COLORS)
    _apply_duration(entry, HeadField.TYPE_COLOR_DURATION, event.duration)


def _apply_duration(entry: EntityTargets, name: str, duration: str | None) -> None:
    """How long a lasting change lasts, where the record says; an unrecognised
    or absent duration leaves the field unset rather than guessing a bucket."""
    if duration in DURATIONS:
        entry.set(name, DURATIONS.index(duration))


def _apply_multi(
    entry: EntityTargets, name: str, values, vocabulary: tuple[str, ...],
) -> None:
    for value in values or ():
        index = vocabulary_slot(str(value), vocabulary)
        if index is not None:
            entry.flag(name, index, len(vocabulary))


def _apply_keywords(entry: EntityTargets, params: dict) -> None:
    name = HeadField.KEYWORDS_LOST if params.get("removed") else HeadField.KEYWORDS_GAINED
    for keyword in params.get("keywords") or ():
        _flag_keyword(entry, name, str(keyword))


def _flag_keyword(entry: EntityTargets, name: str, keyword: str) -> None:
    index = overlay_index(keyword)
    if index is not None:
        entry.flag(name, index, len(OVERLAY_KEYWORDS))


def _apply_mana(entry: EntityTargets, mana, *, sign: int) -> None:
    for color, amount in (mana or {}).items():
        key = str(color).strip().upper()
        if key in COLORS:
            entry.add(mana_field(key), sign * float(amount))


def _appliers() -> dict[EventType, Callable[[EntityTargets, Event], None]]:
    """Every event's applier: the tables' events and the ones read off their
    own parameters, each type named once."""
    tabled: list[tuple[EventType, Callable[[EntityTargets, Event], None]]] = [
        *((event_type, partial(
            _apply_additive, name=name, parameter=parameter, default=default, sign=sign,
        )) for event_type, (name, parameter, default, sign) in _ADDITIVE.items()),
        *((event_type, partial(_apply_setter, name=name, value=value))
          for event_type, (name, value) in _SETTERS.items()),
        *((event_type, partial(_apply_zone_outcome, outcome=outcome))
          for event_type, outcome in _ZONE_BY_EVENT.items()),
        *((event_type, partial(_apply_library_event, library_event=library_event))
          for event_type, library_event in _LIBRARY_EVENT_BY_TYPE.items()),
        *_PARAMETER_APPLIERS.items(),
    ]
    appliers = dict(tabled)
    if len(appliers) != len(tabled):
        raise ValueError("an event type has two appliers")
    return appliers


#: The events no table above states, each read off its own parameters.
_PARAMETER_APPLIERS: dict[EventType, Callable[[EntityTargets, Event], None]] = {
    EventType.ZONE_CHANGE: _apply_zone_change,
    EventType.COUNTER_CHANGE: _apply_counter_change,
    EventType.PT_CHANGE: _apply_pt_change,
    EventType.KEYWORD_CHANGE: lambda entry, event: _apply_keywords(entry, event.params),
    EventType.TYPE_CHANGE: _apply_type_change,
    EventType.COLOR_CHANGE: _apply_color_change,
    EventType.MANA_PRODUCED: lambda entry, event: _apply_mana(
        entry, event.params.get("mana_by_color"), sign=1,
    ),
    EventType.MANA_LOST: lambda entry, event: _apply_mana(
        entry, event.params.get("mana_by_color"), sign=-1,
    ),
}

#: ``event type -> how it folds into an entity's targets``.
_EVENT_APPLIERS = _appliers()


def _apply_playability(record: EffectRecord, slot) -> None:
    """Legality bits, which come from the payload rather than from events."""
    payload = record.payload

    def mark(subject: str, name: str, value: int) -> None:
        entry = slot(subject)
        entry.affected = True
        entry.set(name, value)

    if isinstance(payload, PlayabilityDecisionPayload):
        for candidate in payload.candidates:
            for target in candidate.legal_targets:
                mark(target, HeadField.TARGET_LEGAL, 1)
    elif isinstance(payload, PlayabilityAttackersPayload):
        _mark_legality(mark, HeadField.ATTACKER_LEGAL, payload.legal_attackers, payload)
    elif isinstance(payload, PlayabilityBlockersPayload):
        _mark_legality(mark, HeadField.BLOCKER_LEGAL, payload.legal_blockers, payload)
        # min_blockers is read at the anchored attacker, not at the blockers.
        mark(payload.anchor_attacker, HeadField.MIN_BLOCKERS, payload.min_blockers)


def _mark_legality(mark, name: str, legal, payload) -> None:
    """One combat declaration's legal entities as 1, its forbidden ones as 0."""
    for entity in legal:
        mark(entity, name, 1)
    for forbidden in payload.forbidden:
        mark(forbidden.entity, name, 0)


def _apply_continuous(record: EffectRecord, slot) -> None:
    """A continuous record's contributions are its targets directly."""
    payload = record.payload
    if not isinstance(payload, ContinuousPayload):
        return
    for contribution in payload.contributions:
        entry = slot(contribution.entity)
        entry.affected = True
        if contribution.pt_boost != (0, 0):
            entry.add(HeadField.POWER_DELTA, float(contribution.pt_boost[0]))
            entry.add(HeadField.TOUGHNESS_DELTA, float(contribution.pt_boost[1]))
        keywords = contribution.keyword_changes
        for keyword in keywords.gained:
            _flag_keyword(entry, HeadField.KEYWORDS_GAINED, keyword)
        for keyword in keywords.lost:
            _flag_keyword(entry, HeadField.KEYWORDS_LOST, keyword)
        # Whole-class tokens (``all-creature-types``) match no slot and drop.
        for change, vocabulary, gained_field, lost_field in (
            (contribution.type_changes, CARD_TYPES,
             HeadField.TYPES_GAINED, HeadField.TYPES_LOST),
            (contribution.color_changes, COLORS,
             HeadField.COLORS_GAINED, HeadField.COLORS_LOST),
        ):
            _apply_multi(entry, gained_field, change.gained, vocabulary)
            _apply_multi(entry, lost_field, change.lost, vocabulary)
            # The displaced values cannot be named, so the lost field is
            # unknown rather than silently empty.
            if change.sets_outright and lost_field not in entry.fields:
                entry.unknown.add(lost_field)

