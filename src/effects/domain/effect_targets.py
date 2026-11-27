"""What actually happened, as the per-entity head's targets.

The one place the event vocabulary meets the field layout. An entity is
affected when any event names it, not when some field changed: a 0-damage hit
still affected its target. A field no event wrote is absent rather than zero
unless the layout gives it a silent value for this record's class and entity
kind (:meth:`EntityTargets.supervised`); zero everywhere would teach the head
that a destroyed creature was untapped.

A playability record trains no gate (root spec § 11.4): what it observes is
which entities could be chosen, not what an action changed, so its legality
fields train on every entity they name whether or not the gate's target fires.
A cost half trains no gate either: whether an entity was affected is the
effect half's question, and what its payment did trains without the gate at
the entities that paid (:mod:`effects.domain.cost_targets`).

The legality fields are read off the payload in
:mod:`effects.domain.legality_targets`, and where an event Forge names no
subject for lands is :mod:`effects.domain.event_placement`'s;
``target_legality_subjects`` and ``placed_subjects`` are importable from here
too.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from effects.domain.closed_list_misses import closed_list_misses
from effects.domain.continuous_lines import LineChange, abilities_changed, line_changes
from effects.domain.cost_targets import apply_paid_costs, is_cost_half
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
    counter_delta_field,
    mana_field,
)
from effects.domain.event_placement import from_zone, placed_subjects, to_zone
from effects.domain.event_schema import DamageType, Event, EventType, StateFlag, signed_delta
from effects.domain.input_vocabularies import (
    CARD_TYPES,
    DESIGNATIONS,
    FACE_STATES,
    SUPERTYPES,
    vocabulary_slot,
)
from effects.domain.keyword_names import changed_keyword
from effects.domain.lasting_change_labels import (
    cancel_enchant_swaps,
    mixed_pt_durations,
    mixed_type_color_durations,
)
from effects.domain.legality_targets import apply_playability
from effects.domain.legality_targets import target_legality_subjects as target_legality_subjects
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import (
    ContinuousPayload,
    Contribution,
    EffectRecord,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    RewritePayload,
    TriggerPayload,
    events_of,
    is_game_end,
)
from effects.domain.sampling import SamplingClass, sampling_class
from effects.domain.state_snapshot import COLORS, Zone
from effects.domain.surface_builder import has_card_slot

#: The zone outcome each moving event implies; the record keeps destroy and
#: sacrifice apart, the target collapses them.
_ZONE_BY_EVENT: dict[EventType, ZoneOutcome] = {
    EventType.DESTROYED: ZoneOutcome.TO_GRAVEYARD,
    EventType.SACRIFICED: ZoneOutcome.TO_GRAVEYARD,
    EventType.FACE_CHANGE: ZoneOutcome.FACE_CHANGED,
    EventType.SPELL_COUNTERED: ZoneOutcome.COUNTERED,
}

#: A ``zone_change``'s destination as a zone outcome. A move to the
#: battlefield or the library is refined once the whole record is read
#: (:func:`_refine_zone_outcomes`): a blink against any other entry, and the
#: library's top, bottom or shuffle. A destination the table does not name
#: (the stack, the sideboard, a mutated permanent's ``merged`` cards) leaves
#: the outcome unknown: the card moved, so "stayed" would be wrong.
_ZONE_BY_DESTINATION: dict[str, ZoneOutcome] = {
    Zone.GRAVEYARD: ZoneOutcome.TO_GRAVEYARD,
    Zone.EXILE: ZoneOutcome.EXILED,
    Zone.HAND: ZoneOutcome.TO_HAND,
    Zone.LIBRARY: ZoneOutcome.LIBRARY_TOP,
    Zone.BATTLEFIELD: ZoneOutcome.BLINKED,
    Zone.COMMAND: ZoneOutcome.TO_COMMAND,
    #: The collector's spelling of a move to no zone at all.
    "none": ZoneOutcome.CEASED_TO_EXIST,
}
_LIBRARY_EVENT_BY_TYPE: dict[EventType, LibraryEvent] = {
    EventType.LIBRARY_REORDERED: LibraryEvent.REORDER,
    EventType.LIBRARY_SHUFFLED: LibraryEvent.SHUFFLE,
}

@dataclass(frozen=True)
class _Additive:
    """An event that adds to one field: the parameter it reads, the value
    that parameter takes when absent, and the sign the sum takes it with."""

    name: str
    parameter: str
    default: float
    sign: float


#: ``type -> what it adds to``.
_ADDITIVE: dict[EventType, _Additive] = {
    EventType.DAMAGE_HEALED: _Additive(HeadField.DAMAGE_TAKEN, "amount", 0, -1.0),
    EventType.DAMAGE_PREVENTED: _Additive(HeadField.DAMAGE_PREVENTED, "amount", 0, 1.0),
    EventType.POISON_CHANGE: _Additive(HeadField.POISON_DELTA, "delta", 0, 1.0),
    EventType.ENERGY_CHANGE: _Additive(HeadField.ENERGY_DELTA, "delta", 0, 1.0),
    EventType.RADIATION_CHANGE: _Additive(HeadField.RAD_DELTA, "delta", 0, 1.0),
    EventType.CARD_DRAWN: _Additive(HeadField.CARDS_DRAWN, "count", 1, 1.0),
    EventType.CARD_DISCARDED: _Additive(HeadField.CARDS_DISCARDED, "count", 1, 1.0),
    EventType.CARD_MILLED: _Additive(HeadField.CARDS_MILLED, "count", 1, 1.0),
}

_SETTERS: dict[EventType, tuple[str, int]] = {
    EventType.TAPPED: (HeadField.TAP_STATE, 1),
    EventType.UNTAPPED: (HeadField.TAP_STATE, 0),
    EventType.CONTROL_CHANGE: (HeadField.CONTROL_CHANGE, 1),
    EventType.ATTACHED: (HeadField.ATTACHED_TO_CHANGE, 1),
    EventType.UNATTACHED: (HeadField.ATTACHED_TO_CHANGE, 1),
    EventType.REGENERATED: (HeadField.REGENERATED, 1),
    EventType.REMOVED_FROM_COMBAT: (HeadField.REMOVED_FROM_COMBAT, 1),
    EventType.BECAME_BLOCKED: (HeadField.BECAME_BLOCKED, 1),
    EventType.TEXT_CHANGE: (HeadField.TEXT_CHANGED, 1),
    EventType.MONARCH_CHANGED: (HeadField.BECAME_MONARCH, 1),
    EventType.INITIATIVE_TAKEN: (HeadField.TOOK_INITIATIVE, 1),
}

#: The restrictions a ``restriction_change`` names that are designations.
_RESTRICTION_DESIGNATIONS: dict[str, str] = {"goad": "goaded"}

#: ``card_looked_at.params.mode`` -> the library event it is.
_LOOK_MODES: dict[str, LibraryEvent] = {
    "Scry": LibraryEvent.SCRY,
    "Surveil": LibraryEvent.SURVEIL,
    "SearchedLibrary": LibraryEvent.TUTOR,
}

#: ``state_flag_change`` flags that are outputs of their own, each to
#: its binary field; ``class_level`` counts levels gained instead.
_STATE_FLAG_FIELDS: dict[StateFlag, HeadField] = {
    StateFlag.DOOR_UNLOCKED: HeadField.DOOR_UNLOCKED,
    StateFlag.PAIRED: HeadField.PAIRED,
    StateFlag.MERGED: HeadField.MERGED,
    StateFlag.ENTERED_ATTACKING: HeadField.ENTERED_IN_COMBAT,
    StateFlag.ENTERED_BLOCKING: HeadField.ENTERED_IN_COMBAT,
    StateFlag.CITY_BLESSING: HeadField.GAINED_CITY_BLESSING,
    StateFlag.DUNGEON_COMPLETED: HeadField.COMPLETED_DUNGEON,
}


def keyword_index(raw: str) -> int | None:
    """The keyword-output column a keyword, in any spelling Forge writes,
    sets (:func:`~effects.domain.keyword_names.changed_keyword`)."""
    return vocabulary_slot(changed_keyword(raw), KEYWORDS_CHANGED)


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
        self.clear(name, width)
        if 0 <= index < width:
            self.fields[name][index] = 1.0

    def clear(self, name: str, width: int) -> None:
        """Write a multi-binary field with no member set, where none is: the
        record says every member it does not set is unchanged."""
        if not isinstance(self.fields.get(name), list):
            self.fields[name] = [0.0] * width

    def forget(self, name: str) -> None:
        """Declare ``name`` unknown for this entity, dropping what was written."""
        self.fields.pop(name, None)
        self.unknown.add(name)


class RecordTargets(dict[str, EntityTargets]):
    """``entity or player id -> EntityTargets`` for one record, and whether
    the record trains the affected gate (a playability record does not).

    ``unsupervised`` counts, by kind, the events the record carries that no
    output column reads (:func:`unsupervised_kind`): their subjects are
    affected, and nothing else is written for them. It also counts the
    subtypes and keywords the record changes that the closed output lists lack
    (:func:`closed_list_misses.closed_list_misses`), written to no column.
    """

    def __init__(
        self, *, gated: bool = True, record_class: SamplingClass | None = None,
        players: frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        self.gated = gated
        self.unsupervised: Counter[str] = Counter()
        self._record_class = record_class
        self._players = players

    def entry(self, subject: str) -> EntityTargets:
        """The subject's entry, made on first use: a player's at the player
        scope, any other subject's at the permanent scope, each in the
        record's sampling class."""
        entry = self.get(subject)
        if entry is None:
            entry = self[subject] = EntityTargets(
                record_class=self._record_class,
                scope=FieldScope.PLAYER if subject in self._players else FieldScope.PERMANENT,
            )
        return entry


def unsupervised_kind(event: Event) -> str | None:
    """The kind an event is counted under when no output column reads it,
    else None: a change of Forge's hidden keywords (its text, ``CARDNAME can't
    block.``, names no keyword column), a reveal from any zone but a library
    (``library_events`` reads a library's only), a name change and a mana cost
    change (no column in the gen-3 layout)."""
    if event.type is EventType.KEYWORD_CHANGE and event.params.get("hidden"):
        return "hidden_keyword_change"
    if event.type is EventType.CARD_REVEALED and not _from_a_library(event):
        return "card_revealed_off_library"
    if event.type in (EventType.NAME_CHANGE, EventType.MANA_COST_CHANGE):
        return event.type.value
    return None


def derive_targets(record: EffectRecord) -> RecordTargets:
    """``entity or player id -> EntityTargets`` for one record; none for the
    game-end marker, which names nobody and trains no gate."""
    if is_game_end(record):
        return RecordTargets(gated=False)
    targets = RecordTargets(
        gated=_trains_gate(record), record_class=sampling_class(record),
        players=frozenset(player.id for player in record.state.players),
    )
    events = _target_events(record)
    for event in events:
        kind = unsupervised_kind(event)
        if kind is not None:
            targets.unsupervised[kind] += 1
        for subject in event.subjects:
            entry = targets.entry(subject)
            entry.affected = True
            _apply_event(entry, event)

    if events:
        _RecordPass(record, events, targets).run()
    apply_playability(record, targets.entry)
    _apply_continuous(record, targets.entry)
    apply_paid_costs(record, targets.entry)
    targets.unsupervised.update(closed_list_misses(record))
    _drop_slotless(record, targets)
    return targets


def _drop_slotless(record: EffectRecord, targets: RecordTargets) -> None:
    """A snapshot entity with no ``[CARD]`` slot, a library card below the
    top, trains no gate and no field; its player's draw, mill and library
    events stay (:func:`surface_builder.has_card_slot`)."""
    for entity in record.state.entities:
        if not has_card_slot(entity):
            targets.pop(entity.id, None)


def _trains_gate(record: EffectRecord) -> bool:
    """Whether the record trains the affected gate: neither a playability
    record nor a cost half, nor a rewrite that cannot say who its
    replacement left unaffected."""
    return not (
        _is_playability(record) or is_cost_half(record) or _outcome_partly_unknown(record))


def _outcome_partly_unknown(record: EffectRecord) -> bool:
    """Whether a rewrite record cannot say who its replacement left
    unaffected, so it trains no gate: what came in is unknown
    (``incoming_unknown``: what changed is unknown), or the ability it set
    aside ran and another record holds that run's events. The events it does
    hold (its own outgoing event, a run it holds) are still supervised."""
    payload = record.payload
    if not isinstance(payload, RewritePayload):
        return False
    return payload.incoming_unknown or (
        bool(payload.buffered_ran) and not payload.holds_buffered_run)


def _is_playability(record: EffectRecord) -> bool:
    return isinstance(record.payload, (
        PlayabilityDecisionPayload, PlayabilityAttackersPayload, PlayabilityBlockersPayload,
    ))


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


def _apply_additive(entry: EntityTargets, event: Event, *, additive: _Additive) -> None:
    entry.add(
        additive.name,
        additive.sign * float(event.params.get(additive.parameter, additive.default)),
    )


def _apply_setter(entry: EntityTargets, _event: Event, *, name: str, value: int) -> None:
    entry.set(name, value)


def _apply_zone_outcome(
    entry: EntityTargets, _event: Event, *, outcome: ZoneOutcome,
) -> None:
    entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(outcome))


def _apply_library_event(
    entry: EntityTargets, _event: Event, *, library_event: LibraryEvent,
) -> None:
    _flag_library(entry, library_event)


def _apply_card_revealed(entry: EntityTargets, event: Event) -> None:
    """A reveal names the player who revealed; only a library's reveal is a
    library event, a hand's (Duress) is none."""
    if _from_a_library(event):
        _flag_library(entry, LibraryEvent.REVEAL)


def _from_a_library(event: Event) -> bool:
    return from_zone(event) == Zone.LIBRARY


def _move_is_the_outcome(entry: EntityTargets, destination: str) -> bool:
    """Whether the entity's zone outcome is still the one a move to
    ``destination`` wrote, a destination the outcomes name."""
    outcome = _ZONE_BY_DESTINATION.get(destination)
    if outcome is None:
        return False
    return entry.fields.get(HeadField.ZONE_OUTCOME) == ZONE_OUTCOMES.index(outcome)


def _is_library_bottom(position: int, size: int | None) -> bool:
    """Whether a card put ``position`` cards deep into a library of ``size``
    went to the bottom: a negative position counts from the bottom, 0 is the
    top, and a position at or past the library's size (or any but 0, where
    the size is unknown) is the bottom."""
    if position < 0:
        return True
    return position > 0 and (size is None or position >= size)


def _flag_library(entry: EntityTargets, library_event: LibraryEvent) -> None:
    entry.flag(
        HeadField.LIBRARY_EVENTS, LIBRARY_EVENTS.index(library_event), len(LIBRARY_EVENTS),
    )


def _apply_zone_change(entry: EntityTargets, event: Event) -> None:
    outcome = _ZONE_BY_DESTINATION.get(to_zone(event))
    if outcome is None:
        entry.forget(HeadField.ZONE_OUTCOME)
        return
    entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(outcome))


def _apply_life_change(entry: EntityTargets, event: Event) -> None:
    entry.add(HeadField.LIFE_DELTA, signed_delta(event))


#: Damage dealt as -1/-1 counters (wither, infect to a creature): the record's
#: own ``counter_change`` adds them, and no damage is marked.


def _apply_damage_dealt(entry: EntityTargets, event: Event) -> None:
    """Marked damage, never the damage dealt as -1/-1 counters
    (:meth:`_RecordPass._damage_as_counters` takes a deathtouch witherer's out)."""
    if event.params.get("damage_type") == DamageType.M1M1_COUNTERS:
        return
    entry.add(HeadField.DAMAGE_TAKEN, float(event.params.get("amount", 0)))


def _apply_counter_change(entry: EntityTargets, event: Event) -> None:
    counter = str(event.params.get("counter_type", OTHER_COUNTER))
    entry.add(counter_delta_field(counter), signed_delta(event))


def _apply_pt_change(entry: EntityTargets, event: Event) -> None:
    entry.add(HeadField.POWER_DELTA, float(event.params.get("power_delta", 0)))
    entry.add(HeadField.TOUGHNESS_DELTA, float(event.params.get("toughness_delta", 0)))
    _apply_duration(entry, HeadField.PT_DURATION, event.duration)


def _apply_type_change(entry: EntityTargets, event: Event) -> None:
    params = event.params
    for gained, lost, vocabulary in _TYPE_CHANNELS:
        _apply_multi(entry, gained, params.get("types_added"), vocabulary)
        _apply_multi(entry, lost, params.get("types_removed"), vocabulary)
    _apply_duration(entry, HeadField.TYPE_COLOR_DURATION, event.duration)


#: A type change's three readings: the card types, the subtypes of the
#: closed list and the supertypes, each a gained and a lost field.
_TYPE_CHANNELS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (HeadField.TYPES_GAINED, HeadField.TYPES_LOST, CARD_TYPES),
    (HeadField.SUBTYPES_GAINED, HeadField.SUBTYPES_LOST, SUBTYPES_CHANGED),
    (HeadField.SUPERTYPES_GAINED, HeadField.SUPERTYPES_LOST, SUPERTYPES),
)


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
    """Keywords gained or lost; a hidden keyword's text names no column
    (:func:`unsupervised_kind`)."""
    if params.get("hidden"):
        return
    name = HeadField.KEYWORDS_LOST if params.get("removed") else HeadField.KEYWORDS_GAINED
    for keyword in params.get("keywords") or ():
        _flag_keyword(entry, name, str(keyword))


def _flag_keyword(entry: EntityTargets, name: str, keyword: str) -> None:
    index = keyword_index(keyword)
    if index is not None:
        entry.flag(name, index, len(KEYWORDS_CHANGED))


def _flag_designation(entry: EntityTargets, designation: str, gained: bool) -> None:
    index = vocabulary_slot(designation, DESIGNATIONS)
    if index is not None:
        name = HeadField.DESIGNATIONS_GAINED if gained else HeadField.DESIGNATIONS_LOST
        entry.flag(name, index, len(DESIGNATIONS))


def _apply_phased(entry: EntityTargets, event: Event) -> None:
    """Phasing out and in are two outcomes, and the phased-out designation
    comes and goes with them."""
    out = bool(event.params.get("out", True))
    outcome = ZoneOutcome.PHASED_OUT if out else ZoneOutcome.PHASED_IN
    entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(outcome))
    _flag_designation(entry, "phased_out", gained=out)


def _apply_face_change(entry: EntityTargets, event: Event) -> None:
    entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(ZoneOutcome.FACE_CHANGED))
    face = vocabulary_slot(str(event.params.get("to_state") or ""), FACE_STATES)
    if face is not None:
        entry.set(HeadField.FACE_AFTER, face)


def _apply_restriction(entry: EntityTargets, event: Event) -> None:
    designation = _RESTRICTION_DESIGNATIONS.get(str(event.params.get("restriction", "")))
    if designation is not None:
        _flag_designation(entry, designation, gained=bool(event.params.get("value", True)))


def _apply_state_flag(entry: EntityTargets, event: Event) -> None:
    flag = str(event.params.get("flag", ""))
    if flag == StateFlag.CLASS_LEVEL:
        entry.add(HeadField.CLASS_LEVELS_GAINED, 1.0)
        return
    field = _STATE_FLAG_FIELDS.get(flag)
    if field is not None:
        entry.set(field, 1.0 if event.params.get("value", True) else 0.0)
        return
    _flag_designation(entry, flag, gained=bool(event.params.get("value", True)))


def _apply_ability_change(entry: EntityTargets, event: Event) -> None:
    """An animation's grants (a count of the named abilities) and whether it
    removed every other ability."""
    if event.params.get("removed"):
        entry.set(HeadField.ABILITIES_LOST, 1)
    granted = [name for name in str(event.params.get("abilities") or "").split(",") if name.strip()]
    if granted:
        entry.add(HeadField.ABILITIES_GAINED, float(len(granted)))


def _apply_speed(entry: EntityTargets, event: Event) -> None:
    entry.add(HeadField.SPEED_GAINED, max(float(event.params.get("delta", 0)), 0.0))


def _apply_mana(entry: EntityTargets, mana, *, sign: int) -> None:
    for color, amount in (mana or {}).items():
        key = str(color).strip().upper()
        if key in COLORS:
            entry.add(mana_field(key), sign * float(amount))


def _appliers() -> dict[EventType, Callable[[EntityTargets, Event], None]]:
    """Every event's applier: the tables' events and the ones read off their
    own parameters, each type named once."""
    tabled: list[tuple[EventType, Callable[[EntityTargets, Event], None]]] = [
        *((event_type, partial(_apply_additive, additive=additive))
          for event_type, additive in _ADDITIVE.items()),
        *((event_type, partial(_apply_setter, name=name, value=value))
          for event_type, (name, value) in _SETTERS.items()),
        *((event_type, partial(_apply_zone_outcome, outcome=outcome))
          for event_type, outcome in _ZONE_BY_EVENT.items()
          if event_type not in _PARAMETER_APPLIERS),
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
    EventType.DAMAGE_DEALT: _apply_damage_dealt,
    EventType.LIFE_CHANGE: _apply_life_change,
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
    EventType.PHASED: _apply_phased,
    EventType.FACE_CHANGE: _apply_face_change,
    EventType.RESTRICTION_CHANGE: _apply_restriction,
    EventType.STATE_FLAG_CHANGE: _apply_state_flag,
    EventType.ABILITY_CHANGE: _apply_ability_change,
    EventType.SPEED_CHANGED: _apply_speed,
    EventType.CARD_REVEALED: _apply_card_revealed,
}

#: ``event type -> how it folds into an entity's targets``.
_EVENT_APPLIERS = _appliers()


class _RecordPass:
    """What one event cannot say alone: read against the record's other
    events and its snapshot, once every event has been applied."""

    def __init__(self, record: EffectRecord, events: tuple[Event, ...],
                 targets: RecordTargets) -> None:
        self.record = record
        self.events = events
        self.targets = targets
        self.entities = {entity.id: entity for entity in record.state.entities}
        self.players = {player.id: player for player in record.state.players}
        self.by_type: dict[EventType, list[Event]] = defaultdict(list)
        for event in events:
            self.by_type[event.type].append(event)
        #: Whose library each shuffle shuffled.
        self.shuffled = {
            subject for event in self.by_type[EventType.LIBRARY_SHUFFLED]
            for subject in event.subjects
        }
        #: Fields an unplaceable event leaves unknown on every entity.
        self.unknown: set[str] = set()

    def run(self) -> None:
        self._damage_as_counters()
        self._zone_outcomes()
        self._library_events()
        self._base_pt()
        mixed_pt_durations(self.by_type[EventType.PT_CHANGE], self.targets)
        mixed_type_color_durations(
            [*self.by_type[EventType.TYPE_CHANGE], *self.by_type[EventType.COLOR_CHANGE]],
            self.targets,
        )
        cancel_enchant_swaps(self.targets.values())
        self._copies()
        self._venture()
        for name in self.unknown:
            for entry in self.targets.values():
                entry.forget(name)

    # ── damage dealt as counters ──

    def _damage_as_counters(self) -> None:
        """A deathtouch source with wither or infect reads ``deathtouch``
        (Forge's damage type), yet its damage to a creature is -1/-1 counters,
        which the record's ``counter_change`` adds: each such hit matched by a
        -1/-1 counter change of its amount on its subject is taken back out of
        the marked damage."""
        counters = self._m1m1_counters()
        for event in self.by_type[EventType.DAMAGE_DEALT]:
            if event.params.get("damage_type") == DamageType.DEATHTOUCH:
                self._take_back_counters(event, counters)

    def _m1m1_counters(self) -> dict[str, list[float]]:
        """``subject -> the -1/-1 counter changes the record makes on it``, in
        event order."""
        counters: dict[str, list[float]] = defaultdict(list)
        m1m1 = (
            event for event in self.by_type[EventType.COUNTER_CHANGE]
            if str(event.params.get("counter_type", "")) == "M1M1"
        )
        for event in m1m1:
            for subject in event.subjects:
                counters[subject].append(signed_delta(event))
        return counters

    def _take_back_counters(self, event: Event, counters: dict[str, list[float]]) -> None:
        """One deathtouch hit's damage taken back out of the marked damage at
        each subject holding a -1/-1 counter change of its amount, using that
        change up."""
        amount = float(event.params.get("amount", 0))
        if not amount:
            return
        for subject in event.subjects:
            if amount in counters[subject]:
                counters[subject].remove(amount)
                self.targets[subject].add(HeadField.DAMAGE_TAKEN, -amount)

    # ── zone outcomes ──

    def _zone_outcomes(self) -> None:
        """Refine each subject's last move: onto the battlefield is a blink
        only after leaving it in the same record, else it entered; into a
        library is the top, the bottom or a shuffle, and unknown where the
        record cannot say which."""
        for subject, history in self._moves().items():
            self._refine_move(subject, history)

    def _moves(self) -> dict[str, list[Event]]:
        """``subject -> its zone changes``, in event order."""
        moves: dict[str, list[Event]] = defaultdict(list)
        for event in self.by_type[EventType.ZONE_CHANGE]:
            for subject in event.subjects:
                moves[subject].append(event)
        return moves

    def _refine_move(self, subject: str, history: list[Event]) -> None:
        """One subject's last move refined, unless its destination is unnamed
        or a later event (destroyed, countered, phased) said more."""
        entry = self.targets[subject]
        last = history[-1]
        destination = to_zone(last)
        if not _move_is_the_outcome(entry, destination):
            return
        if destination == Zone.BATTLEFIELD:
            self._entered(entry, subject, history)
        elif destination == Zone.LIBRARY:
            self._into_library(entry, subject, last)

    def _entered(self, entry: EntityTargets, subject: str, history: list[Event]) -> None:
        left = any(from_zone(event) == Zone.BATTLEFIELD for event in history[:-1])
        if not left:
            entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(ZoneOutcome.ENTERED))
        self._face_cast_with(entry, subject, history[-1])

    def _face_cast_with(self, entry: EntityTargets, subject: str, move: Event) -> None:
        """A spell enters with the face it was cast with; from anywhere else
        the record does not say which face is up. A face an event named
        stays."""
        if HeadField.FACE_AFTER in entry.fields or from_zone(move) != Zone.STACK:
            return
        entity = self.entities.get(subject)
        if entity is None or entity.face_state is None:
            return
        face = vocabulary_slot(entity.face_state, FACE_STATES)
        if face is not None:
            entry.set(HeadField.FACE_AFTER, face)

    def _into_library(self, entry: EntityTargets, subject: str, move: Event) -> None:
        outcome = self._library_outcome(subject, move)
        if outcome is None:
            entry.forget(HeadField.ZONE_OUTCOME)
            return
        entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(outcome))

    def _library_outcome(self, subject: str, move: Event) -> ZoneOutcome | None:
        """Where in its owner's library a card went: the position the move
        names, else shuffled in where its owner's library was shuffled (any
        shuffle, for a card whose owner the snapshot does not hold); None
        where the record cannot say."""
        owner = self._owner(subject)
        position = move.params.get("library_position")
        if position is not None:
            return self._placed_in_library(owner, int(position))
        if self._shuffled(owner):
            return ZoneOutcome.SHUFFLED_INTO_LIBRARY
        return None

    def _placed_in_library(self, owner: str | None, position: int) -> ZoneOutcome:
        size = self.players[owner].library if owner in self.players else None
        if _is_library_bottom(position, size):
            return ZoneOutcome.LIBRARY_BOTTOM
        return ZoneOutcome.LIBRARY_TOP

    def _shuffled(self, owner: str | None) -> bool:
        """Whether ``owner``'s library was shuffled; any library, for an owner
        the snapshot does not name."""
        if owner is None:
            return bool(self.shuffled)
        return owner in self.shuffled

    def _owner(self, subject: str) -> str | None:
        entity = self.entities.get(subject)
        if entity is None:
            return None
        return entity.owner or entity.controller

    # ── library events ──

    def _library_events(self) -> None:
        """A ``card_looked_at`` names its library action in ``params.mode``
        (``Scry``, ``Surveil``, or ``SearchedLibrary``: a tutor). A record
        without the mode reads it from what else happened: a scry and a
        surveil both look at and reorder cards, a surveil is the one that also
        mills. A tutor is the named mode only: a shuffle after cards left the
        library is no evidence of a search (a mill, a draw or an exile from
        the top shuffles too)."""
        milled = {
            subject for event in self.by_type[EventType.CARD_MILLED]
            for subject in event.subjects
        }
        for event in self.by_type[EventType.CARD_LOOKED_AT]:
            named = _LOOK_MODES.get(str(event.params.get("mode") or ""))
            for subject in event.subjects:
                kind = named or (LibraryEvent.SURVEIL if subject in milled else LibraryEvent.SCRY)
                _flag_library(self.targets[subject], kind)

    # ── base power and toughness ──

    def _base_pt(self) -> None:
        """A P/T-setting effect's ``set_pt`` against the snapshot's base: the
        base delta. Without ``set_pt`` the record does not say whether the
        base moved or a boost did, so the field stays unwritten."""
        for event in self.by_type[EventType.PT_CHANGE]:
            set_pt = event.params.get("set_pt")
            if _is_pt_pair(set_pt):
                self._set_base_pt(event.subjects, set_pt)

    def _set_base_pt(self, subjects, set_pt) -> None:
        for subject in subjects:
            base = _snapshot_base(self.entities.get(subject))
            entry = self.targets[subject]
            entry.set(HeadField.BASE_POWER_DELTA, float(set_pt[0]) - base[0])
            entry.set(HeadField.BASE_TOUGHNESS_DELTA, float(set_pt[1]) - base[1])

    # ── copies and retargets ──

    def _copies(self) -> None:
        """Copies and retargets, placed on the object they happened to.

        Forge names no subject for these: a Clone's copy is placed on the
        Clone, which is the cause and enters from the stack in the same
        record; a spell copy or a retarget on the stack objects the acting
        ability targets, else on its cause when that is a stack object. An
        event placed nowhere leaves its field unknown in the record."""
        for event in self.by_type[EventType.PERMANENT_COPIED]:
            self._place(placed_subjects(self.record, event, self.events), HeadField.COPIED,
                        lambda entry: entry.set(HeadField.COPIED, 1))
        for event in self.by_type[EventType.SPELL_COPIED]:
            count = float(event.params.get("count", 1) or 1)
            self._place(
                placed_subjects(self.record, event, self.events), HeadField.SPELL_COPIES,
                lambda entry, count=count: entry.add(HeadField.SPELL_COPIES, count),
            )
        for event in self.by_type[EventType.TARGETS_CHANGED]:
            self._place(
                placed_subjects(self.record, event, self.events), HeadField.TARGETS_CHANGED,
                lambda entry: entry.set(HeadField.TARGETS_CHANGED, 1),
            )

    def _place(self, subjects, name: str, write) -> None:
        if not subjects:
            self.unknown.add(name)
            return
        for subject in subjects:
            entry = self.targets.entry(subject)
            entry.affected = True
            write(entry)

    # ── players ──

    def _venture(self) -> None:
        for event in self.by_type[EventType.DUNGEON_VENTURED]:
            self._place(placed_subjects(self.record, event, self.events), HeadField.VENTURED,
                        lambda entry: entry.set(HeadField.VENTURED, 1))


def _apply_continuous(record: EffectRecord, slot) -> None:
    """A continuous record's contributions are its targets directly, read
    against its board, which is the board without the static."""
    payload = record.payload
    if not isinstance(payload, ContinuousPayload):
        return
    entities = {entity.id: entity for entity in record.state.entities}
    for contribution in payload.contributions:
        entry = slot(contribution.entity)
        entry.affected = True
        _apply_contribution(entry, contribution, entities.get(contribution.entity))


def _apply_contribution(entry: EntityTargets, contribution: Contribution, entity) -> None:
    if contribution.pt_boost != (0, 0):
        entry.add(HeadField.POWER_DELTA, float(contribution.pt_boost[0]))
        entry.add(HeadField.TOUGHNESS_DELTA, float(contribution.pt_boost[1]))
    _apply_set_pt(entry, contribution.set_pt, entity)
    if contribution.controller is not None:
        entry.set(HeadField.CONTROL_CHANGE, 1)
    granted, taken = abilities_changed(contribution)
    if granted:
        entry.add(HeadField.ABILITIES_GAINED, float(granted))
    if taken:
        entry.set(HeadField.ABILITIES_LOST, 1)
    for line in line_changes(contribution, entity):
        _apply_channel(entry, line)


def _apply_set_pt(entry: EntityTargets, set_pt, entity) -> None:
    """Against the board without the static, as a resolution's ``set_pt`` is
    against its snapshot: no base on a permanent that was no creature there is
    a base of 0/0."""
    if set_pt is None:
        return
    base = _snapshot_base(entity)
    for half, field_name in ((0, HeadField.BASE_POWER_DELTA), (1, HeadField.BASE_TOUGHNESS_DELTA)):
        if set_pt[half] is not None:
            entry.set(field_name, float(set_pt[half]) - base[half])


def _snapshot_base(entity) -> tuple[int, int]:
    """An entity's base power and toughness on the snapshot; 0/0 for one
    the snapshot does not hold or that was no creature there."""
    if entity is None or entity.pt is None:
        return (0, 0)
    return entity.pt.base


def _is_pt_pair(set_pt) -> bool:
    """Whether a ``set_pt`` names both a power and a toughness."""
    return isinstance(set_pt, (list, tuple)) and len(set_pt) == 2


def _apply_channel(entry: EntityTargets, line: LineChange) -> None:
    """One line's gained and lost values; a line the static replaces names
    every value it displaced, so its lost field is written even where empty."""
    _apply_multi(entry, line.gained_field, line.gained, line.vocabulary)
    if line.gained_unknown:
        entry.forget(line.gained_field)
    _apply_multi(entry, line.lost_field, line.lost, line.vocabulary)
    if line.lost_unknown:
        if line.lost_field not in entry.fields:
            entry.unknown.add(line.lost_field)
    elif line.replaces:
        entry.clear(line.lost_field, len(line.vocabulary))
