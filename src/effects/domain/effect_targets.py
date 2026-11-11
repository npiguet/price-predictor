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
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from effects.domain.effect_head_input import (
    CARD_TYPES,
    DESIGNATIONS,
    FACE_STATES,
    SUPERTYPES,
    vocabulary_slot,
)
from effects.domain.effect_head_layout import (
    DURATIONS,
    LIBRARY_EVENTS,
    OTHER_COUNTER,
    OUTPUT_COUNTER_TYPES,
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
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
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
from effects.domain.state_snapshot import COLORS, Zone, normalize_counter_type

#: The zone outcome each moving event implies; the record keeps destroy and
#: sacrifice apart, the target collapses them.
_ZONE_BY_EVENT: dict[EventType, ZoneOutcome] = {
    EventType.DESTROYED: ZoneOutcome.DIED,
    EventType.SACRIFICED: ZoneOutcome.DIED,
    EventType.FACE_CHANGE: ZoneOutcome.FACE_CHANGED,
    EventType.SPELL_COUNTERED: ZoneOutcome.COUNTERED,
}

#: A ``zone_change``'s destination as a zone outcome. A move to the
#: battlefield or the library is refined once the whole record is read
#: (:func:`_refine_zone_outcomes`): a blink against any other entry, and the
#: library's top, bottom or shuffle.
_ZONE_BY_DESTINATION: dict[str, ZoneOutcome] = {
    Zone.GRAVEYARD: ZoneOutcome.DIED,
    Zone.EXILE: ZoneOutcome.EXILED,
    Zone.HAND: ZoneOutcome.TO_HAND,
    Zone.LIBRARY: ZoneOutcome.LIBRARY_TOP,
    Zone.BATTLEFIELD: ZoneOutcome.BLINKED,
    Zone.COMMAND: ZoneOutcome.TO_COMMAND,
    #: The collector's spelling of a move to no zone at all.
    "none": ZoneOutcome.CEASED_TO_EXIST,
}
#: The outcome of a ``zone_change`` to a destination the table does not name.
_STAYED = ZoneOutcome.STAYED

_LIBRARY_EVENT_BY_TYPE: dict[EventType, LibraryEvent] = {
    EventType.LIBRARY_REORDERED: LibraryEvent.REORDER,
    EventType.CARD_REVEALED: LibraryEvent.REVEAL,
    EventType.LIBRARY_SHUFFLED: LibraryEvent.SHUFFLE,
}

#: ``type -> (field, parameter, default when absent, sign)``.
_ADDITIVE: dict[EventType, tuple[str, str, float, float]] = {
    EventType.DAMAGE_DEALT: (HeadField.DAMAGE_TAKEN, "amount", 0, 1.0),
    EventType.DAMAGE_HEALED: (HeadField.DAMAGE_TAKEN, "amount", 0, -1.0),
    EventType.DAMAGE_PREVENTED: (HeadField.DAMAGE_PREVENTED, "amount", 0, 1.0),
    EventType.LIFE_CHANGE: (HeadField.LIFE_DELTA, "delta", 0, 1.0),
    EventType.POISON_CHANGE: (HeadField.POISON_DELTA, "delta", 0, 1.0),
    EventType.ENERGY_CHANGE: (HeadField.ENERGY_DELTA, "delta", 0, 1.0),
    EventType.RADIATION_CHANGE: (HeadField.RAD_DELTA, "delta", 0, 1.0),
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
    EventType.REGENERATED: (HeadField.REGENERATED, 1),
    EventType.REMOVED_FROM_COMBAT: (HeadField.REMOVED_FROM_COMBAT, 1),
    EventType.BECAME_BLOCKED: (HeadField.BECAME_BLOCKED, 1),
    EventType.TEXT_CHANGE: (HeadField.TEXT_CHANGED, 1),
    EventType.MONARCH_CHANGED: (HeadField.BECAME_MONARCH, 1),
    EventType.INITIATIVE_TAKEN: (HeadField.TOOK_INITIATIVE, 1),
    EventType.PLAYER_LOST: (HeadField.GAME_LOST, 1),
}

#: The restrictions a ``restriction_change`` names that are designations.
_RESTRICTION_DESIGNATIONS: dict[str, str] = {"goad": "goaded"}

#: ``card_looked_at.params.mode`` -> the library event it is (T123d).
_LOOK_MODES: dict[str, LibraryEvent] = {
    "Scry": LibraryEvent.SCRY,
    "Surveil": LibraryEvent.SURVEIL,
    "SearchedLibrary": LibraryEvent.TUTOR,
}

#: ``state_flag_change`` flags that are outputs of their own (T123g), each to
#: its binary field; ``class_level`` counts levels gained instead.
_STATE_FLAG_FIELDS: dict[str, HeadField] = {
    "door_unlocked": HeadField.DOOR_UNLOCKED,
    "paired": HeadField.PAIRED,
    "merged": HeadField.MERGED,
    "entered_attacking": HeadField.ENTERED_IN_COMBAT,
    "entered_blocking": HeadField.ENTERED_IN_COMBAT,
    "city_blessing": HeadField.GAINED_CITY_BLESSING,
    "dungeon_completed": HeadField.COMPLETED_DUNGEON,
}


def keyword_index(raw: str) -> int | None:
    """The keyword-output column a keyword, in any spelling Forge writes, sets."""
    return vocabulary_slot(overlay_keyword(raw), KEYWORDS_CHANGED)


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

    def forget(self, name: str) -> None:
        """Declare ``name`` unknown for this entity, dropping what was written."""
        self.fields.pop(name, None)
        self.unknown.add(name)


class RecordTargets(dict[str, EntityTargets]):
    """``entity or player id -> EntityTargets`` for one record, and whether
    the record trains the affected gate (a playability record does not)."""

    def __init__(self, *, gated: bool = True) -> None:
        super().__init__()
        self.gated = gated


def derive_targets(record: EffectRecord) -> RecordTargets:
    """``entity or player id -> EntityTargets`` for one record."""
    record_class = sampling_class(record)
    targets = RecordTargets(gated=not _is_playability(record))
    players = {player.id for player in record.state.players}

    def slot(subject: str) -> EntityTargets:
        entry = targets.get(subject)
        if entry is None:
            entry = targets[subject] = EntityTargets(
                record_class=record_class,
                scope=FieldScope.PLAYER if subject in players else FieldScope.PERMANENT,
            )
        return entry

    events = _target_events(record)
    for event in events:
        for subject in event.subjects:
            entry = slot(subject)
            entry.affected = True
            _apply_event(entry, event)

    if events:
        _RecordPass(record, events, slot, targets).run()
    _apply_playability(record, slot)
    _apply_continuous(record, slot)
    return targets


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
    _flag_library(entry, library_event)


def _flag_library(entry: EntityTargets, library_event: LibraryEvent) -> None:
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
    counter = normalize_counter_type(str(event.params.get("counter_type", OTHER_COUNTER)))
    name = (
        counter_field(counter)
        if counter in _OUTPUT_COUNTERS
        else HeadField.COUNTERS_DELTA_OTHER
    )
    entry.add(name, float(event.params.get("delta", 0)))


_OUTPUT_COUNTERS = frozenset(OUTPUT_COUNTER_TYPES)


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


def _displaceable(record: EffectRecord, entity) -> dict[str, tuple[str, ...]] | None:
    """Each lost field's values on the board a continuous record's static
    displaces, keyed by the lost field; None where that board is not the one
    without the static (before gen-3) or the entity is not on it."""
    if entity is None or record.generation is None or record.generation < 3:
        return None
    return {
        HeadField.TYPES_LOST: entity.types,
        HeadField.SUBTYPES_LOST: entity.subtypes,
        HeadField.SUPERTYPES_LOST: entity.supertypes,
        HeadField.COLORS_LOST: entity.colors,
    }


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
    if flag == "class_level":
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
        *((event_type, partial(
            _apply_additive, name=name, parameter=parameter, default=default, sign=sign,
        )) for event_type, (name, parameter, default, sign) in _ADDITIVE.items()),
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
}

#: ``event type -> how it folds into an entity's targets``.
_EVENT_APPLIERS = _appliers()


class _RecordPass:
    """What one event cannot say alone: read against the record's other
    events and its snapshot, once every event has been applied."""

    def __init__(self, record: EffectRecord, events: tuple[Event, ...], slot,
                 targets: RecordTargets) -> None:
        self.record = record
        self.events = events
        self.slot = slot
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
        self._zone_outcomes()
        self._library_events()
        self._base_pt()
        self._copies()
        self._game_end()
        self._venture()
        for name in self.unknown:
            for entry in self.targets.values():
                entry.forget(name)

    # ── zone outcomes ──

    def _zone_outcomes(self) -> None:
        """Refine each subject's last move: onto the battlefield is a blink
        only after leaving it in the same record, else it entered; into a
        library is the top, the bottom or a shuffle, and unknown where the
        record cannot say which."""
        moves: dict[str, list[Event]] = defaultdict(list)
        for event in self.by_type[EventType.ZONE_CHANGE]:
            for subject in event.subjects:
                moves[subject].append(event)
        for subject, history in moves.items():
            entry = self.targets[subject]
            last = history[-1]
            destination = str(last.params.get("to_zone", "")).lower()
            written = entry.fields.get(HeadField.ZONE_OUTCOME)
            if written != ZONE_OUTCOMES.index(_ZONE_BY_DESTINATION.get(destination, _STAYED)):
                continue  # a later event (destroyed, countered, phased) said more
            if destination == Zone.BATTLEFIELD:
                self._entered(entry, subject, history)
            elif destination == Zone.LIBRARY:
                self._into_library(entry, subject, last)

    def _entered(self, entry: EntityTargets, subject: str, history: list[Event]) -> None:
        left = any(
            str(event.params.get("from_zone", "")).lower() == Zone.BATTLEFIELD
            for event in history[:-1]
        )
        if not left:
            entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(ZoneOutcome.ENTERED))
        # A spell enters with the face it was cast with; from anywhere else
        # the record does not say which face is up.
        entity = self.entities.get(subject)
        source = str(history[-1].params.get("from_zone", "")).lower()
        if (
            HeadField.FACE_AFTER not in entry.fields and entity is not None
            and source == Zone.STACK and entity.face_state is not None
        ):
            face = vocabulary_slot(entity.face_state, FACE_STATES)
            if face is not None:
                entry.set(HeadField.FACE_AFTER, face)

    def _into_library(self, entry: EntityTargets, subject: str, move: Event) -> None:
        owner = self._owner(subject)
        position = move.params.get("library_position")
        if position is not None:
            size = self.players[owner].library if owner in self.players else None
            position = int(position)
            bottom = position < 0 or (position > 0 and (size is None or position >= size))
            outcome = ZoneOutcome.LIBRARY_BOTTOM if bottom else ZoneOutcome.LIBRARY_TOP
        elif owner in self.shuffled or (owner is None and self.shuffled):
            outcome = ZoneOutcome.SHUFFLED_INTO_LIBRARY
        else:
            entry.forget(HeadField.ZONE_OUTCOME)
            return
        entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(outcome))

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
        mills, and a shuffle after cards left the library is a search."""
        milled = {
            subject for event in self.by_type[EventType.CARD_MILLED]
            for subject in event.subjects
        }
        for event in self.by_type[EventType.CARD_LOOKED_AT]:
            named = _LOOK_MODES.get(str(event.params.get("mode") or ""))
            for subject in event.subjects:
                kind = named or (LibraryEvent.SURVEIL if subject in milled else LibraryEvent.SCRY)
                _flag_library(self.targets[subject], kind)
        searched = set()
        for event in self.by_type[EventType.ZONE_CHANGE]:
            if str(event.params.get("from_zone", "")).lower() != Zone.LIBRARY:
                continue
            if str(event.params.get("to_zone", "")).lower() == Zone.LIBRARY:
                continue
            for subject in event.subjects:
                searched.add(self._owner(subject))
        for player in self.shuffled:
            if player in searched or (None in searched and player in self.targets):
                _flag_library(self.targets[player], LibraryEvent.TUTOR)

    # ── base power and toughness ──

    def _base_pt(self) -> None:
        """A P/T-setting effect's ``set_pt`` against the snapshot's base: the
        base delta. Without ``set_pt`` the record does not say whether the
        base moved or a boost did, so the field stays unwritten."""
        for event in self.by_type[EventType.PT_CHANGE]:
            set_pt = event.params.get("set_pt")
            if not isinstance(set_pt, (list, tuple)) or len(set_pt) != 2:
                continue
            for subject in event.subjects:
                entity = self.entities.get(subject)
                base = entity.pt.base if entity is not None and entity.pt is not None else (0, 0)
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
            entry = self.slot(subject)
            entry.affected = True
            write(entry)

    # ── players ──

    def _game_end(self) -> None:
        """A win is the other players' loss, and a draw is every player's
        (root spec § 11.2): each player of a record carrying the game's end
        is affected and supervised."""
        winners = {
            subject for event in self.by_type[EventType.PLAYER_WON]
            for subject in event.subjects
        }
        drawn = bool(self.by_type[EventType.GAME_DRAWN])
        if not winners and not drawn:
            return
        for player in self.players:
            entry = self.slot(player)
            entry.affected = True
            lost = drawn or player not in winners
            if entry.fields.get(HeadField.GAME_LOST) != 1:
                entry.set(HeadField.GAME_LOST, 1 if lost else 0)

    def _venture(self) -> None:
        for event in self.by_type[EventType.DUNGEON_VENTURED]:
            self._place(placed_subjects(self.record, event, self.events), HeadField.VENTURED,
                        lambda entry: entry.set(HeadField.VENTURED, 1))


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
    entities = {entity.id: entity for entity in record.state.entities}
    cause = event.params.get("cause")
    if event.type is EventType.PERMANENT_COPIED:
        entered = {
            subject for other in (events if events is not None else events_of(record))
            if other.type is EventType.ZONE_CHANGE
            and str(other.params.get("from_zone", "")).lower() == Zone.STACK
            and str(other.params.get("to_zone", "")).lower() == Zone.BATTLEFIELD
            for subject in other.subjects
        }
        return (cause,) if cause in entered else ()
    if event.type in (EventType.SPELL_COPIED, EventType.TARGETS_CHANGED):
        on_stack = tuple(
            ref for ref in record.state.refs.targets
            if ref in entities and entities[ref].zone == Zone.STACK
        )
        if on_stack:
            return on_stack
        if cause in entities and entities[cause].zone == Zone.STACK:
            return (cause,)
        return ()
    if event.type is EventType.DUNGEON_VENTURED:
        found = entities.get(cause)
        player = found.controller if found is not None else record.actor_player
        return (player,) if player in {p.id for p in record.state.players} else ()
    return ()


def _apply_playability(record: EffectRecord, slot) -> None:
    """Legality fields from the payload rather than from events, on every
    entity each one is defined for, the gate untouched (root spec § 11.4)."""
    payload = record.payload
    if isinstance(payload, PlayabilityDecisionPayload):
        _decision_legality(record, payload, slot)
    elif isinstance(payload, PlayabilityAttackersPayload):
        _attackers_legality(record, payload, slot)
    elif isinstance(payload, PlayabilityBlockersPayload):
        _mark_legality(slot, HeadField.BLOCKER_LEGAL, payload.legal_blockers, payload)
        # min_blockers is read at the anchored attacker, not at the blockers.
        slot(payload.anchor_attacker).set(HeadField.MIN_BLOCKERS, payload.min_blockers)
        if payload.must_block is not None:
            must = set(payload.must_block)
            for listed in (*payload.legal_blockers, *(f.entity for f in payload.forbidden)):
                slot(listed).set(HeadField.COMBAT_REQUIREMENT, 1 if listed in must else 0)


def target_legality_subjects(record: EffectRecord) -> tuple[str, ...]:
    """Every card and player a decision record writes ``target_legal`` at:
    each of the snapshot's, and each of the candidate's ``legal_targets``,
    when the candidate has a targeting clause (a legal target, or
    ``has_legal_target`` false for want of one); none otherwise."""
    payload = record.payload
    if not isinstance(payload, PlayabilityDecisionPayload) or not payload.candidates:
        return ()
    candidate = payload.candidates[0]
    if not candidate.legal_targets and candidate.has_legal_target:
        return ()
    return tuple(dict.fromkeys((
        *(e.id for e in record.state.entities), *(p.id for p in record.state.players),
        *candidate.legal_targets)))


def _decision_legality(record, payload: PlayabilityDecisionPayload, slot) -> None:
    """``target_legal`` at :func:`target_legality_subjects`: 1 in the
    candidate's ``legal_targets`` (the union over its targeting clauses), 0
    elsewhere."""
    subjects = target_legality_subjects(record)
    if not subjects:
        return
    legal = set(payload.candidates[0].legal_targets)
    for ref in subjects:
        slot(ref).set(HeadField.TARGET_LEGAL, 1 if ref in legal else 0)


def _attackers_legality(record, payload: PlayabilityAttackersPayload, slot) -> None:
    """The declaration's legal and forbidden attackers; on an anchored record
    which players, planeswalkers and battles the anchor may attack, whether
    it must, and on a declared anchor the tap the attack made (FR-037)."""
    _mark_legality(slot, HeadField.ATTACKER_LEGAL, payload.legal_attackers, payload)
    anchor = payload.anchor_attacker
    if anchor is None:
        return
    defenders = set(payload.legal_defenders)
    candidates = [player.id for player in record.state.players] + [
        entity.id for entity in record.state.entities
        if entity.zone == Zone.BATTLEFIELD
        and {"planeswalker", "battle"} & {kind.lower() for kind in entity.types}
    ]
    for ref in candidates:
        slot(ref).set(HeadField.ATTACK_TARGET_LEGAL, 1 if ref in defenders else 0)
    entry = slot(anchor)
    entry.set(HeadField.COMBAT_REQUIREMENT, 1 if payload.must_attack else 0)
    if payload.tapped_by_attack is not None:
        # The tap is what the declaration did to the anchor, so the anchor
        # is affected; the record trains no gate, so that is all it says.
        entity = next((e for e in record.state.entities if e.id == anchor), None)
        before = 1 if entity is not None and entity.tapped else 0
        entry.affected = True
        entry.set(HeadField.TAP_STATE, 1 if payload.tapped_by_attack else before)


def _mark_legality(slot, name: str, legal, payload) -> None:
    """One combat declaration's legal entities as 1, its forbidden ones as 0."""
    for entity in legal:
        slot(entity).set(name, 1)
    for forbidden in payload.forbidden:
        slot(forbidden.entity).set(name, 0)


def _apply_continuous(record: EffectRecord, slot) -> None:
    """A continuous record's contributions are its targets directly."""
    payload = record.payload
    if not isinstance(payload, ContinuousPayload):
        return
    entities = {entity.id: entity for entity in record.state.entities}
    for contribution in payload.contributions:
        entry = slot(contribution.entity)
        entry.affected = True
        if contribution.pt_boost != (0, 0):
            entry.add(HeadField.POWER_DELTA, float(contribution.pt_boost[0]))
            entry.add(HeadField.TOUGHNESS_DELTA, float(contribution.pt_boost[1]))
        if contribution.set_pt is not None:
            # Against the board without the static, as a resolution's
            # ``set_pt`` is against its snapshot: no base on a permanent that
            # was no creature there is a base of 0/0.
            entity = entities.get(contribution.entity)
            base = entity.pt.base if entity is not None and entity.pt is not None else (0, 0)
            for half, field_name in ((0, HeadField.BASE_POWER_DELTA),
                                     (1, HeadField.BASE_TOUGHNESS_DELTA)):
                if contribution.set_pt[half] is not None:
                    entry.set(field_name, float(contribution.set_pt[half]) - base[half])
        if contribution.controller is not None:
            entry.set(HeadField.CONTROL_CHANGE, 1)
        keywords = contribution.keyword_changes
        for keyword in keywords.gained:
            _flag_keyword(entry, HeadField.KEYWORDS_GAINED, keyword)
        for keyword in keywords.lost:
            _flag_keyword(entry, HeadField.KEYWORDS_LOST, keyword)
        # Whole-class tokens (``all-creature-types``) match no slot and drop.
        entity = entities.get(contribution.entity)
        before = _displaceable(record, entity)
        channels = [
            (contribution.type_changes, vocabulary, gained, lost)
            for gained, lost, vocabulary in _TYPE_CHANNELS
        ] + [(contribution.color_changes, COLORS,
              HeadField.COLORS_GAINED, HeadField.COLORS_LOST)]
        for change, vocabulary, gained_field, lost_field in channels:
            _apply_multi(entry, gained_field, change.gained, vocabulary)
            _apply_multi(entry, lost_field, change.lost, vocabulary)
            if not change.sets_outright:
                continue
            if before is None:
                # The displaced values cannot be named, so the lost field is
                # unknown rather than silently empty.
                if lost_field not in entry.fields:
                    entry.unknown.add(lost_field)
                continue
            # What the line held on the board without the static, less what
            # the static sets it to, is what it displaced.
            entry.flag(lost_field, -1, len(vocabulary))
            kept = {str(value) for value in change.gained}
            _apply_multi(entry, lost_field,
                         [value for value in before[lost_field] if value not in kept],
                         vocabulary)
