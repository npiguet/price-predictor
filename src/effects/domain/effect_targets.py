"""What actually happened, as the per-entity head's targets.

The one place the event vocabulary meets the field layout. An entity is
affected when any event names it, not when some field changed: a 0-damage hit
still affected its target. A field no event wrote is absent rather than zero
unless the layout gives it a silent value for this record's class and entity
kind (:meth:`EntityTargets.supervised`); zero everywhere would teach the head
that a destroyed creature was untapped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from effects.domain.effect_head_input import CARD_TYPES, COUNTER_TYPES, OVERLAY_KEYWORDS
from effects.domain.effect_head_layout import (
    CREATED_OBJECT_SLOTS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    DURATIONS,
    LIBRARY_EVENTS,
    VERDICT_BITS,
    VERDICT_WIDTH,
    ZONE_OUTCOMES,
    CreatedSlot,
    FieldScope,
    FieldSpec,
    HeadField,
    ZoneOutcome,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.keyword_names import overlay_keyword
from effects.domain.records import (
    ActivationPayload,
    ContinuousPayload,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    RecordKind,
    RewritePayload,
    TriggerPayload,
    events_of,
)
from effects.domain.sampling import sampling_class
from effects.domain.state_snapshot import COLORS

#: The zone outcome each moving event implies; the record keeps destroy and
#: sacrifice apart, the target collapses them.
_ZONE_BY_EVENT: dict[EventType, ZoneOutcome] = {
    EventType.DESTROYED: ZoneOutcome.DIED,
    EventType.SACRIFICED: ZoneOutcome.DIED,
    EventType.PHASED: ZoneOutcome.PHASED_OUT,
    EventType.FACE_CHANGE: ZoneOutcome.FACE_CHANGED,
    EventType.SPELL_COUNTERED: ZoneOutcome.COUNTERED,
}

_ZONE_BY_DESTINATION: dict[str, ZoneOutcome] = {
    "graveyard": ZoneOutcome.DIED,
    "exile": ZoneOutcome.EXILED,
    "hand": ZoneOutcome.TO_HAND,
    "library": ZoneOutcome.LIBRARY_TOP,
    "battlefield": ZoneOutcome.BLINKED,
}
#: The outcome of a ``zone_change`` to a destination the table does not name.
_STAYED = ZoneOutcome.STAYED

_LIBRARY_EVENT_BY_TYPE: dict[EventType, str] = {
    EventType.CARD_LOOKED_AT: "scry",
    EventType.LIBRARY_REORDERED: "reorder",
    EventType.CARD_REVEALED: "reveal",
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

_OVERLAY_INDEX: dict[str, int] = {
    keyword: index for index, keyword in enumerate(OVERLAY_KEYWORDS)
}
#: ``lower-cased member -> its flag``, for the type and colour channels.
_TYPE_SLOTS: dict[str, int] = {t.lower(): i for i, t in enumerate(CARD_TYPES)}
_COLOR_SLOTS: dict[str, int] = {c.lower(): i for i, c in enumerate(COLORS)}


def overlay_index(raw: str) -> int | None:
    """The overlay slot a keyword, in any spelling Forge writes, sets."""
    return _OVERLAY_INDEX.get(overlay_keyword(raw))


@dataclass
class EntityTargets:
    """One entity's targets for this record.

    ``fields`` holds what the record wrote and nothing else, ``unknown`` what
    it declares it cannot say; without ``record_class`` and ``scope`` only
    written values are scored.
    """

    affected: bool = False
    fields: dict[str, float | int | list[float]] = field(default_factory=dict)
    record_class: str | None = None
    scope: FieldScope | None = None
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

    for event in _outcome_events(record):
        for subject in event.subjects:
            entry = slot(subject)
            entry.affected = True
            _apply_event(entry, event)

    _apply_playability(record, slot)
    _apply_continuous(record, slot)
    return targets


def _outcome_events(record: EffectRecord) -> tuple[Event, ...]:
    """The events that are this record's outcome: none on a trigger record,
    whose events caused it and whose effect is its resolution's record."""
    if isinstance(record.payload, TriggerPayload):
        return ()
    return events_of(record)


def _apply_event(entry: EntityTargets, event: Event) -> None:
    """Fold one event into an entity's targets."""
    params = event.params
    if event.type in _ADDITIVE:
        name, parameter, default, sign = _ADDITIVE[event.type]
        entry.add(name, sign * float(params.get(parameter, default)))
        return
    if event.type in _SETTERS:
        entry.set(*_SETTERS[event.type])
        return
    if event.type in _ZONE_BY_EVENT:
        entry.set(HeadField.ZONE_OUTCOME, ZONE_OUTCOMES.index(_ZONE_BY_EVENT[event.type]))
        return
    if event.type in _LIBRARY_EVENT_BY_TYPE:
        entry.flag(
            HeadField.LIBRARY_EVENTS,
            LIBRARY_EVENTS.index(_LIBRARY_EVENT_BY_TYPE[event.type]),
            len(LIBRARY_EVENTS),
        )
        return
    match event.type:
        case EventType.ZONE_CHANGE:
            destination = str(params.get("to_zone", "")).lower()
            entry.set(
                HeadField.ZONE_OUTCOME,
                ZONE_OUTCOMES.index(_ZONE_BY_DESTINATION.get(destination, _STAYED)),
            )
        case EventType.COUNTER_CHANGE:
            counter = str(params.get("counter_type", "OTHER")).upper()
            name = (
                f"counters_delta_{counter.lower()}"
                if counter in COUNTER_TYPES
                else HeadField.COUNTERS_DELTA_OTHER
            )
            entry.add(name, float(params.get("delta", 0)))
        case EventType.PT_CHANGE:
            entry.add(HeadField.POWER_DELTA, float(params.get("power_delta", 0)))
            entry.add(HeadField.TOUGHNESS_DELTA, float(params.get("toughness_delta", 0)))
            _apply_duration(entry, HeadField.PT_DURATION, event.duration)
        case EventType.KEYWORD_CHANGE:
            _apply_keywords(entry, params)
        case EventType.TYPE_CHANGE:
            _apply_multi(entry, HeadField.TYPES_GAINED, params.get("types_added"), _TYPE_SLOTS)
            _apply_multi(entry, HeadField.TYPES_LOST, params.get("types_removed"), _TYPE_SLOTS)
            _apply_duration(entry, HeadField.TYPE_COLOR_DURATION, event.duration)
        case EventType.COLOR_CHANGE:
            _apply_multi(entry, HeadField.COLORS_GAINED, params.get("colors"), _COLOR_SLOTS)
            _apply_multi(entry, HeadField.COLORS_LOST, params.get("colors_removed"), _COLOR_SLOTS)
            _apply_duration(entry, HeadField.TYPE_COLOR_DURATION, event.duration)
        case EventType.MANA_PRODUCED:
            _apply_mana(entry, params.get("mana_by_color"), sign=1)
        case EventType.MANA_LOST:
            _apply_mana(entry, params.get("mana_by_color"), sign=-1)


def _apply_duration(entry: EntityTargets, name: str, duration: str | None) -> None:
    """How long a lasting change lasts, where the record says; an unrecognised
    or absent duration leaves the field unset rather than guessing a bucket."""
    if duration in DURATIONS:
        entry.set(name, DURATIONS.index(duration))


def _apply_multi(entry: EntityTargets, name: str, values, slots: dict[str, int]) -> None:
    for value in values or ():
        index = slots.get(str(value).strip().lower())
        if index is not None:
            entry.flag(name, index, len(slots))


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
            entry.add(f"mana_delta_{key.lower()}", sign * float(amount))


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
        for entity in payload.legal_attackers:
            mark(entity, HeadField.ATTACKER_LEGAL, 1)
        for forbidden in payload.forbidden:
            mark(forbidden.entity, HeadField.ATTACKER_LEGAL, 0)
    elif isinstance(payload, PlayabilityBlockersPayload):
        for entity in payload.legal_blockers:
            mark(entity, HeadField.BLOCKER_LEGAL, 1)
        for forbidden in payload.forbidden:
            mark(forbidden.entity, HeadField.BLOCKER_LEGAL, 0)
        # min_blockers is read at the anchored attacker, not at the blockers.
        mark(payload.anchor_attacker, HeadField.MIN_BLOCKERS, payload.min_blockers)


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
        for keyword in contribution.keywords:
            # The type and colour channels' "-" convention.
            lost = keyword.startswith("-")
            _flag_keyword(
                entry, HeadField.KEYWORDS_LOST if lost else HeadField.KEYWORDS_GAINED,
                keyword.lstrip("-"),
            )
        for channel, slots, gained_field, lost_field in (
            (contribution.types, _TYPE_SLOTS, HeadField.TYPES_GAINED, HeadField.TYPES_LOST),
            (contribution.colors, _COLOR_SLOTS, HeadField.COLORS_GAINED, HeadField.COLORS_LOST),
        ):
            gained, lost = _split_contribution(channel)
            _apply_multi(entry, gained_field, gained, slots)
            _apply_multi(entry, lost_field, lost, slots)
            if "=" in channel and lost_field not in entry.fields:
                entry.unknown.add(lost_field)


def _split_contribution(tokens: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """One contribution channel, split into what it grants and what it takes.

    A removal carries a leading ``-``; a lone ``=`` marks a channel the static
    sets outright, whose displaced values the record cannot name (the snapshot
    is the board after the static), so the caller marks the lost field
    unknown. Whole-class tokens (``all-creature-types``) match no slot and drop.
    """
    gained: list[str] = []
    lost: list[str] = []
    for token in tokens:
        if token == "=":
            continue
        if token.startswith("-"):
            lost.append(token[1:])
        else:
            gained.append(token)
    return gained, lost


# ── the [ACT] and [GLOBAL] heads (FR-060a) ──────────────────────────────


def verdict_targets(
    record: EffectRecord,
) -> tuple[list[float], list[bool]] | None:
    """The verdict head's targets at ``[ACT]`` and which of them this record sets.

    A ``decision`` record sets its candidate's verdict bits, a cost half the
    mana it paid by colour, a ``trigger`` record whether it fired; every other
    column is masked out, since a cost half is no evidence of playability.
    """
    target = [0.0] * VERDICT_WIDTH
    mask = [False] * VERDICT_WIDTH
    payload = record.payload
    bits = len(VERDICT_BITS)
    if isinstance(payload, PlayabilityDecisionPayload):
        if not payload.candidates:
            return None
        candidate = payload.candidates[0]
        for index, value in enumerate(
            (candidate.can_play, candidate.affordable, candidate.has_legal_target)
        ):
            target[index] = 1.0 if value else 0.0
            mask[index] = True
        return target, mask
    if record.moment is Moment.ACTIVATION and isinstance(payload, ActivationPayload):
        paid = payload.costs.mana_by_color
        for offset, color in enumerate(COLORS):
            target[bits + offset] = float(paid.get(color, 0))
            mask[bits + offset] = True
        return target, mask
    if isinstance(payload, TriggerPayload):
        target[-1] = 1.0 if payload.fired else 0.0
        mask[-1] = True
        return target, mask
    return None


def supervises_created_objects(record: EffectRecord) -> bool:
    """Effect halves and in-place rewrites carry created-objects targets; any
    other rewrite ran another ability, whose effect half supervises it."""
    if record.kind is RecordKind.RESOLUTION:
        return record.moment is Moment.RESOLUTION
    if record.kind is RecordKind.REWRITE:
        return isinstance(record.payload, RewritePayload) and record.payload.rewritten
    return False


def created_objects_targets(record: EffectRecord) -> list[float]:
    """The created-objects head's targets at ``[GLOBAL]`` (FR-078).

    Scripted groups first by id, then characteristics-only groups by
    descending count; the flag past the last slot marks more than K groups.
    """
    groups: dict[tuple, TokenGroup] = {}
    for event in events_of(record):
        if event.type is not EventType.TOKEN_CREATED:
            continue
        found = TokenGroup.of(event.params)
        group = groups.setdefault(found.key, found)
        group.count += int(event.params.get("count", 1) or 1)

    ordered = sorted(
        (g for g in groups.values() if g.scripted), key=lambda g: g.id,
    ) + sorted(
        (g for g in groups.values() if not g.scripted), key=lambda g: -g.count,
    )
    target = [0.0] * CREATED_OBJECTS_WIDTH
    for slot, group in enumerate(ordered[:CREATED_OBJECT_SLOTS]):
        group.write(target, slot * CREATED_SLOT_WIDTH)
    target[-1] = 1.0 if len(ordered) > CREATED_OBJECT_SLOTS else 0.0
    return target


@dataclass
class TokenGroup:
    """Tokens one record created that the head counts as one group (FR-078):
    by token-script id where the event names one, else by characteristics."""

    scripted: bool
    id: str
    power: object
    toughness: object
    types: str
    keywords: tuple
    count: int = 0

    @classmethod
    def of(cls, params: dict) -> TokenGroup:
        script = params.get("token_script_id") or None
        traits = params.get("characteristics") or {}
        keywords = traits.get("keywords") or ()
        if isinstance(keywords, str):
            keywords = (keywords,)
        return cls(
            scripted=script is not None, id=str(script or ""),
            power=traits.get("power"), toughness=traits.get("toughness"),
            types=str(traits.get("types", "")), keywords=tuple(keywords),
        )

    @property
    def key(self) -> tuple:
        if self.scripted:
            return ("script", self.id)
        return ("traits", self.types, self.power, self.toughness,
                tuple(sorted(map(str, self.keywords))))

    def write(self, target: list[float], base: int) -> None:
        """This group's slot, from ``base`` on."""
        target[base + CreatedSlot.PRESENT] = 1.0
        target[base + CreatedSlot.SCRIPTED] = 1.0 if self.scripted else 0.0
        target[base + CreatedSlot.COUNT] = float(self.count)
        target[base + CreatedSlot.POWER] = max(_number(self.power), 0.0)
        target[base + CreatedSlot.TOUGHNESS] = max(_number(self.toughness), 0.0)
        card_types = self.types.split(" - ", 1)[0].lower().split()
        for index, card_type in enumerate(CARD_TYPES):
            if card_type in card_types:
                target[base + CreatedSlot.TYPES + index] = 1.0
        for keyword in self.keywords:
            index = overlay_index(str(keyword))
            if index is not None:
                target[base + CreatedSlot.KEYWORDS + index] = 1.0


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
