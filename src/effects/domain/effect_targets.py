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
- **A field with no event is absent, not zero, unless silence is itself the
  answer.** An entity's ``fields`` hold only what the record wrote, and
  :meth:`EntityTargets.supervised` decides what the head is scored on: a written
  value, else the field's silent value where the layout gives one for this
  record's class and this kind of entity (:mod:`effect_head_layout`), else
  nothing. Writing 0 for every unmentioned field would teach the head that a
  destroyed creature was untapped and a player's zone did not change.
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

    ``affected`` is the gate. ``fields`` holds what the record wrote and
    nothing else; ``unknown`` the fields it declares it cannot say. The record's
    sampling class and the entity's kind decide which silent values apply
    (:meth:`supervised`); without them only written values are scored.
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
    """The events that are this record's outcome, which is not every event it carries.

    A fired trigger's event is what caused it, read as the pending-event overlay
    on its input; what the trigger then does arrives as the resolution record
    of the ability it put on the stack. A trigger record supervises the verdict
    head's fired bit alone (base spec, record-kind table).
    """
    if isinstance(record.payload, TriggerPayload):
        return ()
    return events_of(record)


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
            _apply_duration(entry, "pt_duration", event.duration)
        case EventType.KEYWORD_CHANGE:
            _apply_keywords(entry, params)
        case EventType.TYPE_CHANGE:
            _apply_multi(entry, "types_gained", params.get("types_added"), _TYPE_SLOTS)
            _apply_multi(entry, "types_lost", params.get("types_removed"), _TYPE_SLOTS)
            _apply_duration(entry, "type_color_duration", event.duration)
        case EventType.COLOR_CHANGE:
            _apply_multi(entry, "colors_gained", params.get("colors"), _COLOR_SLOTS)
            _apply_multi(entry, "colors_lost", params.get("colors_removed"), _COLOR_SLOTS)
            _apply_duration(entry, "type_color_duration", event.duration)
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


def _apply_duration(entry: EntityTargets, name: str, duration: str | None) -> None:
    """How long a lasting change lasts, where the record says.

    The two duration fields were the only ones in the head no path wrote, and
    the cause was upstream: an ``Event``'s ``duration`` had a setter nothing
    called, so a pump until end of turn and a permanent one recorded
    identically and there was nothing to learn the difference from.

    An unrecognised or absent duration leaves the field unset rather than
    guessing a bucket, which reads as "this record does not say".
    """
    if duration in DURATIONS:
        entry.set(name, DURATIONS.index(duration))


def _apply_multi(entry: EntityTargets, name: str, values, slots: dict[str, int]) -> None:
    for value in values or ():
        index = slots.get(str(value).strip().lower())
        if index is not None:
            entry.flag(name, index, len(slots))


def _apply_keywords(entry: EntityTargets, params: dict) -> None:
    name = "keywords_lost" if params.get("removed") else "keywords_gained"
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
                mark(target, "target_legal", 1)
    elif isinstance(payload, PlayabilityAttackersPayload):
        for entity in payload.legal_attackers:
            mark(entity, "attacker_legal", 1)
        for forbidden in payload.forbidden:
            mark(forbidden.entity, "attacker_legal", 0)
    elif isinstance(payload, PlayabilityBlockersPayload):
        for entity in payload.legal_blockers:
            mark(entity, "blocker_legal", 1)
        for forbidden in payload.forbidden:
            mark(forbidden.entity, "blocker_legal", 0)
        # min_blockers is read at the anchored attacker, not at the blockers.
        mark(payload.anchor_attacker, "min_blockers", payload.min_blockers)


def _apply_continuous(record: EffectRecord, slot) -> None:
    """A continuous record's contributions are its targets directly."""
    payload = record.payload
    if not isinstance(payload, ContinuousPayload):
        return
    for contribution in payload.contributions:
        entry = slot(contribution.entity)
        entry.affected = True
        if contribution.pt_boost != (0, 0):
            entry.add("power_delta", float(contribution.pt_boost[0]))
            entry.add("toughness_delta", float(contribution.pt_boost[1]))
        for keyword in contribution.keywords:
            # Same "-" convention as the type and colour channels: a static
            # that takes a keyword away marks it, because the contribution is
            # one flat list and the head has a field for each direction.
            lost = keyword.startswith("-")
            _flag_keyword(
                entry, "keywords_lost" if lost else "keywords_gained",
                keyword.lstrip("-"),
            )
        for channel, slots, name in (
            (contribution.types, _TYPE_SLOTS, "types"),
            (contribution.colors, _COLOR_SLOTS, "colors"),
        ):
            gained, lost = _split_contribution(channel)
            _apply_multi(entry, f"{name}_gained", gained, slots)
            _apply_multi(entry, f"{name}_lost", lost, slots)
            if "=" in channel and f"{name}_lost" not in entry.fields:
                entry.unknown.add(f"{name}_lost")


def _split_contribution(tokens: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """One contribution channel, split into what it grants and what it takes.

    The collector writes a channel as one list because the record schema fixes
    the field, so a removal is marked with a leading ``-`` and a lone ``=``
    marks the channel as one the static sets outright rather than adds to.

    A replacement reports only what it sets. What it displaced is not in the
    record: the snapshot beside it is the board after the static applied, so
    reading the entity's current types back would name what it still has, not
    what it lost. The caller marks the lost field unknown, which is the one
    honest answer: the loss skips it rather than scoring it as an empty set.

    Tokens naming a whole class of type -- ``all-creature-types`` and the
    ``-all-*`` removals -- match nothing in ``CARD_TYPES`` and drop here. They
    are kept in the record because they are real, and the model has no field
    for "every creature type at once" to put them in.
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

    Three kinds supervise it, each a different slice: a ``decision`` record its
    candidate's three verdict bits (Java writes one candidate per record, so the
    candidate is the payload's first), a cost half the mana it paid by colour,
    and a ``trigger`` record whether it fired. None for every other record, and
    each column the record says nothing about is left out of its mask — a cost
    half is no evidence that the ability was playable.
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
    """Effect halves and in-place rewrites carry created-objects targets.

    A rewrite only where ``outgoing`` is non-null: every other result was
    carried out by running another ability, whose own effect half supervises
    what it made (base spec § Output heads).
    """
    if record.kind is RecordKind.RESOLUTION:
        return record.moment is Moment.RESOLUTION
    if record.kind is RecordKind.REWRITE:
        return isinstance(record.payload, RewritePayload) and record.payload.rewritten
    return False


def created_objects_targets(record: EffectRecord) -> list[float]:
    """The created-objects head's targets at ``[GLOBAL]`` (FR-078).

    One group per distinct token: by token-script id where the event names one
    (scripted), otherwise by its characteristics. Slots are in canonical order —
    scripted groups sorted by id, characteristics-only groups after them by
    descending count — and the flag past the last slot marks more than K
    groups. Every slot carries the group's count, power, toughness and type and
    keyword flags, read from the event's ``characteristics``.
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
    """Tokens one record created that the head counts as one group (FR-078).

    Grouped by token-script id where the event names one, otherwise by the
    token's characteristics.
    """

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
