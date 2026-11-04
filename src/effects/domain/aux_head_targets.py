"""What the ``[ACT]`` and ``[GLOBAL]`` heads train on (FR-060a).

The per-entity targets (:mod:`effects.domain.effect_targets`) say what
happened to each entity; these say what the acting line's verdict was, which
objects it created and what it did to the game's turn structure, one row per
record rather than per entity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from effects.domain.closed_lists import subtypes_of, type_tokens
from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    MANA_SYMBOLS,
    vocabulary_slot,
)
from effects.domain.effect_head_layout import (
    CREATED_KINDS,
    CREATED_OBJECT_SLOTS,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    DAY_NIGHT_CHANGES,
    FAILED_CHECKS,
    GLOBAL_LAYOUT,
    REPLACEMENT_RESULTS,
    RESOLUTION_OUTCOMES,
    VERDICT_BITS,
    VERDICT_LAYOUT,
    CreatedKind,
    CreatedSlot,
    DayNightChange,
    GlobalField,
    VerdictField,
)
from effects.domain.effect_targets import keyword_index
from effects.domain.event_schema import EventType
from effects.domain.output_vocabularies import SUBTYPES_CHANGED
from effects.domain.provenance import TraitKind
from effects.domain.records import (
    ActivationPayload,
    Candidate,
    EffectRecord,
    PlayabilityDecisionPayload,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
    events_of,
)
from effects.domain.state_snapshot import COLORS, Zone, normalize_counter_type

#: A record head's targets: one value per target column, one flag per field.
HeadRow = tuple[list[float], list[bool]]

#: The events a resolution's ``effects_created`` counts.
_EFFECT_OBJECTS = frozenset({
    EventType.CONTINUOUS_EFFECT_CREATED, EventType.DELAYED_TRIGGER_CREATED,
})


def verdict_targets(record: EffectRecord) -> HeadRow | None:
    """The verdict head's targets at ``[ACT]`` and which fields this record sets.

    A ``decision`` record sets its candidate's verdict bits, timing, failed
    checks and cost adjustment; a cost half the mana it paid by colour and its
    resolution outcome; an effect half the effect objects and delayed
    triggers it created; a ``rewrite`` record the replacement's result; a
    ``trigger`` record whether it fired. Every other field is masked out,
    since a cost half is no evidence of playability.
    """
    row = VERDICT_LAYOUT.blank()
    match record.payload:
        case PlayabilityDecisionPayload(candidates=(candidate, *_)):
            _decision_verdict(record, candidate, row)
        case ActivationPayload(costs=costs, outcome=outcome):
            paid = costs.mana_by_color
            VERDICT_LAYOUT.put(row, VerdictField.MANA_PAID,
                               [float(paid.get(color, 0)) for color in COLORS])
            if outcome is not None:
                VERDICT_LAYOUT.put(row, VerdictField.RESOLUTION_OUTCOME,
                                   RESOLUTION_OUTCOMES.index(outcome))
        case ResolutionPayload():
            created = sum(1 for event in events_of(record) if event.type in _EFFECT_OBJECTS)
            VERDICT_LAYOUT.put(row, VerdictField.EFFECTS_CREATED, created)
        case RewritePayload(result=result) if result is not None:
            VERDICT_LAYOUT.put(row, VerdictField.REPLACEMENT_RESULT,
                               REPLACEMENT_RESULTS.index(result))
        case TriggerPayload(fired=fired):
            VERDICT_LAYOUT.put(row, VerdictField.TRIGGER_FIRED, 1.0 if fired else 0.0)
        case _:
            return None
    return row


def _decision_verdict(record: EffectRecord, candidate: Candidate, row: HeadRow) -> None:
    for bit, value in zip(
        VERDICT_BITS, (candidate.can_play, candidate.affordable, candidate.has_legal_target),
    ):
        VERDICT_LAYOUT.put(row, bit, 1.0 if value else 0.0)
    if candidate.timing is not None:
        VERDICT_LAYOUT.put(row, VerdictField.TIMING, 1.0 if candidate.timing else 0.0)
    if candidate.failed_checks is not None:
        failed = {check.check for check in candidate.failed_checks}
        VERDICT_LAYOUT.put(row, VerdictField.FAILED_CHECKS,
                           [1.0 if check in failed else 0.0 for check in FAILED_CHECKS])
    adjustment = cost_adjustment(record, candidate)
    if adjustment is not None:
        VERDICT_LAYOUT.put(row, VerdictField.COST_ADJUSTMENT, adjustment)


_SYMBOL = re.compile(r"\{([^}]*)\}")


def mana_symbols(cost: str) -> dict[str, int] | None:
    """A Forge mana cost string (``{2}{R}``, ``{W/U}{X}``) as symbol counts in
    the snapshot's ``mana_cost`` spelling; None where it holds a symbol the
    layout does not list."""
    counts: dict[str, int] = {}
    for symbol in _SYMBOL.findall(cost):
        symbol = symbol.strip().upper()
        if symbol.isdigit():
            if int(symbol):
                counts["generic"] = counts.get("generic", 0) + int(symbol)
            continue
        if symbol not in MANA_SYMBOLS:
            return None
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def cost_adjustment(record: EffectRecord, candidate: Candidate) -> list[float] | None:
    """The cost Forge would charge for a spell, less its printed cost, per
    mana symbol kind (root spec § 11.2); None where the record cannot say:
    a land play, an ability or alternative cost (whose printed cost is not
    the card's), a face not up, or a cost or card the layout cannot spell."""
    if candidate.land_play or not candidate.ability:
        return None
    key = candidate.ability[0]
    # A card's cast is its first spell ability (a permanent's the implicit
    # one); the later ones are its activated abilities, each with a cost of
    # its own.
    if key.trait_kind != TraitKind.SPELL or key.index_within_kind != 0:
        return None
    host = next(
        (entity for entity in record.state.entities if entity.id == record.state.refs.source),
        None,
    )
    if (
        host is None or host.mana_cost is None or key.face != host.face
        or host.zone == Zone.BATTLEFIELD
    ):
        return None
    charged = candidate.cost_after_adjustment.get("mana")
    if not isinstance(charged, str):
        return None
    after = mana_symbols(charged)
    if after is None or any(symbol not in MANA_SYMBOLS for symbol in host.mana_cost):
        return None
    return [
        float(after.get(symbol, 0) - host.mana_cost.get(symbol, 0)) for symbol in MANA_SYMBOLS
    ]


def supervises_created_objects(record: EffectRecord) -> bool:
    """Effect halves and in-place rewrites carry created-objects targets; any
    other rewrite ran another ability, whose effect half supervises it."""
    match record.payload:
        case ResolutionPayload():
            return True
        case RewritePayload(rewritten=rewritten):
            return rewritten
    return False


def global_targets(record: EffectRecord) -> HeadRow | None:
    """The global head's targets at ``[GLOBAL]``: turns and phases added or
    skipped, and day becoming night or night day. Supervised by the records
    that supervise the created-objects head, whose events list every such
    change; None for any other."""
    if not supervises_created_objects(record):
        return None
    counts = dict.fromkeys(_TURN_STRUCTURE, 0.0)
    day_night = DayNightChange.UNCHANGED
    for event in events_of(record):
        if event.type in _TURN_STRUCTURE:
            counts[event.type] += float(event.params.get("count", 1) or 1)
        elif event.type is EventType.DAY_NIGHT_CHANGED:
            day_night = (
                DayNightChange.BECAME_DAY if event.params.get("to") == "day"
                else DayNightChange.BECAME_NIGHT
            )
    row = GLOBAL_LAYOUT.blank()
    GLOBAL_LAYOUT.put(row, GlobalField.TURN_STRUCTURE_CHANGE, list(counts.values()))
    GLOBAL_LAYOUT.put(row, GlobalField.DAY_NIGHT_CHANGE, DAY_NIGHT_CHANGES.index(day_night))
    return row


#: The turn-structure events, in ``turn_structure_change``'s member order.
_TURN_STRUCTURE: tuple[EventType, ...] = (
    EventType.TURN_ADDED, EventType.TURN_SKIPPED,
    EventType.PHASE_ADDED, EventType.PHASE_SKIPPED,
)


def created_objects_targets(record: EffectRecord) -> tuple[list[float], list[bool]]:
    """The created-objects head's targets at ``[GLOBAL]`` (FR-078), and which
    columns the record says.

    Tokens, cards made outside the game and effect objects, each kind in
    groups; scripted groups first by id, then the rest by descending count;
    the flag past the last slot marks more than K groups. A group's
    controller and counters are trained only where its event names them.
    """
    groups: dict[tuple, CreatedGroup] = {}
    for event in events_of(record):
        found = CreatedGroup.of(event.type, event.params, record.actor_player)
        if found is None:
            continue
        group = groups.setdefault(found.key, found)
        group.count += int(event.params.get("count", 1) or 1)

    ordered = sorted(
        (g for g in groups.values() if g.scripted), key=lambda g: g.id,
    ) + sorted(
        (g for g in groups.values() if not g.scripted), key=lambda g: -g.count,
    )
    target = [0.0] * CREATED_OBJECTS_WIDTH
    known = [True] * CREATED_OBJECTS_WIDTH
    for slot, group in enumerate(ordered[:CREATED_OBJECT_SLOTS]):
        group.write(target, known, slot * CREATED_SLOT_WIDTH)
    target[-1] = 1.0 if len(ordered) > CREATED_OBJECT_SLOTS else 0.0
    return target, known


@dataclass
class CreatedGroup:
    """Objects one record created that the head counts as one group (FR-078):
    tokens by token-script id where the event names one, else by
    characteristics; a made card by name; an effect object by what it does."""

    kind: CreatedKind
    scripted: bool
    id: str
    power: object = None
    toughness: object = None
    types: str = ""
    keywords: tuple = ()
    #: Whether a player other than the actor controls the group; None where
    #: the event does not say.
    other_controller: bool | None = None
    #: Counter type to count, None where the event does not say.
    counters: dict[str, int] | None = None
    count: int = 0

    @classmethod
    def of(cls, event_type: EventType, params: dict, actor: str) -> CreatedGroup | None:
        if event_type is EventType.TOKEN_CREATED:
            script = params.get("token_script_id") or None
            traits = params.get("characteristics") or {}
            keywords = traits.get("keywords") or ()
            if isinstance(keywords, str):
                keywords = (keywords,)
            controller = traits.get("controller")
            counters = traits.get("counters")
            return cls(
                kind=CreatedKind.TOKEN, scripted=script is not None, id=str(script or ""),
                power=traits.get("power"), toughness=traits.get("toughness"),
                types=str(traits.get("types", "")), keywords=tuple(keywords),
                other_controller=None if controller is None else controller != actor,
                counters=dict(counters) if isinstance(counters, dict) else None,
            )
        if event_type is EventType.CARD_MADE:
            return cls(kind=CreatedKind.CARD, scripted=False, id=str(params.get("card_name", "")))
        if event_type is EventType.CONTINUOUS_EFFECT_CREATED:
            return cls(kind=CreatedKind.EFFECT, scripted=False, id=str(params.get("layers", "")))
        return None

    @property
    def key(self) -> tuple:
        if self.scripted:
            return ("script", self.id)
        if self.kind is not CreatedKind.TOKEN:
            return (self.kind, self.id)
        return ("traits", self.types, self.power, self.toughness,
                tuple(sorted(map(str, self.keywords))))

    def write(self, target: list[float], known: list[bool], base: int) -> None:
        """This group's slot, from ``base`` on."""
        target[base + CreatedSlot.PRESENT] = 1.0
        target[base + CreatedSlot.SCRIPTED] = 1.0 if self.scripted else 0.0
        target[base + CreatedSlot.COUNT] = float(self.count)
        target[base + CreatedSlot.POWER] = max(_number(self.power), 0.0)
        target[base + CreatedSlot.TOUGHNESS] = max(_number(self.toughness), 0.0)
        target[base + CreatedSlot.KIND + CREATED_KINDS.index(self.kind)] = 1.0
        if self.other_controller is None:
            known[base + CreatedSlot.OTHER_CONTROLLER] = False
        else:
            target[base + CreatedSlot.OTHER_CONTROLLER] = 1.0 if self.other_controller else 0.0
        self._write_counters(target, known, base)
        tokens = type_tokens(self.types)
        for card_type in tokens:
            index = vocabulary_slot(card_type, CARD_TYPES)
            if index is not None:
                target[base + CreatedSlot.TYPES + index] = 1.0
        for subtype in subtypes_of(tokens):
            index = vocabulary_slot(subtype, SUBTYPES_CHANGED)
            if index is not None:
                target[base + CreatedSlot.SUBTYPES + index] = 1.0
        for keyword in self.keywords:
            index = keyword_index(str(keyword))
            if index is not None:
                target[base + CreatedSlot.KEYWORDS + index] = 1.0

    def _write_counters(self, target: list[float], known: list[bool], base: int) -> None:
        other = base + CreatedSlot.COUNTERS + len(COUNTER_TYPES)
        if self.counters is None:
            for column in range(base + CreatedSlot.COUNTERS, other + 1):
                known[column] = False
            return
        for counter_type, amount in self.counters.items():
            name = normalize_counter_type(str(counter_type))
            column = (
                base + CreatedSlot.COUNTERS + COUNTER_TYPES.index(name)
                if name in COUNTER_TYPES else other
            )
            target[column] += max(_number(amount), 0.0)


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
