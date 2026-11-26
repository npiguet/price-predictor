"""The record wire's closed lists, checked as a shard is read.

The decoder keeps a snapshot's strings as the record carries them, so a value
none of these lists names reads on and lands wherever its reader defaults: an
entity's zone in no zone slot, a link type the relation table raises on only
at training, a ``zone_change`` destination read as "stayed". ``shard_io``
counts every such value and reports the first sighting of each, by shard and
line; the record still decodes, since one odd value leaves the rest of the
corpus usable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping

from effects.domain.event_schema import DamageType, EventType
from effects.domain.records import ForkKind
from effects.domain.relation_types import FAMILIES
from effects.domain.state_snapshot import (
    CombatSubstep,
    DayNight,
    EffectDuration,
    EntityRole,
    FaceState,
    LineGrant,
    LineKind,
)

#: The spelling of a member a record that should carry it leaves out.
ABSENT = "<absent>"

#: Every zone the collector can name: Forge's ``ZoneType`` constants, as
#: ``CardSpelling.zoneName`` lower-cases them (``merged`` is a mutated
#: permanent's merged card, ``none`` a move to no zone at all). Pinned to
#: ``ZoneType.java`` by a test.
FORGE_ZONES: tuple[str, ...] = (
    "hand", "library", "graveyard", "battlefield", "exile", "flashback", "command",
    "stack", "sideboard", "ante", "merged", "schemedeck", "planardeck",
    "attractiondeck", "junkyard", "contraptiondeck", "subgame", "extrahand", "none",
)
#: Every phase the snapshot can name: Forge's ``PhaseType`` constants,
#: lower-cased (``SnapshotBuilder.phaseName``). Pinned to ``PhaseType.java``.
FORGE_PHASES: tuple[str, ...] = (
    "untap", "upkeep", "draw", "main1", "combat_begin", "combat_declare_attackers",
    "combat_declare_blockers", "combat_first_strike_damage", "combat_damage",
    "combat_end", "main2", "end_of_turn", "cleanup",
)

#: A ``damage_dealt``'s ``damage_type`` (Forge's ``GameEventCardDamaged.DamageType``
#: as ``BusEvents.damageType`` spells it). Pinned to that source by a test.
DAMAGE_TYPES: tuple[str, ...] = tuple(str(damage_type) for damage_type in DamageType)
#: A ``player_lost``'s ``reason``: Forge's ``GameLossReason`` names. Pinned to
#: ``GameLossReason.java`` by a test.
GAME_LOSS_REASONS: tuple[str, ...] = (
    "Conceded", "LifeReachedZero", "Milled", "Poisoned", "SpellEffect",
    "CommanderDamage", "OpponentWon", "IntentionalDraw",
)
#: A ``choice_made``'s ``choice_kind``, decided by the clause's API
#: (``ChoiceApiRules``' choice readings). Pinned to that source by a test.
CHOICE_KINDS: tuple[str, ...] = (
    "color", "type", "number", "card_name", "player", "cards", "direction",
    "even_odd", "mode",
)
#: A fork record's ``copy_dropped`` kinds: Forge's ``GameCopier.NoCounterpart``
#: names lower-cased (``Forks.dropped``). Pinned to ``GameCopier.java`` by a test.
COPY_DROPPED_KINDS: tuple[str, ...] = (
    "command", "delayed_trigger", "remembered", "link", "attack", "ability", "activation",
    "pending_trigger",
)
#: An event's ``duration`` (``EventDuration.java``): instant, end of turn or
#: permanent, null on an event that writes none.
EVENT_DURATIONS: tuple[str, ...] = ("instant", "end_of_turn", "permanent")

#: An entity's zone and a ``zone_change``'s ends: a zone, or null for a
#: last-known card no zone holds and a move whose end Forge does not name.
_ZONES = frozenset(FORGE_ZONES) | {None}
#: The phase, null where the game has none yet.
_PHASES = frozenset(FORGE_PHASES) | {None}
_SUBSTEPS = frozenset(CombatSubstep) | {None}
_DAY_NIGHT = frozenset(DayNight) | {None}
_DURATIONS = frozenset(EffectDuration)
_FACE_STATES = frozenset(FaceState) | {None}
_ROLES = frozenset(EntityRole) | {None}
_LINE_KINDS = frozenset(LineKind)
#: A line's grant, absent on a printed line.
_LINE_GRANTS = frozenset(LineGrant) | {None}
_LINK_TYPES = frozenset(FAMILIES)
_ATTRIBUTION_SENTINELS = frozenset({"root", "unresolved"})
_EVENT_DURATIONS = frozenset(EVENT_DURATIONS) | {None}
_COPY_DROPPED_KINDS = frozenset(COPY_DROPPED_KINDS)
#: A fork record's ``fork_kind`` (``ForkKind``, pinned to the connector's by
#: a test); a live record carries none.
_FORK_KINDS = frozenset(ForkKind)
#: ``(event type, parameter) -> the values it may take``; a parameter the
#: event does not carry is not checked.
_EVENT_PARAM_LISTS: dict[tuple[str, str], frozenset] = {
    (EventType.ZONE_CHANGE, "from_zone"): _ZONES,
    (EventType.ZONE_CHANGE, "to_zone"): _ZONES,
    (EventType.SPELL_CAST, "from_zone"): _ZONES,
    (EventType.FACE_CHANGE, "to_state"): frozenset(FaceState),
    (EventType.DAMAGE_DEALT, "damage_type"): frozenset(DAMAGE_TYPES),
    (EventType.PLAYER_LOST, "reason"): frozenset(GAME_LOSS_REASONS),
    (EventType.CHOICE_MADE, "choice_kind"): frozenset(CHOICE_KINDS),
}


def off_list_values(raw: Mapping) -> Iterator[tuple[str, object]]:
    """``(member, value)`` for every value of one raw record outside its
    member's closed list; a member a record need not carry is checked only
    where present."""
    yield from _envelope(raw)
    yield from _state(_block(raw, "state"))
    for event in _payload_events(_block(raw, "payload")):
        yield from _event(event)


def _block(holder: Mapping, member: str) -> Mapping:
    """The object ``member`` holds, an empty one where it is absent or empty."""
    return holder.get(member) or {}


def _each(holder: Mapping, member: str) -> Iterable:
    """The items ``member`` holds, none where it is absent or empty."""
    return holder.get(member) or ()


def _envelope(raw: Mapping) -> Iterator[tuple[str, object]]:
    if "mode" not in raw:
        yield "mode", ABSENT
    for kind in _each(raw, "copy_dropped"):
        if _outside(kind, _COPY_DROPPED_KINDS):
            yield "copy_dropped", kind
    yield from _fork_kind(raw)


def _fork_kind(raw: Mapping) -> Iterator[tuple[str, object]]:
    """A fork record's ``fork_kind`` outside its list, or absent: a fork
    record written before the member existed. A live record's is not read."""
    if not raw.get("fork"):
        return
    kind = raw.get("fork_kind", ABSENT)
    if kind is ABSENT or _outside(kind, _FORK_KINDS):
        yield "fork_kind", kind


def _state(state: Mapping) -> Iterator[tuple[str, object]]:
    yield from _global(_block(state, "global"))
    for entity in _each(state, "entities"):
        yield from _entity(entity)
    for link in _each(state, "links"):
        yield from _members(link, "link", _LINK_LISTS)
    pending = state.get("pending_event")
    if pending:
        yield from _event(pending)


def _outside(value: object, allowed: frozenset) -> bool:
    """Whether ``value`` is no member of ``allowed``; a list or an object
    where a string belongs is none."""
    try:
        return value not in allowed
    except TypeError:
        return True


#: ``(member, its list)`` of each block whose members are checked alike.
_GLOBAL_LISTS = (("phase", _PHASES), ("combat_substep", _SUBSTEPS), ("day_night", _DAY_NIGHT))
_ENTITY_LISTS = (("zone", _ZONES), ("face_state", _FACE_STATES), ("role", _ROLES))
_LINE_LISTS = (("kind", _LINE_KINDS), ("grant", _LINE_GRANTS))
_LINK_LISTS = (("type", _LINK_TYPES),)


def _members(block: Mapping, prefix: str, lists) -> Iterator[tuple[str, object]]:
    """``(prefix.member, value)`` for each of ``lists``' members of ``block``
    outside its list."""
    for member, allowed in lists:
        if _outside(block.get(member), allowed):
            yield f"{prefix}.{member}", block.get(member)


def _global(block: Mapping) -> Iterator[tuple[str, object]]:
    yield from _members(block, "global", _GLOBAL_LISTS)
    for member in ("floating_effects", "delayed_triggers"):
        for made in _each(block, member):
            if _bad_duration(made):
                yield f"global.{member}.duration", made["duration"]


def _bad_duration(made: Mapping) -> bool:
    """Whether a floating effect or delayed trigger carries a duration outside
    its list; one carrying none is not judged."""
    return "duration" in made and _outside(made["duration"], _DURATIONS)


def _entity(entity: Mapping) -> Iterator[tuple[str, object]]:
    yield from _members(entity, "entity", _ENTITY_LISTS)
    for line in _each(entity, "lines"):
        yield from _members(line, "line", _LINE_LISTS)


def _payload_events(payload: Mapping) -> Iterator[Mapping]:
    """Every event a payload of any kind carries, a rewrite's buffered
    ability's run included."""
    yield from _each(payload, "events")
    yield from _each(payload, "buffered_events")
    for member in ("incoming", "outgoing", "event"):
        event = payload.get(member)
        if isinstance(event, Mapping):
            yield event


def _event(event: Mapping) -> Iterator[tuple[str, object]]:
    attributed = event.get("attributed_to")
    if attributed is not None and not _is_attribution(attributed):
        yield "event.attributed_to", attributed
    if _outside(event.get("duration"), _EVENT_DURATIONS):
        yield "event.duration", event.get("duration")
    yield from _event_params(event.get("type"), _block(event, "params"))


def _event_params(kind: object, params: Mapping) -> Iterator[tuple[str, object]]:
    for (event_type, name), allowed in _EVENT_PARAM_LISTS.items():
        if kind == event_type and _param_outside(event_type, name, allowed, params):
            yield f"{event_type}.{name}", params.get(name)


def _param_outside(
    event_type: str, name: str, allowed: frozenset, params: Mapping,
) -> bool:
    return _checks_param(event_type, name, params) and _outside(params.get(name), allowed)


def _is_attribution(attributed: object) -> bool:
    """What an event's ``attributed_to`` may name: a sentinel or a chain index."""
    return isinstance(attributed, str) and (
        attributed in _ATTRIBUTION_SENTINELS or attributed.isdigit())


def _checks_param(event_type: str, name: str, params: Mapping) -> bool:
    """Whether an event of ``event_type`` has ``name`` checked: where it
    carries it, and always on a ``zone_change``."""
    return name in params or event_type == EventType.ZONE_CHANGE
