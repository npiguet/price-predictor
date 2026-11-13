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

from collections.abc import Iterator, Mapping

from effects.domain.event_schema import EventType
from effects.domain.relation_types import FAMILIES
from effects.domain.state_snapshot import (
    CombatSubstep,
    DayNight,
    EffectDuration,
    EntityRole,
    FaceState,
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
_LINK_TYPES = frozenset(FAMILIES)
_ATTRIBUTION_SENTINELS = frozenset({"root", "unresolved"})


def off_list_values(raw: Mapping) -> Iterator[tuple[str, object]]:
    """``(member, value)`` for every value of one raw record outside its
    member's closed list; a member a record need not carry is checked only
    where present."""
    if raw.get("generation") == 3 and "mode" not in raw:
        yield "mode", ABSENT
    state = raw.get("state") or {}
    yield from _global(state.get("global") or {})
    for entity in state.get("entities") or ():
        yield from _entity(entity)
    for link in state.get("links") or ():
        if _outside(link.get("type"), _LINK_TYPES):
            yield "link.type", link.get("type")
    pending = state.get("pending_event")
    if pending:
        yield from _event(pending)
    for event in _payload_events(raw.get("payload") or {}):
        yield from _event(event)


def _outside(value: object, allowed: frozenset) -> bool:
    """Whether ``value`` is no member of ``allowed``; a list or an object
    where a string belongs is none."""
    try:
        return value not in allowed
    except TypeError:
        return True


def _global(block: Mapping) -> Iterator[tuple[str, object]]:
    phase = block.get("phase")
    if _outside(phase, _PHASES):
        yield "global.phase", phase
    if _outside(block.get("combat_substep"), _SUBSTEPS):
        yield "global.combat_substep", block.get("combat_substep")
    if _outside(block.get("day_night"), _DAY_NIGHT):
        yield "global.day_night", block.get("day_night")
    for member in ("floating_effects", "delayed_triggers"):
        for made in block.get(member) or ():
            if "duration" in made and _outside(made["duration"], _DURATIONS):
                yield f"global.{member}.duration", made["duration"]


def _entity(entity: Mapping) -> Iterator[tuple[str, object]]:
    if _outside(entity.get("zone"), _ZONES):
        yield "entity.zone", entity.get("zone")
    if _outside(entity.get("face_state"), _FACE_STATES):
        yield "entity.face_state", entity.get("face_state")
    if _outside(entity.get("role"), _ROLES):
        yield "entity.role", entity.get("role")
    for line in entity.get("lines") or ():
        if _outside(line.get("kind"), _LINE_KINDS):
            yield "line.kind", line.get("kind")


def _payload_events(payload: Mapping) -> Iterator[Mapping]:
    """Every event a payload of any kind carries."""
    yield from payload.get("events") or ()
    for member in ("incoming", "outgoing", "event"):
        event = payload.get(member)
        if isinstance(event, Mapping):
            yield event


def _event(event: Mapping) -> Iterator[tuple[str, object]]:
    attributed = event.get("attributed_to")
    if attributed is not None and not (
            isinstance(attributed, str)
            and (attributed in _ATTRIBUTION_SENTINELS or attributed.isdigit())):
        yield "event.attributed_to", attributed
    if event.get("type") == EventType.ZONE_CHANGE:
        params = event.get("params") or {}
        for end in ("from_zone", "to_zone"):
            if _outside(params.get(end), _ZONES):
                yield f"zone_change.{end}", params.get(end)
