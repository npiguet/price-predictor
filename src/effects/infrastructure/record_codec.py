"""The JSON codec for one effect record: its wire form and back.

Every conversion is written out rather than reflected from the dataclass, so
a field renamed in Python never silently renames itself in a corpus that
cannot be rebuilt. Reading and writing whole shards is ``shard_io``'s.

Each block of the record is one ``_x_to_json`` / ``_x_from_json`` pair here,
except the two values whose wire form more than one file shares: an
:class:`Event` (records, and the validator's event digests) and a
:class:`ProvenanceKey` (records, sidecars, the knowledge probes' frozen sets) serialise
themselves beside their fields, in the domain, so every format writes them
alike.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

import orjson

from effects.domain.event_schema import Event
from effects.domain.fork_operation import ForkOperation
from effects.domain.keyword_names import normalize_keyword
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    PAYLOAD_TYPES,
    SUBKINDS_OF,
    ActivationPayload,
    Candidate,
    CollectionMode,
    CombatPayload,
    ConditionMet,
    ContinuousPayload,
    Contribution,
    Costs,
    Curation,
    DirectedTag,
    EffectRecord,
    FailedCheck,
    ForbiddenEntity,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
    TurnActionPayload,
)
from effects.domain.state_snapshot import (
    CardChoices,
    CardHistory,
    CardLine,
    CombatStatus,
    CommandZoneObject,
    CostsPaid,
    Dungeon,
    EntityState,
    GlobalState,
    GrantedTemporary,
    InclusionTier,
    LineRef,
    Link,
    LinkEnd,
    PlayerState,
    PowerToughness,
    RandomResults,
    RealFace,
    Refs,
    StackExtras,
    StateSnapshot,
    TurnCounts,
    normalize_counter_type,
)

#: Envelope keys this code models. Anything else on a line is a field a newer
#: writer added and is kept in ``EffectRecord.extra_fields``.
_KNOWN_ENVELOPE_KEYS = frozenset({
    "record_id", "generation", "run_id", "timestamp", "game_id", "kind", "moment", "subkind",
    "link_id", "mirror_of", "variant_of", "mode", "interventional", "fork",
    "synthetic", "random_seat", "what_if", "aimed_at", "directed", "curation", "actor_player",
    "ability",
    "ability_unresolved", "state", "payload",
})


# ── provenance keys ─────────────────────────────────────────────────────


def _keys_to_json(keys: tuple[ProvenanceKey, ...]) -> list[dict]:
    return [key.as_dict() for key in keys]


#: Every key parsed so far, by its raw JSON fields: a shard repeats the same
#: few hundred keys thousands of times. Bounded by the card pool's traits.
_INTERNED_KEYS: dict[tuple, ProvenanceKey] = {}


def _key_from_json(item: dict) -> ProvenanceKey:
    raw = (
        item.get("script_file"), item.get("face"), item.get("trait_kind"),
        item.get("index_within_kind"), item.get("option"),
    )
    key = _INTERNED_KEYS.get(raw)
    if key is None:
        key = _INTERNED_KEYS[raw] = ProvenanceKey.from_dict(item)
    return key


def _keys_from_json(data: Any) -> tuple[ProvenanceKey, ...]:
    return tuple(_key_from_json(item) for item in data or ())


# ── events ──────────────────────────────────────────────────────────────


def _events_to_json(events: tuple[Event, ...]) -> list[dict]:
    return [event.as_dict() for event in events]


def _events_from_json(data: Any) -> tuple[Event, ...]:
    return tuple(_event_from_json(item) for item in data or ())


def _event_from_json(item: Any) -> Event:
    """One event, with its counter type in this package's spelling.

    Normalized on read, like keywords, because collected shards carry Forge's
    printed spelling and cannot be rewritten.
    """
    event = Event.from_dict(item)
    counter = event.params.get("counter_type")
    if isinstance(counter, str):
        return replace(
            event,
            params={**event.params, "counter_type": normalize_counter_type(counter)},
        )
    return event


# ── snapshot ────────────────────────────────────────────────────────────


def _optional[T, R](convert: Callable[[T], R], value: T | None) -> R | None:
    """``convert(value)``, or None for an absent block: null, or empty on read."""
    return convert(value) if value else None


def _pt_to_json(pt: PowerToughness) -> dict:
    return {
        "base": list(pt.base),
        "boosts": list(pt.boosts),
        "counters": list(pt.counters),
    }


def _pt_from_json(data: dict) -> PowerToughness:
    return PowerToughness(
        base=tuple(data["base"]),
        boosts=tuple(data.get("boosts", (0, 0))),
        counters=tuple(data.get("counters", (0, 0))),
    )


def _pt_boosts_to_json(pt: PowerToughness) -> dict:
    """Gen-3's ``pt_boosts``; ``base_pt`` is ``pt.base`` written alone."""
    return {"temporary": list(pt.boosts), "counters": list(pt.counters)}


def _gen3_pt_from_json(data: dict) -> PowerToughness | None:
    """Gen-3's ``base_pt`` and ``pt_boosts``, both null on a non-creature."""
    base = data.get("base_pt")
    if base is None:
        return None
    boosts = data.get("pt_boosts") or {}
    return PowerToughness(
        base=tuple(base),
        boosts=tuple(boosts.get("temporary", (0, 0))),
        counters=tuple(boosts.get("counters", (0, 0))),
    )


def _line_to_json(line: CardLine) -> dict:
    """One gen-3 line; the optional members only where set, as Java writes them."""
    out: dict = {"keys": _keys_to_json(line.keys), "kind": line.kind,
                 "granted": line.granted}
    if line.active is not None:
        out["active"] = line.active
    if line.cast_with is not None:
        out["cast_with"] = line.cast_with
    if line.chosen is not None:
        out["chosen"] = list(line.chosen)
    if line.fragment is not None:
        out["fragment"] = line.fragment
    if line.affected is not None:
        out["affected"] = list(line.affected)
    return out


def _line_from_json(data: dict) -> CardLine:
    return CardLine(
        keys=_keys_from_json(data.get("keys")),
        kind=data["kind"],
        granted=data.get("granted", False),
        active=data.get("active"),
        cast_with=data.get("cast_with"),
        chosen=_tuple_or_none(data.get("chosen")),
        fragment=data.get("fragment"),
        affected=_tuple_or_none(data.get("affected")),
    )


def _tuple_or_none(data: Any) -> tuple | None:
    """A list member as a tuple, keeping an absent or null one None."""
    return tuple(data) if data is not None else None


def _list_or_none(values: tuple | None) -> list | None:
    return list(values) if values is not None else None


def _absent_or[T, R](convert: Callable[[T], R], value: T | None) -> R | None:
    """``convert(value)``, or None for null; unlike :func:`_optional`, an
    empty block (gen-3's ``chosen: {}``) is a value."""
    return convert(value) if value is not None else None


# ── a gen-3 card's facts ────────────────────────────────────────────────


def _real_face_to_json(face: RealFace) -> dict:
    return {
        "name": face.name, "types": list(face.types), "subtypes": list(face.subtypes),
        "colors": list(face.colors), "mana_value": face.mana_value,
        "base_pt": _list_or_none(face.base_pt),
    }


def _real_face_from_json(data: dict) -> RealFace:
    return RealFace(
        name=data["name"], types=tuple(data.get("types", ())),
        subtypes=tuple(data.get("subtypes", ())), colors=tuple(data.get("colors", ())),
        mana_value=data.get("mana_value", 0), base_pt=_tuple_or_none(data.get("base_pt")),
    )


#: A card's ``chosen`` members in the order the collector writes them, each
#: only when set; the list-valued ones.
_CHOICE_MEMBERS = ("colors", "type", "type2", "number", "player", "cards", "names",
                   "direction", "mode", "even_odd", "sector", "stored_rolls")
_CHOICE_LISTS = frozenset({"colors", "cards", "names", "stored_rolls"})


def _choices_to_json(chosen: CardChoices) -> dict:
    out: dict = {}
    for name in _CHOICE_MEMBERS:
        value = getattr(chosen, name)
        if value is not None:
            out[name] = list(value) if name in _CHOICE_LISTS else value
    return out


def _choices_from_json(data: dict) -> CardChoices:
    return CardChoices(**{
        name: tuple(data[name]) if name in _CHOICE_LISTS else data[name]
        for name in _CHOICE_MEMBERS if data.get(name) is not None
    })


def _costs_paid_to_json(paid: CostsPaid) -> dict:
    return {"mana": dict(paid.mana), "cast_from": paid.cast_from,
            "mana_rider": paid.mana_rider}


def _costs_paid_from_json(data: dict) -> CostsPaid:
    return CostsPaid(mana=dict(data.get("mana", {})), cast_from=data.get("cast_from"),
                     mana_rider=data.get("mana_rider", False))


def _history_to_json(history: CardHistory) -> dict:
    return {
        "entered_this_turn": history.entered_this_turn,
        "attacked_this_turn": history.attacked_this_turn,
        "blocked_this_turn": history.blocked_this_turn,
        "loyalty_activated": history.loyalty_activated,
    }


def _history_from_json(data: dict) -> CardHistory:
    return CardHistory(
        entered_this_turn=data.get("entered_this_turn", False),
        attacked_this_turn=data.get("attacked_this_turn", False),
        blocked_this_turn=data.get("blocked_this_turn", False),
        loyalty_activated=data.get("loyalty_activated", False),
    )


def _combat_status_to_json(combat: CombatStatus) -> dict:
    return {
        "attacking": combat.attacking,
        "blocking": list(combat.blocking),
        "blocked_by": list(combat.blocked_by),
        "became_blocked": combat.became_blocked,
    }


def _combat_status_from_json(data: dict) -> CombatStatus:
    return CombatStatus(
        attacking=data.get("attacking"),
        blocking=tuple(data.get("blocking", ())),
        blocked_by=tuple(data.get("blocked_by", ())),
        became_blocked=data.get("became_blocked", False),
    )


def _granted_temporary_to_json(granted: GrantedTemporary) -> dict:
    return {
        "keywords": list(granted.keywords),
        "abilities": _keys_to_json(granted.abilities),
    }


def _granted_temporary_from_json(data: dict) -> GrantedTemporary:
    return GrantedTemporary(
        keywords=tuple(normalize_keyword(k) for k in data.get("keywords", ())),
        abilities=_keys_from_json(data.get("abilities")),
    )


def _stack_extras_to_json(extras: StackExtras) -> dict:
    return {
        "targets": list(extras.targets),
        "per_target_amounts": dict(extras.per_target_amounts),
        "up_to_counts": dict(extras.up_to_counts),
    }


def _stack_extras_from_json(data: dict) -> StackExtras:
    return StackExtras(
        targets=tuple(data.get("targets", ())),
        per_target_amounts=dict(data.get("per_target_amounts", {})),
        up_to_counts=dict(data.get("up_to_counts", {})),
    )


def _entity_to_json(entity: EntityState) -> dict:
    """One entity in the shape it was read in: gen-3 (``lines`` set) as the
    gen-3 collector writes it, else gen-2's, byte for byte."""
    head = {
        "id": entity.id, "name": entity.name, "zone": entity.zone,
        "controller": entity.controller, "face": entity.face,
        "copy_source": entity.copy_source,
        "token_script_id": entity.token_script_id,
        "types": list(entity.types), "subtypes": list(entity.subtypes),
        "supertypes": list(entity.supertypes), "colors": list(entity.colors),
        "mana_value": entity.mana_value,
    }
    status = {
        "tapped": entity.tapped, "sick": entity.sick, "damage": entity.damage,
        "counters": dict(entity.counters),
        "combat": _optional(_combat_status_to_json, entity.combat),
        "attached_to": entity.attached_to,
        "face_down": entity.face_down,
    }
    stack_extras = _optional(_stack_extras_to_json, entity.stack_extras)
    if entity.lines is not None:
        return _gen3_entity_to_json(entity, head, status, stack_extras)
    return {
        **head,
        "pt": _optional(_pt_to_json, entity.pt),
        **status,
        "granted_attached": _keys_to_json(entity.granted_attached),
        "granted_temporary": _granted_temporary_to_json(entity.granted_temporary),
        "printed": _keys_to_json(entity.printed),
        "stack_extras": stack_extras,
    }


def _gen3_entity_to_json(entity: EntityState, head: dict, status: dict,
                         stack_extras: dict | None) -> dict:
    """A gen-3 entity, its members in the order ``FE/EntityJson.java`` writes them."""
    pt = entity.pt
    return {
        "id": head["id"], "name": head["name"], "zone": head["zone"],
        "owner": entity.owner, "controller": head["controller"], "face": head["face"],
        "face_state": entity.face_state, "copy_source": head["copy_source"],
        "token_script_id": head["token_script_id"], "types": head["types"],
        "subtypes": head["subtypes"], "supertypes": head["supertypes"],
        "colors": head["colors"], "mana_value": head["mana_value"],
        "mana_cost": _absent_or(dict, entity.mana_cost),
        "base_pt": list(pt.base) if pt is not None else None,
        "pt_boosts": _pt_boosts_to_json(pt) if pt is not None else None,
        "printed_loyalty": entity.printed_loyalty,
        "printed_defense": entity.printed_defense,
        **status,
        "real_face": _absent_or(_real_face_to_json, entity.real_face),
        "designations": list(entity.designations),
        "x_paid": entity.x_paid,
        "chosen": _absent_or(_choices_to_json, entity.chosen),
        "costs_paid": _absent_or(_costs_paid_to_json, entity.costs_paid),
        "history": _absent_or(_history_to_json, entity.history),
        "order_rank": entity.order_rank,
        "granted_temporary": _granted_temporary_to_json(entity.granted_temporary),
        "lines": [_line_to_json(line) for line in entity.lines],
        "stack_extras": stack_extras,
        "role": entity.role,
    }


def _entity_from_json(data: dict) -> EntityState:
    """One entity of either shape; a ``lines`` member makes it gen-3, whose
    ``printed`` :class:`EntityState` derives from the lines."""
    lines = data.get("lines")
    if lines is not None:
        return _entity(
            data, pt=_gen3_pt_from_json(data),
            lines=tuple(_line_from_json(line) for line in lines),
            owner=data.get("owner"), face_state=data.get("face_state"),
            mana_cost=_absent_or(dict, data.get("mana_cost")),
            printed_loyalty=data.get("printed_loyalty"),
            printed_defense=data.get("printed_defense"),
            real_face=_absent_or(_real_face_from_json, data.get("real_face")),
            designations=tuple(data.get("designations", ())),
            x_paid=data.get("x_paid"),
            chosen=_absent_or(_choices_from_json, data.get("chosen")),
            costs_paid=_absent_or(_costs_paid_from_json, data.get("costs_paid")),
            history=_absent_or(_history_from_json, data.get("history")),
            order_rank=data.get("order_rank"),
            role=data.get("role"),
        )
    return _entity(
        data, pt=_optional(_pt_from_json, data.get("pt")),
        granted_attached=_keys_from_json(data.get("granted_attached")),
        printed=_keys_from_json(data.get("printed")),
    )


def _entity(data: dict, **shaped) -> EntityState:
    """The members both entity shapes share, plus the ``shaped`` ones."""
    return EntityState(
        id=data["id"], name=data["name"], zone=data["zone"],
        controller=data["controller"], face=data.get("face", 0),
        copy_source=data.get("copy_source"),
        token_script_id=data.get("token_script_id"),
        types=tuple(data.get("types", ())),
        subtypes=tuple(data.get("subtypes", ())),
        supertypes=tuple(data.get("supertypes", ())),
        colors=tuple(data.get("colors", ())),
        mana_value=data.get("mana_value", 0),
        tapped=data.get("tapped", False), sick=data.get("sick", False),
        damage=data.get("damage", 0), counters=dict(data.get("counters", {})),
        combat=_optional(_combat_status_from_json, data.get("combat")),
        attached_to=data.get("attached_to"),
        face_down=data.get("face_down", False),
        granted_temporary=_granted_temporary_from_json(
            data.get("granted_temporary") or {}
        ),
        stack_extras=_optional(_stack_extras_from_json, data.get("stack_extras")),
        **shaped,
    )


def _command_zone_object_to_json(made: CommandZoneObject) -> dict:
    return {
        "id": made.id, "source_lines": _keys_to_json(made.source_lines),
        "affected": list(made.affected), "remembered": list(made.remembered),
        "duration": made.duration, "shield_left": made.shield_left,
    }


def _command_zone_object_from_json(data: dict) -> CommandZoneObject:
    return CommandZoneObject(
        id=data["id"], source_lines=_keys_from_json(data.get("source_lines")),
        affected=tuple(data.get("affected", ())),
        remembered=tuple(data.get("remembered", ())),
        duration=data.get("duration", "permanent"), shield_left=data.get("shield_left"),
    )


def _dungeon_to_json(dungeon: Dungeon) -> dict:
    return {"player": dungeon.player, "dungeon": dungeon.dungeon, "room": dungeon.room}


def _dungeon_from_json(data: dict) -> Dungeon:
    return Dungeon(player=data["player"], dungeon=data["dungeon"], room=data.get("room"))


def _global_to_json(global_: GlobalState, gen3: bool) -> dict:
    out = {
        "turn": global_.turn,
        "phase": global_.phase,
        "active": global_.active,
        "priority": global_.priority,
        "stack_size": global_.stack_size,
        "combat_substep": global_.combat_substep,
        "emblems": _keys_to_json(global_.emblems),
    }
    if gen3:
        out |= {
            "day_night": global_.day_night,
            "floating_effects": [
                _command_zone_object_to_json(made) for made in global_.floating_effects],
            "delayed_triggers": [
                _command_zone_object_to_json(made) for made in global_.delayed_triggers],
            "dungeons": [_dungeon_to_json(dungeon) for dungeon in global_.dungeons],
        }
    return out


def _global_from_json(data: dict) -> GlobalState:
    return GlobalState(
        turn=data["turn"], phase=data["phase"], active=data["active"],
        priority=data.get("priority"), stack_size=data.get("stack_size", 0),
        combat_substep=data.get("combat_substep"),
        emblems=_keys_from_json(data.get("emblems")),
        day_night=data.get("day_night"),
        floating_effects=tuple(
            _command_zone_object_from_json(made) for made in data.get("floating_effects", ())),
        delayed_triggers=tuple(
            _command_zone_object_from_json(made) for made in data.get("delayed_triggers", ())),
        dungeons=tuple(_dungeon_from_json(dungeon) for dungeon in data.get("dungeons", ())),
    )


_TURN_COUNTS = ("cards_drawn", "life_gained", "life_lost", "damage_taken",
                "permanents_to_graveyard", "spells_cast_last_turn", "dungeons_completed",
                "descended")


def _turn_counts_to_json(counts: TurnCounts) -> dict:
    return {name: getattr(counts, name) for name in _TURN_COUNTS}


def _turn_counts_from_json(data: dict) -> TurnCounts:
    return TurnCounts(**{name: data.get(name, 0) for name in _TURN_COUNTS})


def _player_to_json(player: PlayerState, gen3: bool) -> dict:
    out = {
        "id": player.id, "life": player.life, "hand": player.hand,
        "library": player.library, "graveyard": player.graveyard,
        "poison": player.poison, "energy": player.energy,
        "this_turn": dict(player.this_turn),
    }
    # Right after this_turn, on a gen-3 player that carried it: a gen-3
    # record collected before the member existed writes back without it.
    if gen3 and (player.max_land_plays is not None or player.land_plays_unlimited):
        out["max_land_plays"] = player.max_land_plays
    out |= {
        "floating_mana": dict(player.floating_mana),
        "untapped_production": dict(player.untapped_production),
    }
    if gen3:
        out |= {
            "speed": player.speed, "city_blessing": player.city_blessing,
            "rad": player.rad, "monarch": player.monarch,
            "initiative": player.initiative,
            "turn_counts": _absent_or(_turn_counts_to_json, player.turn_counts),
        }
    return out


def _player_from_json(data: dict) -> PlayerState:
    return PlayerState(
        id=data["id"], life=data["life"], hand=data["hand"],
        library=data["library"], graveyard=data["graveyard"],
        poison=data.get("poison", 0), energy=data.get("energy", 0),
        this_turn=dict(data.get("this_turn", {})),
        floating_mana=dict(data.get("floating_mana", {})),
        untapped_production=dict(data.get("untapped_production", {})),
        max_land_plays=data.get("max_land_plays"),
        land_plays_unlimited="max_land_plays" in data and data["max_land_plays"] is None,
        speed=data.get("speed", 0), city_blessing=data.get("city_blessing", False),
        rad=data.get("rad", 0), monarch=data.get("monarch", False),
        initiative=data.get("initiative", False),
        turn_counts=_absent_or(_turn_counts_from_json, data.get("turn_counts")),
    )


def _random_to_json(random: RandomResults) -> dict:
    return {"dice": list(random.dice),
            "coin_flips": {"won": random.coin_flips_won, "lost": random.coin_flips_lost}}


def _random_from_json(data: dict) -> RandomResults:
    flips = data.get("coin_flips") or {}
    return RandomResults(dice=tuple(data.get("dice", ())),
                         coin_flips_won=flips.get("won", 0),
                         coin_flips_lost=flips.get("lost", 0))


def _refs_to_json(refs: Refs, gen3: bool) -> dict:
    out = {
        "targets": list(refs.targets),
        "source": refs.source,
        "modes": list(refs.modes),
        "x": refs.x,
        "choices": dict(refs.choices),
    }
    if gen3:
        out["random"] = _absent_or(_random_to_json, refs.random)
    return out


def _refs_from_json(data: dict) -> Refs:
    return Refs(
        targets=tuple(data.get("targets", ())),
        source=data.get("source"),
        modes=tuple(data.get("modes", ())),
        x=data.get("x"),
        choices=dict(data.get("choices", {})),
        random=_absent_or(_random_from_json, data.get("random")),
    )


def _link_end_to_json(end: LinkEnd) -> str | dict:
    if isinstance(end, LineRef):
        return {"entity": end.entity, "key": end.key.as_dict()}
    return end


def _link_end_from_json(data: str | dict) -> LinkEnd:
    if isinstance(data, dict):
        return LineRef(entity=data["entity"], key=_key_from_json(data["key"]))
    return data


def _link_to_json(link: Link) -> dict:
    out = {"from": _link_end_to_json(link.from_), "to": _link_end_to_json(link.to),
           "type": link.type}
    if link.index is not None:
        out["index"] = link.index
    return out


def _link_from_json(data: dict) -> Link:
    return Link(from_=_link_end_from_json(data["from"]), to=_link_end_from_json(data["to"]),
                type=data["type"], index=data.get("index"))


def _snapshot_to_json(state: StateSnapshot) -> dict:
    """A snapshot in the shape it was read in: ``links`` set makes it gen-3,
    which also writes the gen-3 members of the global block, the players and
    the refs."""
    gen3 = state.links is not None
    out: dict = {
        "global": _global_to_json(state.global_, gen3),
        "players": [_player_to_json(player, gen3) for player in state.players],
        "entities": [_entity_to_json(e) for e in state.entities],
    }
    if state.links is not None:
        out["links"] = [_link_to_json(link) for link in state.links]
    return out | {
        "refs": _refs_to_json(state.refs, gen3),
        "pending_event": (
            state.pending_event.as_dict() if state.pending_event else None
        ),
        "tiers": sorted(int(t) for t in state.tiers),
    }


def _snapshot_from_json(data: dict) -> StateSnapshot:
    pending = data.get("pending_event")
    links = data.get("links")
    return StateSnapshot(
        global_=_global_from_json(data["global"]),
        players=tuple(_player_from_json(p) for p in data.get("players", ())),
        entities=tuple(_entity_from_json(e) for e in data.get("entities", ())),
        refs=_refs_from_json(data.get("refs") or {}),
        pending_event=_event_from_json(pending) if pending else None,
        tiers=frozenset(InclusionTier(t) for t in data.get("tiers", ())),
        links=tuple(_link_from_json(link) for link in links) if links is not None else None,
    )


# ── payloads ────────────────────────────────────────────────────────────


def _costs_to_json(costs: Costs) -> dict:
    return {
        "mana_by_color": dict(costs.mana_by_color),
        "tapped": list(costs.tapped),
        "life": costs.life,
        "sacrificed": list(costs.sacrificed),
        "discarded": list(costs.discarded),
        "exiled": list(costs.exiled),
    }


def _costs_from_json(data: dict) -> Costs:
    return Costs(
        mana_by_color=dict(data.get("mana_by_color", {})),
        tapped=tuple(data.get("tapped", ())),
        life=data.get("life", 0),
        sacrificed=tuple(data.get("sacrificed", ())),
        discarded=tuple(data.get("discarded", ())),
        exiled=tuple(data.get("exiled", ())),
    )


def _activation_to_json(payload: ActivationPayload) -> dict:
    return {"costs": _costs_to_json(payload.costs), "outcome": payload.outcome.value}


def _activation_from_json(data: dict) -> ActivationPayload:
    return ActivationPayload(
        costs=_costs_from_json(data.get("costs") or {}),
        outcome=ResolutionOutcome(data["outcome"]),
    )


def _resolution_to_json(payload: ResolutionPayload) -> dict:
    out: dict = {"events": _events_to_json(payload.events)}
    if payload.conditions_met is not None:
        out["conditions_met"] = [
            {"segment": entry.segment, "met": entry.met} for entry in payload.conditions_met]
    return out


def _resolution_from_json(data: dict) -> ResolutionPayload:
    conditions = data.get("conditions_met")
    return ResolutionPayload(
        events=_events_from_json(data.get("events")),
        conditions_met=tuple(
            ConditionMet(segment=entry["segment"], met=entry["met"]) for entry in conditions
        ) if conditions is not None else None,
    )


def _rewrite_to_json(payload: RewritePayload) -> dict:
    return {
        "incoming": payload.incoming.as_dict(),
        "outgoing": payload.outgoing.as_dict() if payload.outgoing else None,
        "result": payload.result.value if payload.result else None,
        "replaced_by": _keys_to_json(payload.replaced_by),
    }


def _rewrite_from_json(data: dict) -> RewritePayload:
    outgoing = data.get("outgoing")
    result = data.get("result")
    return RewritePayload(
        incoming=_event_from_json(data["incoming"]),
        outgoing=_event_from_json(outgoing) if outgoing else None,
        result=RewriteResult(result) if result else None,
        replaced_by=_keys_from_json(data.get("replaced_by")),
    )


def _contribution_to_json(contribution: Contribution) -> dict:
    return {
        "entity": contribution.entity, "pt_boost": list(contribution.pt_boost),
        "keywords": list(contribution.keywords), "types": list(contribution.types),
        "colors": list(contribution.colors), "name": contribution.name,
    }


def _contribution_from_json(data: dict) -> Contribution:
    return Contribution(
        entity=data["entity"],
        pt_boost=tuple(data.get("pt_boost", (0, 0))),
        keywords=tuple(normalize_keyword(k) for k in data.get("keywords", ())),
        types=tuple(data.get("types", ())),
        colors=tuple(data.get("colors", ())),
        name=data.get("name"),
    )


def _continuous_to_json(payload: ContinuousPayload) -> dict:
    return {
        "contributions": [_contribution_to_json(c) for c in payload.contributions],
        "board_hash": payload.board_hash,
    }


def _continuous_from_json(data: dict) -> ContinuousPayload:
    return ContinuousPayload(
        contributions=tuple(
            _contribution_from_json(c) for c in data.get("contributions", ())
        ),
        board_hash=data.get("board_hash", ""),
    )


def _combat_to_json(payload: CombatPayload) -> dict:
    return {
        "attackers": list(payload.attackers),
        "blocks": {k: list(v) for k, v in payload.blocks.items()},
        "assignment_choices": {
            k: dict(v) for k, v in payload.assignment_choices.items()
        },
        "events": _events_to_json(payload.events),
        "probed_keyword": payload.probed_keyword,
        "probed_entity": payload.probed_entity,
    }


def _is_observed_combat(payload) -> bool:
    return (isinstance(payload, CombatPayload)
            and payload.probed_keyword is None and payload.probed_entity is None)


def _combat_from_json(data: dict) -> CombatPayload:
    return CombatPayload(
        attackers=tuple(data.get("attackers", ())),
        blocks={k: tuple(v) for k, v in (data.get("blocks") or {}).items()},
        assignment_choices={
            k: dict(v)
            for k, v in (data.get("assignment_choices") or {}).items()
        },
        events=_events_from_json(data.get("events")),
        probed_keyword=data.get("probed_keyword"),
        probed_entity=data.get("probed_entity"),
    )


def _trigger_to_json(payload: TriggerPayload) -> dict:
    return {"event": payload.event.as_dict(), "fired": payload.fired}


def _trigger_from_json(data: dict) -> TriggerPayload:
    return TriggerPayload(
        event=_event_from_json(data["event"]), fired=bool(data["fired"]),
    )


def _candidate_to_json(candidate: Candidate) -> dict:
    """One candidate; ``timing``, ``failed_checks`` and ``land_play`` only
    where read, as gen-3 writes them."""
    verdict = {
        "can_play": candidate.can_play,
        "affordable": candidate.affordable,
        "has_legal_target": candidate.has_legal_target,
    }
    if candidate.timing is not None:
        verdict["timing"] = candidate.timing
    out = {
        "ability": _keys_to_json(candidate.ability),
        "verdict": verdict,
        "legal_targets": list(candidate.legal_targets),
        "cost_after_adjustment": dict(candidate.cost_after_adjustment),
        "responsible_static": _keys_to_json(candidate.responsible_static),
    }
    if candidate.failed_checks is not None:
        out["failed_checks"] = [
            {"check": failed.check,
             "responsible_static": _keys_to_json(failed.responsible_static)}
            for failed in candidate.failed_checks]
    if candidate.land_play is not None:
        out["land_play"] = candidate.land_play
    return out


def _candidate_from_json(data: dict) -> Candidate:
    verdict = data["verdict"]
    failed = data.get("failed_checks")
    return Candidate(
        ability=_keys_from_json(data.get("ability")),
        can_play=verdict["can_play"],
        affordable=verdict["affordable"],
        has_legal_target=verdict["has_legal_target"],
        legal_targets=tuple(data.get("legal_targets", ())),
        cost_after_adjustment=dict(data.get("cost_after_adjustment", {})),
        responsible_static=_keys_from_json(data.get("responsible_static")),
        timing=verdict.get("timing"),
        failed_checks=tuple(
            FailedCheck(check=entry["check"],
                        responsible_static=_keys_from_json(entry.get("responsible_static")))
            for entry in failed) if failed is not None else None,
        land_play=data.get("land_play"),
    )


def _decision_to_json(payload: PlayabilityDecisionPayload) -> dict:
    return {"candidates": [_candidate_to_json(c) for c in payload.candidates]}


def _decision_from_json(data: dict) -> PlayabilityDecisionPayload:
    return PlayabilityDecisionPayload(
        candidates=tuple(_candidate_from_json(c) for c in data.get("candidates", ())),
    )


def _forbidden_to_json(forbidden: tuple[ForbiddenEntity, ...]) -> list[dict]:
    return [
        {
            "entity": f.entity,
            "responsible_static": _keys_to_json(f.responsible_static),
        }
        for f in forbidden
    ]


def _forbidden_from_json(data: Any) -> tuple[ForbiddenEntity, ...]:
    return tuple(
        ForbiddenEntity(
            entity=f["entity"],
            responsible_static=_keys_from_json(f.get("responsible_static")),
        )
        for f in data or ()
    )


def _attackers_to_json(payload: PlayabilityAttackersPayload) -> dict:
    """An anchored (gen-3) record writes its anchor's members first."""
    out: dict = {}
    if payload.anchor_attacker is not None:
        out = {
            "anchor_attacker": payload.anchor_attacker,
            "legal_defenders": list(payload.legal_defenders),
            "must_attack": payload.must_attack,
        }
        if payload.tapped_by_attack is not None:
            out["tapped_by_attack"] = payload.tapped_by_attack
    return out | {
        "legal_attackers": list(payload.legal_attackers),
        "forbidden": _forbidden_to_json(payload.forbidden),
    }


def _attackers_from_json(data: dict) -> PlayabilityAttackersPayload:
    return PlayabilityAttackersPayload(
        legal_attackers=tuple(data.get("legal_attackers", ())),
        forbidden=_forbidden_from_json(data.get("forbidden")),
        anchor_attacker=data.get("anchor_attacker"),
        legal_defenders=tuple(data.get("legal_defenders", ())),
        must_attack=data.get("must_attack", False),
        tapped_by_attack=data.get("tapped_by_attack"),
    )


def _blockers_to_json(payload: PlayabilityBlockersPayload) -> dict:
    out = {
        "anchor_attacker": payload.anchor_attacker,
        "legal_blockers": list(payload.legal_blockers),
        "forbidden": _forbidden_to_json(payload.forbidden),
    }
    if payload.must_block is not None:
        out["must_block"] = list(payload.must_block)
    out["min_blockers"] = payload.min_blockers
    return out


def _blockers_from_json(data: dict) -> PlayabilityBlockersPayload:
    return PlayabilityBlockersPayload(
        anchor_attacker=data["anchor_attacker"],
        legal_blockers=tuple(data.get("legal_blockers", ())),
        forbidden=_forbidden_from_json(data.get("forbidden")),
        min_blockers=data.get("min_blockers", 0),
        must_block=_tuple_or_none(data.get("must_block")),
    )


def _turn_action_to_json(payload: TurnActionPayload) -> dict:
    return {"events": _events_to_json(payload.events)}


def _turn_action_from_json(data: dict) -> TurnActionPayload:
    return TurnActionPayload(events=_events_from_json(data.get("events")))


@dataclass(frozen=True, slots=True)
class _PayloadCodec:
    encode: Callable[[Any], dict]
    decode: Callable[[dict], Any]


#: Every payload type's wire form: the values of ``PAYLOAD_TYPES``, each once.
_PAYLOAD_CODECS: dict[type, _PayloadCodec] = {
    ActivationPayload: _PayloadCodec(_activation_to_json, _activation_from_json),
    ResolutionPayload: _PayloadCodec(_resolution_to_json, _resolution_from_json),
    RewritePayload: _PayloadCodec(_rewrite_to_json, _rewrite_from_json),
    ContinuousPayload: _PayloadCodec(_continuous_to_json, _continuous_from_json),
    CombatPayload: _PayloadCodec(_combat_to_json, _combat_from_json),
    TriggerPayload: _PayloadCodec(_trigger_to_json, _trigger_from_json),
    PlayabilityDecisionPayload: _PayloadCodec(_decision_to_json, _decision_from_json),
    PlayabilityAttackersPayload: _PayloadCodec(_attackers_to_json, _attackers_from_json),
    PlayabilityBlockersPayload: _PayloadCodec(_blockers_to_json, _blockers_from_json),
    TurnActionPayload: _PayloadCodec(_turn_action_to_json, _turn_action_from_json),
}


def _payload_to_json(payload) -> dict:
    codec = _PAYLOAD_CODECS.get(type(payload))
    if codec is None:
        raise TypeError(f"no wire form for payload type {type(payload).__name__}")
    return codec.encode(payload)


def _payload_from_json(
    data: dict, kind: RecordKind, discriminator: Moment | StrEnum | None,
):
    payload_type = PAYLOAD_TYPES.get((kind, discriminator))
    if payload_type is None:
        # A ValueError, as the record's own discriminator check raises.
        raise ValueError(
            f"no payload is defined for kind={kind.value} with "
            f"moment/subkind={discriminator.value if discriminator else None}"
        )
    return _PAYLOAD_CODECS[payload_type].decode(data)


# ── records ─────────────────────────────────────────────────────────────


def record_to_dict(record: EffectRecord) -> dict:
    """One record's JSON-serializable dict form, envelope keys first."""
    out: dict = {"record_id": record.record_id}
    # First after record_id where set: its value names a gen-3 shard (FR-018).
    if record.generation is not None:
        out["generation"] = record.generation
    out |= {
        "run_id": record.run_id,
        "timestamp": record.timestamp,
        "game_id": record.game_id,
        "kind": record.kind.value,
        "moment": record.moment.value if record.moment else None,
        "subkind": record.subkind.value if record.subkind else None,
        "link_id": record.link_id,
        "mirror_of": record.mirror_of,
        "variant_of": record.variant_of,
        "mode": record.mode.value,
        "interventional": record.interventional,
        "fork": record.fork,
        "synthetic": record.synthetic,
    }
    # Written only where the source carried it: its presence makes a shard
    # gen-2 (``shard_generation``), so a rewritten gen-1 record stays gen-1.
    if record.random_seat is not None:
        out["random_seat"] = record.random_seat
    if record.what_if is not None:
        out["what_if"] = record.what_if
    if record.aimed_at is not None:
        out["aimed_at"] = list(record.aimed_at)
    if record.directed is not None:
        out["directed"] = {"variant": record.directed.variant,
                           "operation": record.directed.operation.value}
    if record.curation is not None:
        out["curation"] = _curation_to_json(record.curation)
    out |= {
        "actor_player": record.actor_player,
        "ability": (
            _keys_to_json(record.ability) if record.ability is not None else None
        ),
        "ability_unresolved": record.ability_unresolved,
        "state": _snapshot_to_json(record.state),
        "payload": _payload_to_json(record.payload),
    }
    if record.generation is not None and _is_observed_combat(record.payload):
        # The gen-3 collector writes no probe member on an observed combat.
        # Earlier generations keep the nulls this codec has always written,
        # which the pinned bytes of their fixtures hold.
        del out["payload"]["probed_keyword"], out["payload"]["probed_entity"]
    for key, value in record.extra_fields.items():
        out.setdefault(key, value)
    return out


def _curation_to_json(curation: Curation) -> dict:
    return {
        "natural_share": curation.natural_share,
        "copies": curation.copies,
        "drawn_rule": curation.drawn_rule,
        "drawn_rules": list(curation.drawn_rules),
        "condition_entities": list(curation.condition_entities),
        "outcome_entities": list(curation.outcome_entities),
    }


def _curation_from_json(data: dict) -> Curation:
    return Curation(
        natural_share=int(data["natural_share"]),
        copies=int(data["copies"]),
        drawn_rule=data.get("drawn_rule"),
        drawn_rules=tuple(data.get("drawn_rules") or ()),
        condition_entities=tuple(data.get("condition_entities") or ()),
        outcome_entities=tuple(data.get("outcome_entities") or ()),
    )


def record_from_dict(data: dict) -> EffectRecord:
    """Parse one shard line into an :class:`EffectRecord`."""
    kind = RecordKind(data["kind"])
    moment = Moment(data["moment"]) if data.get("moment") else None
    subkinds = SUBKINDS_OF.get(kind, PlayabilitySubkind)
    subkind = subkinds(data["subkind"]) if data.get("subkind") else None
    ability = data.get("ability")
    aimed_at = data.get("aimed_at")
    directed = data.get("directed")
    curation = data.get("curation")
    return EffectRecord(
        record_id=data["record_id"],
        run_id=data["run_id"],
        timestamp=data["timestamp"],
        game_id=data["game_id"],
        kind=kind,
        actor_player=data["actor_player"],
        state=_snapshot_from_json(data["state"]),
        payload=_payload_from_json(data["payload"], kind, moment or subkind),
        mode=CollectionMode(data.get("mode", CollectionMode.DEGRADED.value)),
        moment=moment,
        subkind=subkind,
        link_id=data.get("link_id"),
        mirror_of=data.get("mirror_of"),
        variant_of=data.get("variant_of"),
        interventional=data.get("interventional", False),
        fork=data.get("fork", False),
        synthetic=data.get("synthetic", False),
        random_seat=data.get("random_seat"),
        what_if=data.get("what_if"),
        generation=data.get("generation"),
        aimed_at=tuple(aimed_at) if aimed_at is not None else None,
        directed=(DirectedTag(directed["variant"], ForkOperation(directed["operation"]))
                  if directed is not None else None),
        curation=_curation_from_json(curation) if curation is not None else None,
        ability=_keys_from_json(ability) if ability is not None else None,
        ability_unresolved=data.get("ability_unresolved"),
        extra_fields={
            k: v for k, v in data.items() if k not in _KNOWN_ENVELOPE_KEYS
        },
    )


def format_record_line(record: EffectRecord) -> str:
    """Render one shard line: compact JSON in field order, no trailing newline.

    ``orjson``, as the reader is; non-ASCII text is written as UTF-8 rather
    than ``\\u`` escapes, which every reader decodes alike. A non-string key
    is written as its string, as ``json`` would.
    """
    return orjson.dumps(record_to_dict(record), option=orjson.OPT_NON_STR_KEYS).decode()


def parse_record_line(text: str) -> dict:
    """One shard line as the dict ``record_from_dict`` reads.

    ``orjson`` for speed; it gives the same dicts as ``json``, and its decode
    error subclasses ``json.JSONDecodeError``.
    """
    return orjson.loads(text)
