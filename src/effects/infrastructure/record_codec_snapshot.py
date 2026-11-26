"""The wire form of a record's board snapshot: the global block, players,
refs, links and the snapshot that holds them and the entities.

Part of :mod:`effects.infrastructure.record_codec`; an entity's own form
is :mod:`effects.infrastructure.record_codec_entity`'s.
"""

from __future__ import annotations

from effects.domain.gen3_members import Site, owes
from effects.domain.state_snapshot import (
    CommandZoneObject,
    Dungeon,
    GlobalState,
    InclusionTier,
    LineRef,
    Link,
    LinkEnd,
    PlayerState,
    RandomResults,
    Refs,
    StateSnapshot,
    TurnCounts,
)
from effects.infrastructure.record_codec_entity import _entity_from_json, _entity_to_json
from effects.infrastructure.record_codec_values import (
    _absent_or,
    _event_from_json,
    _key_from_json,
    _keys_from_json,
    _keys_to_json,
    _list_or_none,
    _tuple_or_none,
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


def _global_to_json(global_: GlobalState) -> dict:
    return {
        "turn": global_.turn,
        "phase": global_.phase,
        "active": global_.active,
        "priority": global_.priority,
        "stack_size": global_.stack_size,
        "combat_substep": global_.combat_substep,
        "emblems": _keys_to_json(global_.emblems),
        "day_night": global_.day_night,
        "floating_effects": [
            _command_zone_object_to_json(made) for made in global_.floating_effects],
        "delayed_triggers": [
            _command_zone_object_to_json(made) for made in global_.delayed_triggers],
        "dungeons": [_dungeon_to_json(dungeon) for dungeon in global_.dungeons],
    }


@owes(Site.GLOBAL)
def _global_from_json(data: dict) -> GlobalState:
    return GlobalState(
        turn=data["turn"], phase=data["phase"], active=data["active"],
        priority=data["priority"], stack_size=data["stack_size"],
        combat_substep=data["combat_substep"],
        emblems=_keys_from_json(data["emblems"]),
        day_night=data["day_night"],
        floating_effects=tuple(
            _command_zone_object_from_json(made) for made in data["floating_effects"]),
        delayed_triggers=tuple(
            _command_zone_object_from_json(made) for made in data["delayed_triggers"]),
        dungeons=tuple(_dungeon_from_json(dungeon) for dungeon in data["dungeons"]),
    )


_TURN_COUNTS = ("cards_drawn", "life_gained", "life_lost", "damage_taken",
                "permanents_to_graveyard", "spells_cast_last_turn", "dungeons_completed",
                "descended")


def _turn_counts_to_json(counts: TurnCounts) -> dict:
    return {name: getattr(counts, name) for name in _TURN_COUNTS}


def _turn_counts_from_json(data: dict) -> TurnCounts:
    return TurnCounts(**{name: data.get(name, 0) for name in _TURN_COUNTS})


def _player_to_json(player: PlayerState) -> dict:
    out = {
        "id": player.id, "life": player.life, "hand": player.hand,
        "library": player.library, "graveyard": player.graveyard,
        "poison": player.poison, "energy": player.energy,
        "this_turn": dict(player.this_turn),
    }
    # Right after this_turn, on a player that carried it: a record collected
    # before the member existed writes back without it.
    if player.max_land_plays is not None or player.land_plays_unlimited:
        out["max_land_plays"] = player.max_land_plays
    return out | {
        "floating_mana": dict(player.floating_mana),
        "untapped_production": dict(player.untapped_production),
        "speed": player.speed, "city_blessing": player.city_blessing,
        "rad": player.rad, "monarch": player.monarch,
        "initiative": player.initiative,
        "turn_counts": _absent_or(_turn_counts_to_json, player.turn_counts),
    }


@owes(Site.PLAYER)
def _player_from_json(data: dict) -> PlayerState:
    return PlayerState(
        id=data["id"], life=data["life"], hand=data["hand"],
        library=data["library"], graveyard=data["graveyard"],
        poison=data["poison"], energy=data["energy"],
        this_turn=dict(data["this_turn"]),
        floating_mana=dict(data["floating_mana"]),
        untapped_production=dict(data["untapped_production"]),
        max_land_plays=data["max_land_plays"],
        land_plays_unlimited=data["max_land_plays"] is None,
        speed=data["speed"], city_blessing=data["city_blessing"],
        rad=data["rad"], monarch=data["monarch"],
        initiative=data["initiative"],
        turn_counts=_absent_or(_turn_counts_from_json, data["turn_counts"]),
    )


def _random_to_json(random: RandomResults) -> dict:
    return {"dice": list(random.dice),
            "coin_flips": {"won": random.coin_flips_won, "lost": random.coin_flips_lost}}


def _random_from_json(data: dict) -> RandomResults:
    flips = data.get("coin_flips") or {}
    return RandomResults(dice=tuple(data.get("dice", ())),
                         coin_flips_won=flips.get("won", 0),
                         coin_flips_lost=flips.get("lost", 0))


def _refs_to_json(refs: Refs) -> dict:
    out: dict = {"targets": list(refs.targets)}
    # Both or neither, as the collector writes them since it wrote either.
    if refs.targets_on_stack is not None or refs.targets_at_cast is not None:
        out["targets_at_cast"] = _list_or_none(refs.targets_at_cast)
        out["targets_on_stack"] = list(refs.targets_on_stack or ())
    return out | {
        "source": refs.source,
        "modes": list(refs.modes),
        "x": refs.x,
        "choices": dict(refs.choices),
        "random": _absent_or(_random_to_json, refs.random),
    }


@owes(Site.REFS)
def _refs_from_json(data: dict) -> Refs:
    return Refs(
        targets=tuple(data["targets"]),
        targets_at_cast=_tuple_or_none(data["targets_at_cast"]),
        targets_on_stack=_tuple_or_none(data["targets_on_stack"]),
        source=data["source"],
        modes=tuple(data["modes"]),
        x=data["x"],
        choices=dict(data["choices"]),
        random=_absent_or(_random_from_json, data["random"]),
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


@owes(Site.LINK)
def _link_from_json(data: dict) -> Link:
    return Link(from_=_link_end_from_json(data["from"]), to=_link_end_from_json(data["to"]),
                type=data["type"], index=data.get("index"))


def _snapshot_to_json(state: StateSnapshot) -> dict:
    out: dict = {
        "global": _global_to_json(state.global_),
        "players": [_player_to_json(player) for player in state.players],
        "entities": [_entity_to_json(e) for e in state.entities],
    }
    return out | {
        "links": [_link_to_json(link) for link in state.links],
        "refs": _refs_to_json(state.refs),
        "pending_event": (
            state.pending_event.as_dict() if state.pending_event else None
        ),
        "tiers": sorted(int(t) for t in state.tiers),
    }


@owes(Site.STATE)
def _snapshot_from_json(data: dict) -> StateSnapshot:
    pending = data["pending_event"]
    return StateSnapshot(
        global_=_global_from_json(data["global"]),
        players=tuple(_player_from_json(p) for p in data["players"]),
        entities=tuple(_entity_from_json(e) for e in data["entities"]),
        refs=_refs_from_json(data["refs"]),
        pending_event=_event_from_json(pending) if pending else None,
        tiers=frozenset(InclusionTier(t) for t in data["tiers"]),
        links=tuple(_link_from_json(link) for link in data["links"]),
    )
