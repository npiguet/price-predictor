"""Categories for members a pilot varies on few records, so the fixture
varies them too (``field-coverage`` reads each): an event of a combat or a
turn-based action that a clause made or that lasts, a buffered event other
than a counter change, a rewrite rewriting into another event than damage, a
blocker needed twice or forbidden by a line keying to nothing, a spell
targeting a spell, a choice made, a battle, an emblem, energy, unlimited land
plays, a static's contribution of each rarer kind; the shapes
``effect_family`` names a family by: a charm's cost half, a rewrite through a
damage replacement line, a blocker a ``CantBlock`` static forbids; and a
contribution adding a colour, which tells the catalogue's "sets the colours"
declaration from its "changes a colour" one; the game-end record; an Aura
whose host an effect touches and leaves in place; a zone change a replacement
replaced and a rewrite through a keyword's line; an activated line's effect
half dealing damage to any target; and an attackers record listing no legal
attacker beside a noncreature permanent. Matched last, so they take only what
no other category took."""

from __future__ import annotations

import re
from collections.abc import Callable

from fixture_records.shapes import (
    acting_key,
    entities,
    events,
    global_state,
    is_effect_resolution,
    players,
)
from fixture_records.sidecars import Sidecars
from fixture_records.state_based import aura_host_stays

Test = Callable[[dict], bool]

_CANT_BLOCK = re.compile(r"Mode\$ CantBlock(?!By)\b")
_DAMAGE_REPLACEMENT = "Event$ DamageDone"


def _event_made_or_lasting(event: dict) -> bool:
    return event.get("attributed_to") != "unresolved" or event.get("duration") != "instant"


def _kind_event(kind: str, test: Callable[[dict], bool]) -> Test:
    return lambda record: record["kind"] == kind and any(test(event) for event in events(record))


def _buffered_other_than_counters(record: dict) -> bool:
    return record["kind"] == "rewrite" and any(
        event["type"] != "counter_change" or event.get("attributed_to") != "root"
        for event in record["payload"].get("buffered_events") or ())


def _outgoing_other_than_damage(record: dict) -> bool:
    outgoing = record["payload"].get("outgoing") if record["kind"] == "rewrite" else None
    return outgoing is not None and outgoing["type"] != "damage_dealt"


def _blockers(test: Callable[[dict], bool]) -> Test:
    return lambda record: (record["kind"] == "playability"
                           and record.get("subkind") == "blockers" and test(record["payload"]))


def _needs_two_blockers(payload: dict) -> bool:
    return payload.get("min_blockers") not in (None, 1)


def _forbidden_unresolved(payload: dict) -> bool:
    return any(entry.get("responsible_unresolved") for entry in payload.get("forbidden") or ())


def _refs_member(name: str) -> Test:
    return lambda record: bool((record["state"].get("refs") or {}).get(name))


def _battle(record: dict) -> bool:
    return any(entity.get("printed_defense") is not None for entity in entities(record))


def _emblem(record: dict) -> bool:
    return bool(global_state(record).get("emblems"))


def _energy(record: dict) -> bool:
    return any(player.get("energy") for player in players(record))


def _land_plays_unlimited(record: dict) -> bool:
    return any("max_land_plays" in player and player["max_land_plays"] is None
               for player in players(record))


def _contribution(member: str) -> Test:
    """A continuous record one of whose contributions sets ``member``."""
    return lambda record: record["kind"] == "continuous" and any(
        contribution.get(member) for contribution in record["payload"].get("contributions") or ())


def _colour_added(record: dict) -> bool:
    """A continuous record one of whose contributions adds a colour rather
    than setting the colours outright (no ``=``)."""
    return record["kind"] == "continuous" and any(
        contribution.get("colors") and "=" not in contribution["colors"]
        for contribution in record["payload"].get("contributions") or ())


def _rewrite_of(event_type: str, result: str) -> Test:
    return lambda record: (record["kind"] == "rewrite"
                           and record["payload"]["incoming"]["type"] == event_type
                           and record["payload"].get("result") == result)


def _keyword_rewrite(record: dict) -> bool:
    key = acting_key(record)
    return record["kind"] == "rewrite" and key is not None and key["trait_kind"] == "keyword"


def _first_forbidden_anchor(payload: dict) -> bool:
    forbidden = payload.get("forbidden") or ()
    return (not payload.get("legal_attackers") and bool(forbidden)
            and forbidden[0]["entity"] == payload.get("anchor_attacker"))


def _active_noncreature(record: dict) -> bool:
    active = global_state(record).get("active")
    return any(entity.get("zone") == "battlefield" and entity.get("controller") == active
               and "creature" not in (entity.get("types") or ())
               for entity in entities(record))


def _no_legal_attacker(record: dict) -> bool:
    """An anchored attackers record listing no legal attacker, its anchor the
    first forbidden one, on a board where the active player controls a
    noncreature permanent: the record that judges the unlisted permanents."""
    return (record["kind"] == "playability" and record.get("subkind") == "attackers"
            and _first_forbidden_anchor(record["payload"]) and _active_noncreature(record))


def _game_end_record(record: dict) -> bool:
    """The collector's game-end record: an effect half naming no acting line,
    no reason for it, no cost half and no outcome."""
    return (record["kind"] == "resolution" and record.get("moment") == "resolution"
            and not record.get("ability") and record.get("ability_unresolved") is None
            and record.get("link_id") is None and record["payload"].get("outcome") is None)


class _FamilyShapes:
    """The records whose family only the acting or responsible line's script names."""

    def __init__(self, sidecars: Sidecars) -> None:
        self.sidecars = sidecars

    def charm_cost_half(self, record: dict) -> bool:
        key = acting_key(record)
        return (record["kind"] == "resolution" and record.get("moment") == "activation"
                and key is not None and self.sidecars.api_type(key) == "Charm")

    def damage_replacement_rewrite(self, record: dict) -> bool:
        key = acting_key(record)
        return (record["kind"] == "rewrite" and key is not None
                and key["trait_kind"] == "replacement"
                and _DAMAGE_REPLACEMENT in self.sidecars.script(key))

    def activated_ping(self, record: dict) -> bool:
        """An effect half of an activated line dealing damage to any target."""
        key = acting_key(record)
        line = self.sidecars.line(key) if key is not None else None
        script = (line.get("script_text") or "") if line is not None else ""
        return (is_effect_resolution(record) and line is not None
                and line.get("line_kind") == "activated"
                and script.startswith("AB$ DealDamage") and "ValidTgts$ Any" in script)

    def cant_block_forbidden(self, record: dict) -> bool:
        return _blockers(self._forbids_by_cant_block)(record)

    def _forbids_by_cant_block(self, payload: dict) -> bool:
        return any(_CANT_BLOCK.search(self.sidecars.script(key))
                   for entry in payload.get("forbidden") or ()
                   for key in entry.get("responsible_static") or ())


def rare_member_categories(sidecars: Sidecars) -> dict[str, Test]:
    """The categories, in the order they are matched."""
    shapes = _FamilyShapes(sidecars)
    return {
        "combat-event-made-or-lasting": _kind_event("combat", _event_made_or_lasting),
        "turn-action-event-lasting": _kind_event(
            "turn_action", lambda event: event.get("duration") != "instant"),
        "rewrite-buffered-other": _buffered_other_than_counters,
        "rewrite-outgoing-other": _outgoing_other_than_damage,
        "blockers-min-two": _blockers(_needs_two_blockers),
        "blockers-forbidden-unresolved": _blockers(_forbidden_unresolved),
        "targets-on-stack": _refs_member("targets_on_stack"),
        "choices-made": _refs_member("choices"),
        "battle": _battle,
        "emblem": _emblem,
        "energy": _energy,
        "land-plays-unlimited": _land_plays_unlimited,
        **{f"contribution-{member}": _contribution(member)
           for member in ("set_pt", "colors", "controller", "hidden_keywords")},
        "charm-cost-half": shapes.charm_cost_half,
        "rewrite-damage-replacement": shapes.damage_replacement_rewrite,
        "blockers-cant-block": shapes.cant_block_forbidden,
        "contribution-colour-added": _colour_added,
        "game-end-record": _game_end_record,
        "aura-host-stays": aura_host_stays,
        "rewrite-zone-change-replaced": _rewrite_of("zone_change", "replaced"),
        "rewrite-keyword-line": _keyword_rewrite,
        "activated-ping": shapes.activated_ping,
        "attackers-none-legal": _no_legal_attacker,
    }
