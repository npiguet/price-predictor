"""Cut a small, kind-balanced sample of real records into a test fixture.

Tests of the gen-2 readers, families, value targets and probes need records
shaped exactly like the collector's output, so the fixture is cut from real
shards rather than built by hand. Every category below is filled with a few
records; a chosen cost half brings its effect half, a chosen probe brings the
combat record it mirrors, and a chosen fired trigger brings the resolution
records that follow it in the same game, so the joins the probes make (an
interaction, a link) have both ends present.

The sidecars and converted texts every chosen record names, anywhere in its
envelope, state or payload, are copied beside it under the same tree layout,
so a ``SidecarCache`` rooted there resolves every key.

    python scripts/make_fixture_records.py                 # gen-1
    python scripts/make_fixture_records.py --generation 2  # gen-2
    python scripts/make_fixture_records.py --generation 3 --per-category 2  # gen-3

``--generation 2`` adds the categories the rule detectors are tested on: a
resolution whose effect half kills a creature at toughness 0 or less in the
state-based check (704.5f fires), one that leaves a creature at toughness 1
(704.5f does not fire), a creature killed by damage (704.5g), the legend rule
(704.5j), a planeswalker at loyalty 0 (704.5i) and an Aura whose host left
(704.5m), a damage-step probe stripping deathtouch, triggers a keyword
generates (fired and not), a predefined token
kind made (111.10) and an ability whose source is one, and records from a
random-seat game and from a game the Forge AI played on both seats, and the
shapes the detector families' parameters read: a resolution and a decision
candidate acting through a keyword's own line, a candidate whose card prints a
keyword, a candidate whose cost has a part (``Sac<...>``), a candidate listing
targets, a damage rewrite a prevention shield replaced, a creature leaving the
battlefield in combat, a token's static and a counter-conditioned static.

Gen-2 also pins records by id (``_GEN2_PINNED``): the shapes the rule
detectors were corrected against in T036, each one record read and judged by
hand, found by reading the shards their ids name. ``--append`` adds the pinned
records to the fixture already on disk without rescanning, so every record an
existing test names stays.

``--generation 3`` reads the T079 pilot (``output/effects/records-gen3-pilot``)
and adds, before gen-2's categories, one per story-2 acceptance scenario
(``a1-`` … ``a11-``: a permanent with a static, a trigger and a replacement
line; a creature stripped of every ability; flying granted by another
permanent's static; ward and annihilator fragments; protection; a first-strike
combat; a burn kill, which also brings the dies-trigger records of its game; a
sorcery off turn and a candidate of the seat without priority; an unmet
condition; a declared anchor tapped and one with vigilance), one per
``turn_action`` subkind and per failed check, the anchored legality shapes, and
the new snapshot members (floating effects of both kinds, delayed triggers,
dungeons, day and night, each extra-entity role, a face-down card, dice).
Face-down keys name ``variant-scripts/``, so that tree is a sidecar root too.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from pathlib import Path

import orjson

from effects.domain.script_text import params as params_of
from effects.domain.script_text import root as root_of
from effects.infrastructure.record_codec import record_from_dict
from effects.infrastructure.shard_io import iter_shard_lines, iter_shards

_KEY_FIELDS = {"script_file", "face", "trait_kind", "index_within_kind"}

#: Gen-2 records pinned by id, with what each shows (T036, the rule detectors'
#: review groups). A shard is named by its record id's prefix.
_GEN2_PINNED: dict[str, str] = {
    "71253f20-5472-4ddd-8c58-27c78651878b.3-mv4d43zn1o4f.6392":
        "burn takes a player to 0 life; the game end is in the next record",
    "71253f20-5472-4ddd-8c58-27c78651878b.3-mv4d43zn1o4f.6393":
        "the game-end rescue of orphaned events holding player_won",
    "71253f20-5472-4ddd-8c58-27c78651878b.8-mv3ts9u2nlt0.12970":
        "a game-end rescue spanning a turn, its board read after its events",
    "71253f20-5472-4ddd-8c58-27c78651878b.0-mv46oleh5eqw.8786":
        "a combat dealing lethal damage and poison at once",
    "71253f20-5472-4ddd-8c58-27c78651878b.7-mv4cjj66gz7g.13429":
        "Disintegrate: lethal damage, the creature exiled instead of dying",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv42599vpk5m.16775":
        "Stuffy Doll: lethal damage to an indestructible creature",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv42599vpk5m.1744":
        "Pestilence ends the game in the check: no creature dies",
    "71253f20-5472-4ddd-8c58-27c78651878b.8-mv3ts9u2nlt0.11931":
        "Nyx Infusion: a death with no recorded P/T change",
    "e4713c96-1647-4d93-ac89-039a435f0363.11-mv4w45ojam2q.11464":
        "Bite Down: one damage from a deathtouch source kills",
    "e4713c96-1647-4d93-ac89-039a435f0363.0-mv4yz1t23qrs.5464":
        "Undercity Uprising: deathtouch granted by the resolution, then a fight",
    "e4713c96-1647-4d93-ac89-039a435f0363.11-mv4w45ojam2q.15348":
        "Kami of the Painted Road gains protection from white: its white Aura falls",
    "e4713c96-1647-4d93-ac89-039a435f0363.2-mv541ovjbkr3.2538":
        "a bestowed Aura's host leaves",
    "71253f20-5472-4ddd-8c58-27c78651878b.2-mv465aywnc5h.14696":
        "Ordeal of Thassa sacrifices itself, attributed unresolved",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv4cne1d8rhq.2995":
        "an Aura destroyed as the target, attributed unresolved",
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv4cne1d8rhq.2092":
        "the host destroyed, its Aura exiled in the check",
    "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.11-mv5g0s6bka1j.14623":
        "a lone legend entering at 0/0 dies",
    "71253f20-5472-4ddd-8c58-27c78651878b.10-mv4aecvvg3eo.3650":
        "a dies trigger whose board shows its dead host at toughness 0",
    "e4713c96-1647-4d93-ac89-039a435f0363.7-mv4yq1tr68fr.2370":
        "an Aura's toughness boost lost: marked damage now lethal",
    "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.11-mv5epq5qc9mg.3703":
        "the host stops being a creature: its Aura falls",
    "71253f20-5472-4ddd-8c58-27c78651878b.6-mv461fz6wqim.13501":
        "an endure resolution writing its counters as P1P1",
    "2ed74263-ee28-4948-9621-55dba1160c44.1-mv5im7i940u2.36687":
        "a Saga's last chapter draws, then the Saga is sacrificed in the check",
}

#: Records that follow a fired trigger in its game, kept so an interaction join
#: (trigger -> the resolution it put on the stack) has its second half.
_FOLLOWERS_PER_TRIGGER = 6
#: Trigger records kept per game to find the ones a chosen kill fired.
_RECENT_TRIGGERS = 12


def _keys_in(obj) -> Iterator[dict]:
    """Every provenance-key dict anywhere below ``obj``."""
    if isinstance(obj, dict):
        if _KEY_FIELDS <= obj.keys():
            yield obj
            return
        for value in obj.values():
            yield from _keys_in(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _keys_in(value)


class _Sidecars:
    """Raw sidecar JSON by script file, read on demand from the source roots."""

    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots
        self._cache: dict[str, dict | None] = {}

    def path(self, script_file: str) -> Path | None:
        tree, _, relative = script_file.partition("/")
        root = self.roots.get(tree)
        if root is None:
            return None
        return (root / relative).with_suffix(".provenance.json")

    def get(self, script_file: str) -> dict | None:
        if script_file not in self._cache:
            path = self.path(script_file)
            self._cache[script_file] = (
                json.loads(path.read_text(encoding="utf-8"))
                if path is not None and path.exists() else None
            )
        return self._cache[script_file]

    def line(self, key: dict) -> dict | None:
        """The raw sidecar line a key names, or None."""
        sidecar = self.get(key["script_file"])
        if sidecar is None:
            return None
        for line in sidecar["lines"]:
            for candidate in line["provenance"]:
                if (candidate["face"], candidate["trait_kind"],
                        candidate["index_within_kind"]) == (
                        key["face"], key["trait_kind"], key["index_within_kind"]):
                    return line
        return None

    def api_type(self, key: dict) -> str | None:
        line = self.line(key)
        return line.get("script_api_type") if line is not None else None

    def script(self, key: dict) -> str:
        line = self.line(key)
        return (line.get("script_text") or "") if line is not None else ""

    def prints_keyword(self, script_file: str) -> bool:
        """Whether the card has a line a keyword produced."""
        sidecar = self.get(script_file)
        return sidecar is not None and any(
            line.get("script_api_type") == "Keyword" for line in sidecar["lines"])


def _toughness(entity: dict) -> int | None:
    """Net toughness: gen-2's ``pt`` triple, or gen-3's ``base_pt`` and ``pt_boosts``."""
    if "base_pt" in entity:
        base, boosts = entity["base_pt"], entity.get("pt_boosts")
        if base is None or boosts is None:
            return None
        return base[1] + boosts["temporary"][1] + boosts["counters"][1]
    pt = entity.get("pt")
    if not pt:
        return None
    return sum(part[1] for part in (pt["base"], pt["boosts"], pt["counters"]))


def _toughness_after(record: dict) -> dict[str, tuple[int, bool, bool]]:
    """``entity id -> (toughness after the record's own P/T changes, damaged,
    died in the state-based check)`` for each creature the record names."""
    events = record["payload"].get("events") or []
    entities = {e["id"]: e for e in record["state"]["entities"]}
    delta: dict[str, int] = {}
    damaged: set[str] = set()
    died: set[str] = set()
    uncreatured: set[str] = set()
    for event in events:
        params = event.get("params") or {}
        for subject in event.get("subjects") or ():
            if event["type"] == "type_change" and "creature" in (params.get("types_removed") or ()):
                # No longer a creature, so a death is not 704.5f's (an Aura
                # with nothing to enchant is 704.5m).
                uncreatured.add(subject)
            elif event["type"] == "pt_change":
                delta[subject] = delta.get(subject, 0) + int(params.get("toughness_delta", 0))
            elif event["type"] == "damage_dealt":
                damaged.add(subject)
            elif (event["type"] == "zone_change" and event.get("attributed_to") == "unresolved"
                  and params.get("from_zone") == "battlefield"
                  and params.get("to_zone") == "graveyard"):
                died.add(subject)
    out = {}
    for entity_id in set(delta) | died:
        entity = entities.get(entity_id)
        if (entity is None or entity_id in uncreatured
                or "creature" not in (entity.get("types") or ())):
            continue
        toughness = _toughness(entity)
        if toughness is not None:
            out[entity_id] = (toughness + delta.get(entity_id, 0),
                              entity_id in damaged, entity_id in died)
    return out


def _effect_resolution(record: dict) -> bool:
    return (record["kind"] == "resolution" and record.get("moment") == "resolution"
            and not record["fork"])


def _toughness_zero_death(record: dict) -> bool:
    """704.5f fires: a creature dies in the check at toughness 0 or less, undamaged."""
    return _effect_resolution(record) and any(
        toughness <= 0 and died and not damaged
        for toughness, damaged, died in _toughness_after(record).values())


def _left_at_toughness_one(record: dict) -> bool:
    """704.5f does not fire: a creature's toughness is lowered to 1 and it stays."""
    if not _effect_resolution(record):
        return False
    lowered = {
        subject for event in record["payload"].get("events") or ()
        if event["type"] == "pt_change"
        and int((event.get("params") or {}).get("toughness_delta", 0)) < 0
        for subject in event.get("subjects") or ()}
    return any(entity_id in lowered and toughness == 1 and not died
               for entity_id, (toughness, _, died) in _toughness_after(record).items())


def _lethal_damage_death(record: dict) -> bool:
    """704.5g fires: a creature dealt damage by the resolution dies in the check."""
    return _effect_resolution(record) and any(
        damaged and died and toughness > 0
        for toughness, damaged, died in _toughness_after(record).values())


def _sba_fires(check: str) -> Callable[[dict], bool]:
    """A death the ``sba_zone_move`` detector puts on the rule's fires side and
    sees happen: the detector itself picks the record, so the fixture holds
    what the family test reads. Only effect halves with an unresolved
    battlefield death are parsed, which keeps the scan cheap."""
    from effects.domain.rule_catalogue import Polarity
    from effects.domain.rule_detectors.sba_zone_move import FAMILY

    def test(record: dict) -> bool:
        if not _effect_resolution(record) or not any(
                e["type"] == "zone_change" and e.get("attributed_to") == "unresolved"
                for e in record["payload"].get("events") or ()):
            return False
        found = FAMILY.match(record_from_dict(record), None, {"check": check})
        return found is not None and found.polarity is Polarity.FIRES and found.observed
    return test


#: Rule 111.10's predefined token kinds the fixture looks for.
_PREDEFINED_TOKENS = frozenset({"treasure", "food", "clue", "blood", "map", "gold",
                                "powerstone", "shard", "incubator", "junk"})


def _keyword_trigger(fired: bool) -> Callable[[dict], bool]:
    """A trigger a keyword generates: its acting key is the keyword's trait."""
    def test(record: dict) -> bool:
        keys = record.get("ability") or ()
        return (record["kind"] == "trigger" and bool(record["payload"].get("fired")) == fired
                and bool(keys) and keys[0]["trait_kind"] == "keyword")
    return test


def _subtypes(type_line: str) -> set[str]:
    _, dash, rest = type_line.partition(" - ")
    return {word.lower() for word in rest.split()} if dash else set()


def _predefined_token_made(record: dict) -> bool:
    return _effect_resolution(record) and any(
        e["type"] == "token_created" and _subtypes(str(
            ((e.get("params") or {}).get("characteristics") or {}).get("types") or ""))
        & _PREDEFINED_TOKENS
        for e in record["payload"].get("events") or ())


def _predefined_token_source(record: dict) -> bool:
    if not _effect_resolution(record):
        return False
    source = (record["state"].get("refs") or {}).get("source")
    return any(e["id"] == source and e.get("token_script_id")
               and {s.lower() for s in e.get("subtypes") or ()} & _PREDEFINED_TOKENS
               for e in record["state"]["entities"])


def _candidate_key(record: dict) -> dict | None:
    """A decision record's one candidate's first key."""
    if record["kind"] != "playability" or record.get("subkind") != "decision":
        return None
    candidates = record["payload"].get("candidates") or ()
    keys = (candidates[0].get("ability") or ()) if candidates else ()
    return keys[0] if keys else None


def _combatant_leaves(record: dict) -> bool:
    """An effect half moving a creature that is attacking or blocking off the battlefield."""
    if not _effect_resolution(record):
        return False
    entities = {e["id"]: e for e in record["state"]["entities"]}
    for event in record["payload"].get("events") or ():
        if event["type"] != "zone_change" or (
                (event.get("params") or {}).get("from_zone") != "battlefield"):
            continue
        for subject in event.get("subjects") or ():
            combat = (entities.get(subject) or {}).get("combat") or {}
            if combat.get("attacking") or combat.get("blocking"):
                return True
    return False


def _detector_shapes(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Records the detector families' parameters read: a keyword's own line
    acting, a candidate's card printing a keyword, a cost part, targets, a
    prevention shield, a combatant leaving, a token's or a counter-conditioned
    static."""
    from effects.domain.rule_detectors.continuous_contribution import (
        condition_counter_types,
    )
    from effects.domain.rule_detectors.rewrite_event import PREVENTS

    def first_key(record: dict) -> dict | None:
        keys = record.get("ability") or ()
        return keys[0] if keys else None

    def keyword_resolution(record: dict) -> bool:
        key = first_key(record)
        return (_effect_resolution(record) and key is not None
                and key["trait_kind"] == "keyword"
                and bool(record["payload"].get("events")))

    def keyword_decision(record: dict) -> bool:
        key = _candidate_key(record)
        return key is not None and key["trait_kind"] == "keyword"

    def keyword_card_decision(record: dict) -> bool:
        key = _candidate_key(record)
        return (key is not None and key["trait_kind"] == "spell"
                and sidecars.prints_keyword(key["script_file"]))

    def cost_part_decision(record: dict) -> bool:
        key = _candidate_key(record)
        return key is not None and "<" in (
            params_of(root_of(sidecars.script(key))).get("Cost") or "")

    def targeting_decision(record: dict) -> bool:
        if _candidate_key(record) is None:
            return False
        targets = record["payload"]["candidates"][0].get("legal_targets") or ()
        return 0 < len(targets) < 32

    def prevention_rewrite(record: dict) -> bool:
        key = first_key(record)
        return (record["kind"] == "rewrite" and key is not None
                and record["payload"]["incoming"]["type"] == "damage_dealt"
                and record["payload"].get("result") in ("replaced", "updated")
                and bool(PREVENTS.search(sidecars.script(key))))

    def token_static(record: dict) -> bool:
        key = first_key(record)
        return (record["kind"] == "continuous" and key is not None
                and key["script_file"].startswith("tokenscripts/"))

    def counter_conditioned_static(record: dict) -> bool:
        key = first_key(record)
        return (record["kind"] == "continuous" and key is not None
                and bool(condition_counter_types(sidecars.script(key))))

    return {
        "keyword-resolution": keyword_resolution,
        "keyword-decision": keyword_decision,
        "keyword-card-decision": keyword_card_decision,
        "cost-part-decision": cost_part_decision,
        "targeting-decision": targeting_decision,
        "prevention-rewrite": prevention_rewrite,
        "combatant-leaves": _combatant_leaves,
        "token-static": token_static,
        "counter-conditioned-static": counter_conditioned_static,
    }


def _gen2_categories(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Gen-2's own categories, matched before the shared ones."""
    return {
        **_detector_shapes(sidecars),
        "704.5j-fires": _sba_fires("legend_rule"),
        "704.5m-fires": _sba_fires("aura_illegal"),
        "704.5i-fires": _sba_fires("loyalty_zero"),
        "704.5f-fires": _toughness_zero_death,
        "704.5g-fires": _lethal_damage_death,
        "704.5f-does-not-fire": _left_at_toughness_one,
        "keyword-trigger-fired": _keyword_trigger(True),
        "keyword-trigger-unfired": _keyword_trigger(False),
        "predefined-token-source": _predefined_token_source,
        "predefined-token-made": _predefined_token_made,
        "probe-deathtouch": lambda r: r["kind"] == "combat" and (
            r["payload"].get("probed_keyword") == "deathtouch"),
        "random-seat": lambda r: r.get("random_seat") is True,
    }


#: Gen-3 ``turn_action`` subkinds and decision ``failed_checks`` identifiers,
#: one fixture category each (contracts/record-schema-delta.md).
_TURN_ACTION_SUBKINDS = ("draw", "cleanup_discard", "untap", "phasing", "day_night",
                         "lore_counter")
_FAILED_CHECKS = ("split_second", "casting_speed", "whose_turn", "phase", "combat",
                  "activation_limit", "script_condition", "zone", "static_prohibition",
                  "cost", "script_restriction")
_EXTRA_ROLES = ("library_card", "exiled_with", "mutate_component", "last_known")


def _lines(entity: dict) -> list[dict]:
    return entity.get("lines") or []


def _fragments(entity: dict) -> list[str]:
    """The keyword lines' fragments, lower-cased (``ward:2``, ``flying``)."""
    return [str(line.get("fragment") or "").lower() for line in _lines(entity)
            if line["kind"] == "keyword"]


def _has_keyword(entity: dict, *names: str) -> bool:
    return any(fragment.split(":", 1)[0] in names for fragment in _fragments(entity))


def _on_battlefield(entity: dict) -> bool:
    return entity.get("zone") == "battlefield" and entity.get("role") is None


def _is_creature(entity: dict) -> bool:
    return "creature" in (entity.get("types") or ())


def _deaths(events, *, unresolved: bool = True) -> set[str]:
    """Subjects of battlefield-to-graveyard moves (no clause's doing, by default)."""
    return {subject for event in events or ()
            if event["type"] == "zone_change"
            and (event.get("params") or {}).get("from_zone") == "battlefield"
            and (event.get("params") or {}).get("to_zone") == "graveyard"
            and (not unresolved or event.get("attributed_to") == "unresolved")
            for subject in event.get("subjects") or ()}


def _acting_kind(record: dict) -> str | None:
    keys = record.get("ability") or ()
    return keys[0]["trait_kind"] if keys else None


def _burn_kill(record: dict) -> set[str]:
    """Story 2 acceptance 7's first half: a spell's effect half dealing damage
    to a creature that then dies in the state-based check; the dead subjects."""
    if not _effect_resolution(record) or _acting_kind(record) != "spell":
        return set()
    events = record["payload"].get("events") or ()
    damaged = {subject for event in events if event["type"] == "damage_dealt"
               for subject in event.get("subjects") or ()}
    return damaged & _deaths(events)


def trigger_held_check_deaths(record: dict) -> set[str]:
    """SC-004's violation: a triggered ability's effect half holding a death of
    the check its trigger reached the stack during. Such a death happened before
    the half's snapshot, so the snapshot no longer shows its subject on the
    battlefield; a death the trigger's own resolution causes still does."""
    if not _effect_resolution(record) or _acting_kind(record) != "trigger":
        return set()
    zones = {e["id"]: e.get("zone") for e in record["state"]["entities"]
             if e.get("role") is None}
    return {subject for subject in _deaths(record["payload"].get("events"))
            if zones.get(subject, "battlefield") != "battlefield"}


def _candidate(record: dict) -> dict | None:
    if record["kind"] != "playability" or record.get("subkind") != "decision":
        return None
    candidates = record["payload"].get("candidates") or ()
    return candidates[0] if candidates else None


def _candidate_card(record: dict, candidate: dict) -> dict | None:
    """The entity whose lines carry the candidate's key."""
    keys = candidate.get("ability") or ()
    if not keys:
        return None
    for entity in record["state"]["entities"]:
        if any(keys[0] in line["keys"] for line in _lines(entity)):
            return entity
    return None


def _failed(candidate: dict) -> list[str]:
    return [entry["check"] for entry in candidate.get("failed_checks") or ()]


def sorcery_off_turn(record: dict) -> bool:
    """Story 2 acceptance 8: a sorcery in hand at a priority stop of the
    opponent's turn reads timing false, can_play false, casting_speed failed."""
    candidate = _candidate(record)
    if candidate is None or record["state"]["global"].get("active") == record["actor_player"]:
        return False
    card = _candidate_card(record, candidate)
    verdict = candidate.get("verdict") or {}
    return (card is not None and card.get("zone") == "hand"
            and "sorcery" in (card.get("types") or ())
            and verdict.get("timing") is False and verdict.get("can_play") is False
            and "casting_speed" in _failed(candidate))


def other_seat_candidate(record: dict) -> bool:
    """Story 2 acceptance 8: a candidate of the player without priority."""
    return (_candidate(record) is not None
            and record["state"]["global"].get("priority") not in (None, record["actor_player"]))


def lost_all_abilities(record: dict, sidecars: _Sidecars) -> bool:
    """Story 2 acceptance 2: a battlefield creature some static or floating
    effect strips of every ability (``RemoveAllAbilities$``) lists no line."""
    state = record["state"]
    stripped: set[str] = set()
    for entity in state["entities"]:
        for line in _lines(entity):
            if (line["kind"] == "static" and line.get("active") and line["keys"]
                    and "RemoveAllAbilities" in sidecars.script(line["keys"][0])):
                stripped.update(line.get("affected") or ())
    for effect in state["global"].get("floating_effects") or ():
        if any("RemoveAllAbilities" in sidecars.script(key)
               for key in effect.get("source_lines") or ()):
            stripped.update(effect.get("affected") or ())
    return any(entity["id"] in stripped and _on_battlefield(entity) and _is_creature(entity)
               and not _lines(entity) for entity in state["entities"])


def _own_files(entity: dict) -> set[str]:
    return {key["script_file"] for line in _lines(entity) if not line["granted"]
            for key in line["keys"]}


def granted_by_other_static(record: dict, keyword: str = "flying") -> bool:
    """Story 2 acceptance 3: a permanent carries a keyword line granted by
    another permanent's static, keyed to that static."""
    for entity in record["state"]["entities"]:
        if not _on_battlefield(entity):
            continue
        own = _own_files(entity)
        for line in _lines(entity):
            if (line["kind"] == "keyword" and line["granted"]
                    and str(line.get("fragment") or "").lower() == keyword
                    and any(key["trait_kind"] == "static" and key["script_file"] not in own
                            for key in line["keys"])):
                return True
    return False


def static_trigger_replacement(record: dict) -> bool:
    """Story 2 acceptance 1: a permanent listing a static (flagged ``active``),
    a trigger and a replacement line, each keyed."""
    for entity in record["state"]["entities"]:
        if not _on_battlefield(entity):
            continue
        kinds = {line["kind"] for line in _lines(entity) if line["keys"]}
        if {"static", "trigger", "replacement"} <= kinds and all(
                "active" in line for line in _lines(entity) if line["kind"] == "static"):
            return True
    return False


def _fragment_prefix(prefix: str) -> Callable[[dict], bool]:
    return lambda record: any(
        fragment.startswith(prefix) for entity in record["state"]["entities"]
        if _on_battlefield(entity) for fragment in _fragments(entity))


def first_strike_combat(record: dict) -> bool:
    """Story 2 acceptance 6: an observed combat with a first or double striker,
    its shape and assignment filled."""
    if record["kind"] != "combat" or record["fork"]:
        return False
    payload = record["payload"]
    combatants = set(payload.get("attackers") or ()) | {
        blocker for blockers in (payload.get("blocks") or {}).values() for blocker in blockers}
    return bool(payload.get("assignment_choices")) and any(
        entity["id"] in combatants and _has_keyword(entity, "first strike", "double strike")
        for entity in record["state"]["entities"])


def _anchor(record: dict) -> dict | None:
    if record["kind"] != "playability" or record.get("subkind") != "attackers":
        return None
    anchor = record["payload"].get("anchor_attacker")
    return next((e for e in record["state"]["entities"] if e["id"] == anchor), None)


def _attackers(test: Callable[[dict, dict], bool]) -> Callable[[dict], bool]:
    def check(record: dict) -> bool:
        anchor = _anchor(record)
        return anchor is not None and test(record["payload"], anchor)
    return check


def _floating(prefix: str) -> Callable[[dict], bool]:
    return lambda record: any(
        str(effect["id"]).startswith(prefix)
        for effect in record["state"]["global"].get("floating_effects") or ())


def _empty_mana(record: dict) -> bool:
    return any(event["type"] == "mana_produced" and not any(
        amount > 0 for amount in ((event.get("params") or {}).get("mana_by_color") or {}).values())
        for event in (record["payload"].get("events") or ()))


def _gen3_categories(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Gen-3's own categories: story 2's acceptance scenarios, each
    ``turn_action`` subkind and failed check, the anchored legality records and
    the new snapshot members. Matched before gen-2's and the shared ones."""
    def turn_action(subkind: str) -> Callable[[dict], bool]:
        def test(record: dict) -> bool:
            if record["kind"] != "turn_action" or record.get("subkind") != subkind:
                return False
            return subkind != "draw" or any(
                e["type"] == "card_drawn" for e in record["payload"].get("events") or ())
        return test

    def failed_check(check: str) -> Callable[[dict], bool]:
        def test(record: dict) -> bool:
            candidate = _candidate(record)
            return candidate is not None and check in _failed(candidate)
        return test

    def role(name: str) -> Callable[[dict], bool]:
        return lambda record: any(e.get("role") == name for e in record["state"]["entities"])

    return {
        "a1-static-trigger-replacement": static_trigger_replacement,
        "a2-lost-all-abilities": lambda r: lost_all_abilities(r, sidecars),
        "a3-granted-flying": granted_by_other_static,
        "a4-ward": _fragment_prefix("ward:"),
        "a4-annihilator": _fragment_prefix("annihilator:"),
        "a5-protection-from-goblins": lambda r: any(
            f.startswith("protection") and "goblin" in f for e in r["state"]["entities"]
            if _on_battlefield(e) for f in _fragments(e)),
        "a5-protection": _fragment_prefix("protection"),
        "a6-first-strike-combat": first_strike_combat,
        "a7-burn-kill": lambda r: bool(_burn_kill(r)),
        "a8-sorcery-off-turn": sorcery_off_turn,
        "a8-other-seat": other_seat_candidate,
        "a9-condition-unmet": lambda r: _effect_resolution(r) and any(
            entry.get("met") is False for entry in r["payload"].get("conditions_met") or ()),
        **{f"turn-action-{subkind}": turn_action(subkind) for subkind in _TURN_ACTION_SUBKINDS},
        "a11-attacker-tapped": _attackers(
            lambda p, anchor: p.get("tapped_by_attack") is True),
        "a11-attacker-vigilance": _attackers(
            lambda p, anchor: p.get("tapped_by_attack") is False
            and _has_keyword(anchor, "vigilance")),
        "attacker-must-attack": _attackers(lambda p, anchor: p.get("must_attack") is True),
        "attacker-no-defender": _attackers(lambda p, anchor: p.get("legal_defenders") == []),
        "blockers-must-block": lambda r: r["kind"] == "playability" and r.get(
            "subkind") == "blockers" and bool(r["payload"].get("must_block")),
        **{f"failed-check-{check}": failed_check(check) for check in _FAILED_CHECKS},
        "floating-effect-card": _floating("E"),
        "floating-effect-timestamp": _floating("F"),
        "delayed-trigger": lambda r: bool(r["state"]["global"].get("delayed_triggers")),
        "dungeon": lambda r: bool(r["state"]["global"].get("dungeons")),
        "day-night": lambda r: r["state"]["global"].get("day_night") is not None,
        **{f"role-{name}": role(name) for name in _EXTRA_ROLES},
        "face-down": lambda r: any(e.get("face_state") == "face_down"
                                   for e in r["state"]["entities"]),
        "dice-rolled": lambda r: bool(((r["state"].get("refs") or {}).get("random") or {})
                                      .get("dice")),
        "mana-empty-colors": _empty_mana,
        # The rarer § 8.13 members, so each varies within the fixture too.
        "player-speed": lambda r: any(p.get("speed") for p in r["state"]["players"]),
        "city-blessing": lambda r: any(p.get("city_blessing") for p in r["state"]["players"]),
        "monarch": lambda r: any(p.get("monarch") for p in r["state"]["players"]),
        "x-paid": lambda r: any(e.get("x_paid") for e in r["state"]["entities"]),
        "designations": lambda r: any(e.get("designations") for e in r["state"]["entities"]),
        "printed-loyalty": lambda r: any(e.get("printed_loyalty") is not None
                                         for e in r["state"]["entities"]),
        "chosen-color": lambda r: any((e.get("chosen") or {}).get("colors")
                                      for e in r["state"]["entities"]),
        "keyword-granted-by-effect": lambda r: any(
            line["kind"] == "keyword" and line["granted"] and not line["keys"]
            for e in r["state"]["entities"] for line in _lines(e)),
    }


def _categories(sidecars: _Sidecars) -> dict[str, Callable[[dict], bool]]:
    def acting_api(record: dict) -> str | None:
        keys = record.get("ability") or ()
        return sidecars.api_type(keys[0]) if keys else None

    def is_kind(kind: str, discriminator: str | None = None):
        def test(record: dict) -> bool:
            if record["kind"] != kind:
                return False
            return discriminator is None or (
                record.get("moment") or record.get("subkind")) == discriminator
        return test

    observed_effect = is_kind("resolution", "resolution")
    # Most specific first: a record lands in the first category it fits.
    return {
        "modal": lambda r: is_kind("resolution")(r) and acting_api(r) == "Charm",
        "interventional": lambda r: observed_effect(r) and r["interventional"],
        "combat-probe": lambda r: r["kind"] == "combat" and bool(
            r["payload"].get("probed_keyword")),
        "blockers-forbidden": lambda r: is_kind("playability", "blockers")(r) and bool(
            r["payload"].get("forbidden")),
        "activation": is_kind("resolution", "activation"),
        "effect": lambda r: observed_effect(r) and not r["fork"] and bool(r.get("link_id")),
        "trigger-fired": lambda r: r["kind"] == "trigger" and r["payload"].get("fired"),
        "trigger-unfired": lambda r: r["kind"] == "trigger" and not r["payload"].get("fired"),
        "continuous": is_kind("continuous"),
        "rewrite": is_kind("rewrite"),
        "combat": lambda r: r["kind"] == "combat" and not r["fork"],
        "decision": is_kind("playability", "decision"),
        "attackers": is_kind("playability", "attackers"),
        "blockers": is_kind("playability", "blockers"),
    }


def _pinned_records(records_dir: Path, wanted: dict[str, str]) -> dict[str, dict]:
    """The pinned records, read from the shards their ids name."""
    by_shard: dict[str, set[str]] = defaultdict(set)
    for record_id in wanted:
        by_shard[record_id.rsplit(".", 1)[0]].add(record_id)
    paths = {path.name.split(".jsonl")[0]: path for path in iter_shards(records_dir)}
    found: dict[str, dict] = {}
    for name, ids in sorted(by_shard.items()):
        if name not in paths:
            continue
        for line in iter_shard_lines(paths[name]):
            record = json.loads(line)
            if record["record_id"] in ids:
                found[record["record_id"]] = record
    return found


def _read_fixture(path: Path) -> dict[str, dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return {r["record_id"]: r for r in (json.loads(line) for line in handle if line.strip())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--generation", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--records-dir", type=Path,
                        help="default output/effects/records, records-gen2 for --generation 2, "
                             "records-gen3-pilot for --generation 3")
    parser.add_argument("--cards-folder", action="append", type=Path,
                        help="converted roots, named by tree (repeatable); "
                             "default output/cardsfolder and output/tokenscripts, "
                             "and output/effects/variant-scripts for --generation 3")
    parser.add_argument("--output", type=Path,
                        help="default tests/fixtures/effects/gen<N>-records.jsonl.gz")
    parser.add_argument("--sidecars-out", type=Path,
                        help="default tests/fixtures/effects/gen<N>-sidecars")
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--max-shards", type=int, default=400)
    parser.add_argument("--append", action="store_true",
                        help="gen-2: add the pinned records to the fixture on disk, no scan")
    args = parser.parse_args()
    gen = args.generation
    args.records_dir = args.records_dir or Path(
        {1: "output/effects/records", 2: "output/effects/records-gen2",
         3: "output/effects/records-gen3-pilot"}[gen])
    args.output = args.output or Path(f"tests/fixtures/effects/gen{gen}-records.jsonl.gz")
    args.sidecars_out = args.sidecars_out or Path(f"tests/fixtures/effects/gen{gen}-sidecars")

    folders = args.cards_folder or [
        Path("output/cardsfolder"), Path("output/tokenscripts"),
        # A face-down card's keys name its variant script (T071).
        *([Path("output/effects/variant-scripts")] if gen == 3 else [])]
    sidecars = _Sidecars({folder.name: folder for folder in folders})
    categories = _categories(sidecars)
    if gen >= 2:
        categories = {**_gen2_categories(sidecars), **categories}
    if gen == 3:
        categories = {**_gen3_categories(sidecars), **categories}
    chosen: dict[str, list[dict]] = defaultdict(list)
    wanted_links: set[str] = set()
    wanted_mirrors: set[str] = set()
    extras: dict[str, dict] = {}

    if args.append:
        if gen != 2:
            parser.error("--append adds gen-2's pinned records")
        return _append_pinned(args, sidecars)

    shards = iter_shards(args.records_dir)
    # Interleave the collection runs rather than reading one run's shards first,
    # so a category only another run reaches (probes, variants) is found early.
    shards = shards[::7] + [s for i, s in enumerate(shards) if i % 7]
    forge_ai_wanted = gen >= 2
    for shard_no, shard in enumerate(shards[: args.max_shards]):
        followers: dict[str, int] = {}
        # The trigger records just written in each game: a dies trigger's record
        # precedes the held effect half of the kill that fired it (acceptance 7).
        recent_triggers: dict[str, deque] = defaultdict(lambda: deque(maxlen=_RECENT_TRIGGERS))
        seat_games: set[str] = set()
        plain: dict[str, list[dict]] = defaultdict(list)
        for line in iter_shard_lines(shard):
            try:
                # orjson, as the codec reads: a gen-3 shard is tens of
                # gigabytes of JSON.
                record = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            record_id = record["record_id"]
            game = record["game_id"]
            if record.get("random_seat"):
                seat_games.add(game)
            elif forge_ai_wanted and not record["fork"] and len(plain[game]) < 2:
                plain[game].append(record)
            if followers.get(game) and record["kind"] == "resolution":
                extras[record_id] = record
                followers[game] -= 1
            if record.get("link_id") in wanted_links and record_id not in extras:
                extras[record_id] = record
            if record_id in wanted_mirrors:
                extras[record_id] = record
            if record["kind"] == "trigger":
                recent_triggers[game].append(record)
            for name, test in categories.items():
                if len(chosen[name]) >= args.per_category or not test(record):
                    continue
                chosen[name].append(record)
                if record.get("link_id"):
                    wanted_links.add(record["link_id"])
                if record.get("mirror_of"):
                    wanted_mirrors.add(record["mirror_of"])
                if name in ("trigger-fired", "a7-burn-kill"):
                    followers[game] = _FOLLOWERS_PER_TRIGGER
                if name == "a7-burn-kill":
                    dead = _burn_kill(record)
                    for trigger in recent_triggers[game]:
                        event = trigger["payload"].get("event") or {}
                        if set(event.get("subjects") or ()) & dead:
                            extras[trigger["record_id"]] = trigger
                break
        if forge_ai_wanted:
            # Games never span shards, so a game is known to be Forge AI on
            # both seats once its shard has been read through.
            for game, kept in plain.items():
                if game not in seat_games and len(chosen["forge-ai"]) < args.per_category:
                    chosen["forge-ai"].append(kept[0])
        if all(len(chosen[name]) >= args.per_category for name in categories) and (
                not forge_ai_wanted or len(chosen["forge-ai"]) >= args.per_category):
            break
        if shard_no % 25 == 0:
            filled = sum(len(chosen[name]) >= args.per_category for name in categories)
            print(f"{shard_no} shards read, {filled}/{len(categories)} categories filled",
                  file=sys.stderr)

    records: dict[str, dict] = {}
    for name in chosen:
        for record in chosen[name]:
            records[record["record_id"]] = record
    records.update(extras)
    if gen == 2:
        records.update(_pinned_records(args.records_dir, _GEN2_PINNED))
    missing = [name for name in [*categories, *(["forge-ai"] if forge_ai_wanted else [])]
               if not chosen[name]]

    # Every record must parse with the current reader before it is a fixture.
    for record in records.values():
        record_from_dict(record)

    _write_fixture(args.output, records)
    copied = _copy_sidecars(records, sidecars, args.sidecars_out, keep_existing=False)

    for name in chosen:
        print(f"{name:22s} {len(chosen[name])}", file=sys.stderr)
    print(f"{len(records)} records ({len(extras)} joined) -> {args.output}; "
          f"{copied} sidecars -> {args.sidecars_out}", file=sys.stderr)
    if missing:
        print(f"no record found for: {', '.join(missing)}", file=sys.stderr)
    return 0


def _write_fixture(output: Path, records: dict[str, dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output, "wt", encoding="utf-8") as out:
        for record in sorted(records.values(), key=lambda r: (r["game_id"], r["timestamp"])):
            out.write(json.dumps(record, separators=(",", ":")) + "\n")


def _copy_sidecars(records: dict[str, dict], sidecars: _Sidecars, sidecars_out: Path,
                   *, keep_existing: bool) -> int:
    """Copy every sidecar and converted text the records name; with
    ``keep_existing``, a file already in the fixture is left as it is."""
    copied = 0
    for script_file in sorted({key["script_file"]
                               for record in records.values()
                               for key in _keys_in(record)}):
        source = sidecars.path(script_file)
        if source is None or not source.exists():
            continue
        tree, _, relative = script_file.partition("/")
        target = (sidecars_out / tree / relative).with_suffix(".provenance.json")
        if keep_existing and target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        text = source.with_name(source.name[: -len(".provenance.json")] + ".txt")
        if text.exists():
            shutil.copyfile(text, target.with_name(text.name))
        copied += 1
    return copied


def _append_pinned(args, sidecars: _Sidecars) -> int:
    """Add the pinned records to the fixture on disk, leaving every record in it."""
    records = _read_fixture(args.output)
    pinned = _pinned_records(args.records_dir, _GEN2_PINNED)
    for record in pinned.values():
        record_from_dict(record)
    added = {rid: r for rid, r in pinned.items() if rid not in records}
    records.update(added)
    _write_fixture(args.output, records)
    copied = _copy_sidecars(added, sidecars, args.sidecars_out, keep_existing=True)
    missing = sorted(set(_GEN2_PINNED) - set(pinned))
    print(f"{len(added)} pinned records added ({len(records)} in all) -> {args.output}; "
          f"{copied} sidecars -> {args.sidecars_out}", file=sys.stderr)
    if missing:
        print(f"pinned records not found: {', '.join(missing)}", file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
