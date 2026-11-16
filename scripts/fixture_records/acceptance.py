"""Records for gen-3's story-2 acceptance scenarios (``a1-`` … ``a11-``)."""

from __future__ import annotations

from collections.abc import Callable

from fixture_records.shapes import (
    acting_kind,
    battlefield,
    candidate,
    deaths,
    entities,
    events,
    failed_checks,
    fragments,
    global_state,
    has_keyword,
    is_creature,
    is_effect_resolution,
    lines,
    on_battlefield,
    subjects,
)
from fixture_records.sidecars import Sidecars

_REMOVE_ALL = "RemoveAllAbilities"


def static_trigger_replacement(record: dict) -> bool:
    """Acceptance 1: a permanent listing a static (flagged ``active``), a
    trigger and a replacement line, each keyed."""
    return any(_lists_static_trigger_replacement(entity) for entity in battlefield(record))


def _lists_static_trigger_replacement(entity: dict) -> bool:
    kinds = {line["kind"] for line in lines(entity) if line["keys"]}
    return {"static", "trigger", "replacement"} <= kinds and all(
        "active" in line for line in lines(entity) if line["kind"] == "static")


def _strips_all(line: dict, sidecars: Sidecars) -> bool:
    return (line["kind"] == "static" and bool(line.get("active")) and bool(line["keys"])
            and _REMOVE_ALL in sidecars.script(line["keys"][0]))


def _stripped_by_statics(record: dict, sidecars: Sidecars) -> set[str]:
    return {affected for entity in entities(record) for line in lines(entity)
            if _strips_all(line, sidecars) for affected in line.get("affected") or ()}


def _stripped_by_floating(record: dict, sidecars: Sidecars) -> set[str]:
    return {affected for effect in global_state(record).get("floating_effects") or ()
            if any(_REMOVE_ALL in sidecars.script(key) for key in effect.get("source_lines") or ())
            for affected in effect.get("affected") or ()}


def lost_all_abilities(record: dict, sidecars: Sidecars) -> bool:
    """Acceptance 2: a battlefield creature some static or floating effect
    strips of every ability (``RemoveAllAbilities$``) lists no line."""
    stripped = _stripped_by_statics(record, sidecars) | _stripped_by_floating(record, sidecars)
    return any(entity["id"] in stripped and on_battlefield(entity) and is_creature(entity)
               and not lines(entity) for entity in entities(record))


def _own_files(entity: dict) -> set[str]:
    return {key["script_file"] for line in lines(entity) if not line["granted"]
            for key in line["keys"]}


def _granted_by_other(line: dict, keyword: str, own: set[str]) -> bool:
    """A granted keyword line keyed to a static of another permanent."""
    return (line["kind"] == "keyword" and line["granted"]
            and str(line.get("fragment") or "").lower() == keyword
            and any(key["trait_kind"] == "static" and key["script_file"] not in own
                    for key in line["keys"]))


def _carries_granted(entity: dict, keyword: str) -> bool:
    own = _own_files(entity)
    return any(_granted_by_other(line, keyword, own) for line in lines(entity))


def granted_by_other_static(record: dict, keyword: str = "flying") -> bool:
    """Acceptance 3: a permanent carries a keyword line granted by another
    permanent's static, keyed to that static."""
    return any(_carries_granted(entity, keyword) for entity in battlefield(record))


def fragment_prefix(prefix: str) -> Callable[[dict], bool]:
    """Acceptance 4 and 5: a permanent with a keyword fragment like ``ward:``."""
    return lambda record: any(
        fragment.startswith(prefix)
        for entity in battlefield(record) for fragment in fragments(entity))


def protection_from_goblins(record: dict) -> bool:
    return any(fragment.startswith("protection") and "goblin" in fragment
               for entity in battlefield(record) for fragment in fragments(entity))


def _combatants(payload: dict) -> set[str]:
    blocks = (payload.get("blocks") or {}).values()
    return set(payload.get("attackers") or ()) | {
        blocker for blockers in blocks for blocker in blockers}


def first_strike_combat(record: dict) -> bool:
    """Acceptance 6: an observed combat with a first or double striker, its
    shape and assignment filled."""
    if record["kind"] != "combat" or record["fork"]:
        return False
    combatants = _combatants(record["payload"])
    return bool(record["payload"].get("assignment_choices")) and any(
        entity["id"] in combatants and has_keyword(entity, "first strike", "double strike")
        for entity in entities(record))


def burn_kill(record: dict) -> set[str]:
    """Acceptance 7's first half: a spell's effect half dealing damage to a
    creature that then dies in the state-based check; the dead subjects."""
    if not is_effect_resolution(record) or acting_kind(record) != "spell":
        return set()
    damaged = {subject for event in events(record) if event["type"] == "damage_dealt"
               for subject in subjects(event)}
    return damaged & deaths(events(record))


def burn_kills(record: dict) -> bool:
    return bool(burn_kill(record))


def _candidate_card(record: dict, found: dict) -> dict | None:
    """The entity whose lines carry the candidate's key."""
    keys = found.get("ability") or ()
    if not keys:
        return None
    return next((entity for entity in entities(record)
                 if any(keys[0] in line["keys"] for line in lines(entity))), None)


def _is_sorcery_in_hand(card: dict | None) -> bool:
    return (card is not None and card.get("zone") == "hand"
            and "sorcery" in (card.get("types") or ()))


def _fails_casting_speed(found: dict) -> bool:
    verdict = found.get("verdict") or {}
    return (verdict.get("timing") is False and verdict.get("can_play") is False
            and "casting_speed" in failed_checks(found))


def sorcery_off_turn(record: dict) -> bool:
    """Acceptance 8: a sorcery in hand at a priority stop of the opponent's
    turn reads timing false, can_play false, casting_speed failed."""
    found = candidate(record)
    if found is None or global_state(record).get("active") == record["actor_player"]:
        return False
    return _is_sorcery_in_hand(_candidate_card(record, found)) and _fails_casting_speed(found)


def other_seat_candidate(record: dict) -> bool:
    """Acceptance 8: a candidate of the player without priority."""
    return (candidate(record) is not None
            and global_state(record).get("priority") not in (None, record["actor_player"]))


def condition_unmet(record: dict) -> bool:
    """Acceptance 9: an effect half one of whose conditions was not met."""
    return is_effect_resolution(record) and any(
        entry.get("met") is False for entry in record["payload"].get("conditions_met") or ())


def _anchor(record: dict) -> dict | None:
    if record["kind"] != "playability" or record.get("subkind") != "attackers":
        return None
    anchor = record["payload"].get("anchor_attacker")
    return next((entity for entity in entities(record) if entity["id"] == anchor), None)


def anchored_attackers(test: Callable[[dict, dict], bool]) -> Callable[[dict], bool]:
    """Acceptance 11 and the anchored legality shapes: an attackers record
    whose payload and anchor attacker pass ``test``."""
    def check(record: dict) -> bool:
        anchor = _anchor(record)
        return anchor is not None and test(record["payload"], anchor)
    return check


def _tapped_by_attack(payload: dict, anchor: dict) -> bool:
    return payload.get("tapped_by_attack") is True


def _vigilant_untapped(payload: dict, anchor: dict) -> bool:
    return payload.get("tapped_by_attack") is False and has_keyword(anchor, "vigilance")


def acceptance_categories(sidecars: Sidecars) -> dict[str, Callable[[dict], bool]]:
    """Acceptance 1 to 9, in the order they are matched."""
    return {
        "a1-static-trigger-replacement": static_trigger_replacement,
        "a2-lost-all-abilities": lambda record: lost_all_abilities(record, sidecars),
        "a3-granted-flying": granted_by_other_static,
        "a4-ward": fragment_prefix("ward:"),
        "a4-annihilator": fragment_prefix("annihilator:"),
        "a5-protection-from-goblins": protection_from_goblins,
        "a5-protection": fragment_prefix("protection"),
        "a6-first-strike-combat": first_strike_combat,
        "a7-burn-kill": burn_kills,
        "a8-sorcery-off-turn": sorcery_off_turn,
        "a8-other-seat": other_seat_candidate,
        "a9-condition-unmet": condition_unmet,
    }


def attacker_tapped_categories() -> dict[str, Callable[[dict], bool]]:
    """Acceptance 11: a declared anchor tapped, and one with vigilance."""
    return {
        "a11-attacker-tapped": anchored_attackers(_tapped_by_attack),
        "a11-attacker-vigilance": anchored_attackers(_vigilant_untapped),
    }
