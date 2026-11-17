"""Checks over whole games and combats: lost games, rescue records, lethal damage."""

from __future__ import annotations

from collections import Counter

from effects.infrastructure.record_codec import record_from_dict

from .model import Check, Line, TallyCheck, events_of, is_effect_half, is_fork, payload_of

OUTCOME_EVENTS = ("player_lost", "player_won")


class MarkedGames(Check):
    """Games with a game-end marker and no `player_lost` or `player_won` in any live record."""

    def __init__(self) -> None:
        self.marked: set[str] = set()
        self.decided: set[str] = set()

    def feed(self, record: dict) -> None:
        game = record.get("game_id")
        if record.get("kind") == "turn_action" and record.get("subkind") == "game_end":
            self.marked.add(game)
        if not is_fork(record) and any(e.get("type") in OUTCOME_EVENTS for e in events_of(record)):
            self.decided.add(game)

    def lines(self) -> list[Line]:
        undecided = len(self.marked - self.decided)
        return [Line("marked-games-without-player-lost", undecided, len(self.marked))]


def _is_live_rescue(record: dict) -> bool:
    if not is_effect_half(record) or is_fork(record):
        return False
    return payload_of(record).get("outcome") is None


class RescueRecords(TallyCheck):
    """Rescue records (an outcome-less effect half with no mana) holding more than the outcome."""

    NAMES = ("rescue-records-holding-more-than-the-outcome",)

    def feed(self, record: dict) -> None:
        if not _is_live_rescue(record):
            return
        types = [e.get("type") for e in events_of(record)]
        if "mana_produced" not in types:
            extra = any(t not in OUTCOME_EVENTS for t in types)
            self.add("rescue-records-holding-more-than-the-outcome", extra)


def _damage_hits(events: list[dict]) -> list[tuple[str, int, bool]]:
    """(entity, amount, deathtouch) per entity of every `damage_dealt` event."""
    hits = []
    for event in events:
        if event.get("type") != "damage_dealt":
            continue
        params = event.get("params") or {}
        deathtouch = params.get("damage_type") == "deathtouch"
        subjects = event.get("subjects") or ()
        hits += [(s, params.get("amount", 0), deathtouch) for s in subjects if s.startswith("E")]
    return hits


def _left_play(events: list[dict]) -> set[str]:
    return {
        subject
        for event in events
        if event.get("type") in ("destroyed", "sacrificed", "zone_change")
        for subject in event.get("subjects") or ()
    }


def _is_battlefield_creature(entity) -> bool:
    if entity is None:
        return False
    return entity.on_battlefield and "creature" in entity.types


def _is_lethal(entity, amount: int, touched: bool) -> bool:
    if not _is_battlefield_creature(entity):
        return False
    return touched or amount >= entity.remaining_toughness


def lethal_creatures(record: dict) -> set[str]:
    """Battlefield creatures dealt damage at or past their toughness, or touched by deathtouch."""
    hits = _damage_hits(events_of(record))
    damage: Counter = Counter()
    for entity_id, amount, _ in hits:
        damage[entity_id] += amount
    touched = {entity_id for entity_id, _, deathtouch in hits if deathtouch}
    state = record_from_dict(record).state
    lethal = set()
    for entity_id, amount in damage.items():
        entity = state.entity(entity_id)
        if _is_lethal(entity, amount, entity_id in touched):
            lethal.add(entity_id)
    return lethal


def _is_natural_combat(record: dict) -> bool:
    if record.get("kind") != "combat" or record.get("synthetic"):
        return False
    return not is_fork(record)


class LethalCombat(TallyCheck):
    """Lethal creatures of natural combats with no death event, by whether the game ended."""

    GROUPS = ("game-ending-combat", "other-combat")
    NAMES = tuple(f"lethal-creatures-without-death[{group}]" for group in GROUPS)

    def feed(self, record: dict) -> None:
        if not _is_natural_combat(record):
            return
        events = events_of(record)
        lost = any(e.get("type") == "player_lost" for e in events)
        group = "game-ending-combat" if lost else "other-combat"
        died = _left_play(events)
        for creature in lethal_creatures(record):
            self.add(f"lethal-creatures-without-death[{group}]", creature not in died)
