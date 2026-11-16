"""Gen-3 categories for the snapshot members: floating effects of both kinds,
delayed triggers, dungeons, day and night, each extra-entity role, a
face-down card, dice, and the rarer § 8.13 members, so each varies within the
fixture too."""

from __future__ import annotations

from collections.abc import Callable

from fixture_records.shapes import (
    entities,
    events,
    global_state,
    is_effect_resolution,
    lines,
    players,
)

_EXTRA_ROLES = ("library_card", "exiled_with", "mutate_component", "last_known")

Test = Callable[[dict], bool]


def _floating(prefix: str) -> Test:
    return lambda record: any(str(effect["id"]).startswith(prefix)
                              for effect in global_state(record).get("floating_effects") or ())


def _global_member(name: str) -> Test:
    return lambda record: bool(global_state(record).get(name))


def _day_night(record: dict) -> bool:
    return global_state(record).get("day_night") is not None


def _role(name: str) -> Test:
    return lambda record: any(entity.get("role") == name for entity in entities(record))


def _face_down(record: dict) -> bool:
    return any(entity.get("face_state") == "face_down" for entity in entities(record))


def _dice_rolled(record: dict) -> bool:
    return bool(((record["state"].get("refs") or {}).get("random") or {}).get("dice"))


def _produces_no_mana(event: dict) -> bool:
    by_color = (event.get("params") or {}).get("mana_by_color") or {}
    return not any(amount > 0 for amount in by_color.values())


def _empty_mana(record: dict) -> bool:
    return any(event["type"] == "mana_produced" and _produces_no_mana(event)
               for event in events(record))


def _player_member(name: str) -> Test:
    return lambda record: any(player.get(name) for player in players(record))


def _entity_member(name: str) -> Test:
    return lambda record: any(entity.get(name) for entity in entities(record))


def _chosen(name: str) -> Test:
    return lambda record: any((entity.get("chosen") or {}).get(name)
                              for entity in entities(record))


def _printed_loyalty(record: dict) -> bool:
    return any(entity.get("printed_loyalty") is not None for entity in entities(record))


def _keyword_granted_by_effect(record: dict) -> bool:
    return any(line["kind"] == "keyword" and line["granted"] and not line["keys"]
               for entity in entities(record) for line in lines(entity))


def _land_plays_not_one(record: dict) -> bool:
    return any("max_land_plays" in player and player["max_land_plays"] != 1
               for player in players(record))


def _rewrite_buffered_events(record: dict) -> bool:
    return record["kind"] == "rewrite" and bool(record["payload"].get("buffered_events"))


def _forbidden_blocker(member: str) -> Test:
    """A blockers record with a forbidden blocker whose ``member`` is set."""
    def test(record: dict) -> bool:
        return (record["kind"] == "playability" and record.get("subkind") == "blockers"
                and any(entry.get(member) for entry in record["payload"].get("forbidden") or ()))
    return test


def _effect_object_created(record: dict) -> bool:
    return is_effect_resolution(record) and sum(
        event["type"] in ("continuous_effect_created", "delayed_trigger_created")
        for event in events(record)) == 1


def snapshot_member_categories() -> dict[str, Test]:
    """The snapshot members' categories, in the order they are matched."""
    return {
        "floating-effect-card": _floating("E"),
        "floating-effect-timestamp": _floating("F"),
        "delayed-trigger": _global_member("delayed_triggers"),
        "dungeon": _global_member("dungeons"),
        "day-night": _day_night,
        **{f"role-{name}": _role(name) for name in _EXTRA_ROLES},
        "face-down": _face_down,
        "dice-rolled": _dice_rolled,
        "mana-empty-colors": _empty_mana,
        "player-speed": _player_member("speed"),
        "city-blessing": _player_member("city_blessing"),
        "monarch": _player_member("monarch"),
        "x-paid": _entity_member("x_paid"),
        "designations": _entity_member("designations"),
        "printed-loyalty": _printed_loyalty,
        "chosen-color": _chosen("colors"),
        "chosen-type": _chosen("type"),
        "keyword-granted-by-effect": _keyword_granted_by_effect,
        "initiative": _player_member("initiative"),
        "land-plays-not-one": _land_plays_not_one,
        "rewrite-buffered-events": _rewrite_buffered_events,
        "blockers-required-elsewhere": _forbidden_blocker("required_elsewhere"),
        "blockers-forbidden-own-line": _forbidden_blocker("responsible_own"),
        "effect-object-created": _effect_object_created,
    }
