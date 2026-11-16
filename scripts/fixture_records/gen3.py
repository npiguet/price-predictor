"""Gen-3's own categories: story 2's acceptance scenarios, each
``turn_action`` subkind and failed check, the anchored legality records, the
new snapshot members, the combats the keyword detector's real-record tests
read and the fork tags. Matched before gen-2's and the shared ones."""

from __future__ import annotations

from collections.abc import Callable

from effects.domain.rule_detectors.combat_keyword_effect import FAMILY as COMBAT_KEYWORD
from effects.infrastructure.record_codec import record_from_dict
from fixture_records.acceptance import (
    acceptance_categories,
    anchored_attackers,
    attacker_tapped_categories,
)
from fixture_records.members import snapshot_member_categories
from fixture_records.shapes import candidate, events, failed_checks
from fixture_records.sidecars import Sidecars

#: Gen-3 ``turn_action`` subkinds and decision ``failed_checks`` identifiers,
#: one fixture category each (contracts/record-schema-delta.md).
_TURN_ACTION_SUBKINDS = ("draw", "cleanup_discard", "untap", "phasing", "day_night",
                         "lore_counter")
_FAILED_CHECKS = ("split_second", "casting_speed", "whose_turn", "phase", "combat",
                  "activation_limit", "land_play_limit", "script_condition", "zone",
                  "static_prohibition", "cost", "script_restriction")

Test = Callable[[dict], bool]


def _turn_action(subkind: str) -> Test:
    """A ``turn_action`` of ``subkind``; a draw only when a card was drawn."""
    def test(record: dict) -> bool:
        if record["kind"] != "turn_action" or record.get("subkind") != subkind:
            return False
        return subkind != "draw" or any(
            event["type"] == "card_drawn" for event in events(record))
    return test


def _failed_check(check: str) -> Test:
    def test(record: dict) -> bool:
        found = candidate(record)
        return found is not None and check in failed_checks(found)
    return test


def _land_play(test: Callable[[dict, list[str]], bool]) -> Test:
    """A decision whose candidate is a land play (FR-033a) and passes ``test``
    on its verdict and failed checks."""
    def check(record: dict) -> bool:
        found = candidate(record)
        return (found is not None and found.get("land_play") is True
                and test(found.get("verdict") or {}, failed_checks(found)))
    return check


def _playable(verdict: dict, failed: list[str]) -> bool:
    return verdict.get("can_play") is True


def _off_turn(verdict: dict, failed: list[str]) -> bool:
    return "whose_turn" in failed


def _not_sorcery_time(verdict: dict, failed: list[str]) -> bool:
    return "casting_speed" in failed and "whose_turn" not in failed


def _must_attack(payload: dict, anchor: dict) -> bool:
    return payload.get("must_attack") is True


def _no_defender(payload: dict, anchor: dict) -> bool:
    return payload.get("legal_defenders") == []


def _blockers_must_block(record: dict) -> bool:
    return (record["kind"] == "playability" and record.get("subkind") == "blockers"
            and bool(record["payload"].get("must_block")))


def _legality_categories() -> dict[str, Test]:
    """The anchored attackers and blockers shapes, each failed check and the
    land-play candidates: playable, past the turn's limit, off their player's
    turn, and on it at a time no sorcery could be cast."""
    return {
        **attacker_tapped_categories(),
        "attacker-must-attack": anchored_attackers(_must_attack),
        "attacker-no-defender": anchored_attackers(_no_defender),
        "blockers-must-block": _blockers_must_block,
        **{f"failed-check-{check}": _failed_check(check) for check in _FAILED_CHECKS},
        "land-play-playable": _land_play(_playable),
        "land-play-off-turn": _land_play(_off_turn),
        "land-play-not-sorcery-time": _land_play(_not_sorcery_time),
    }


def _combat_keyword(lookup, keyword: str, polarity: str | None) -> Test:
    """An observed combat ``combat_keyword_effect`` matches for ``keyword``, on
    the side ``polarity`` names (either, when None): the detector picks the
    record, so the fixture holds what the family's real-record tests read."""
    def test(record: dict) -> bool:
        if record["kind"] != "combat" or record["fork"]:
            return False
        found = COMBAT_KEYWORD.match(record_from_dict(record), lookup, {"keyword": keyword})
        return found is not None and (polarity is None or found.polarity.value == polarity)
    return test


def _directed_fork(record: dict) -> bool:
    return record.get("directed") is not None


def _copy_dropped(record: dict) -> bool:
    return bool(record.get("copy_dropped"))


def _combat_and_fork_categories(lookup) -> dict[str, Test]:
    return {
        "combat-vigilance-fires": _combat_keyword(lookup, "vigilance", "fires"),
        "combat-vigilance-near-miss": _combat_keyword(lookup, "vigilance", "does_not_fire"),
        "combat-first-strike-regular": _combat_keyword(lookup, "first_strike", None),
        "directed-fork": _directed_fork,
        "copy-dropped": _copy_dropped,
    }


def gen3_categories(sidecars: Sidecars, lookup) -> dict[str, Test]:
    """Gen-3's own categories, in the order they are matched; ``lookup`` is a
    sidecar cache over the same trees, for the detectors."""
    return {
        **acceptance_categories(sidecars),
        **{f"turn-action-{subkind}": _turn_action(subkind)
           for subkind in _TURN_ACTION_SUBKINDS},
        **_legality_categories(),
        **snapshot_member_categories(),
        **_combat_and_fork_categories(lookup),
    }
