"""Forced plays: effect halves of an interventional fork that nobody cast."""

from __future__ import annotations

from collections import Counter

from .model import (
    Line,
    TallyCheck,
    entities_of,
    events_of,
    first_key,
    is_forced_half,
    payload_of,
    phase_of,
    state_of,
)

COMBAT_PHASES = (
    "combat_declare_attackers",
    "combat_declare_blockers",
    "combat_damage",
    "combat_end",
)
NOT_FORCED_TRAITS = ("trigger", "static", "replacement")


def _is_playable_forced(record: dict) -> bool:
    if not is_forced_half(record) or payload_of(record).get("outcome") is None:
        return False
    return first_key(record).get("trait_kind") not in NOT_FORCED_TRAITS


def _is_protection_line(line: dict) -> bool:
    if line.get("kind") != "keyword" or line.get("keys"):
        return False
    return bool(line.get("granted")) and str(line.get("fragment", "")).startswith("Protection")


def _protection_mark(record: dict) -> bool:
    return any(
        _is_protection_line(line)
        for entity in entities_of(record)
        if entity.get("zone") == "battlefield"
        for line in entity.get("lines") or ()
    )


def _active_library(record: dict) -> int | None:
    state = state_of(record)
    active = (state.get("global") or {}).get("active")
    player = next((p for p in state.get("players") or [] if p.get("id") == active), None)
    return player.get("library") if player else None


def _turn_of(record: dict) -> tuple:
    return (record.get("game_id"), (state_of(record).get("global") or {}).get("turn"))


def _is_interventional_fork(record: dict) -> bool:
    if not record.get("fork") or record.get("directed"):
        return False
    return bool(record.get("interventional"))


def _is_live_draw(record: dict) -> bool:
    if record.get("fork"):
        return False
    return record.get("kind") == "turn_action" and record.get("subkind") == "draw"


class ForcedHalves(TallyCheck):
    """Forced effect halves by phase, and those stamped before the step's turn-based action.

    A draw-phase half whose active library still holds the pre-draw count, or a
    combat-step half with nothing in combat, was stamped before that action.
    Turns that held a protection fork are left out, as the review left them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.candidates: list[tuple[tuple, str | None, bool, int | None]] = []
        self.protection_turns: set[tuple] = set()
        self.draw_library: dict[tuple, int | None] = {}

    def feed(self, record: dict) -> None:
        turn = _turn_of(record)
        if _is_live_draw(record):
            self.draw_library[turn] = _active_library(record)
        if _is_interventional_fork(record) and _protection_mark(record):
            self.protection_turns.add(turn)
        if _is_playable_forced(record):
            in_combat = any(e.get("combat") for e in entities_of(record))
            self.candidates.append((turn, phase_of(record), in_combat, _active_library(record)))

    def lines(self) -> list[Line]:
        kept = [c for c in self.candidates if c[0] not in self.protection_turns]
        by_phase = Counter(c[1] for c in kept)
        for phase, count in sorted(by_phase.items(), key=lambda item: str(item[0])):
            self.pairs[f"forced-halves[{phase}]"] = [count, len(kept)]
        early = sum(1 for c in kept if self._before_turn_action(c))
        self.pairs["forced-halves-stamped-before-turn-action"] = [early, len(kept)]
        self.pairs["forced-halves"] = [len(kept), len(kept)]
        return super().lines()

    def _before_turn_action(self, candidate: tuple) -> bool:
        turn, phase, in_combat, library = candidate
        if phase in COMBAT_PHASES:
            return not in_combat
        return phase == "draw" and library == self.draw_library.get(turn, -1)


def _attached_hosts(record: dict) -> set[str]:
    links = [x for x in state_of(record).get("links") or [] if x.get("type") == "attached_to"]
    return {x["to"] for x in links if isinstance(x.get("to"), str)}


def _is_state_based_leave(event: dict) -> bool:
    if event.get("type") != "zone_change" or event.get("attributed_to") != "unresolved":
        return False
    return (event.get("params") or {}).get("from_zone") == "battlefield"


def _unresolved_leaves(record: dict) -> list[str]:
    """Subjects of battlefield exits stamped as state-based (`unresolved`)."""
    leaves = [e for e in events_of(record) if _is_state_based_leave(e)]
    return [subject for event in leaves for subject in event.get("subjects") or ()]


def _has_unattached(record: dict) -> bool:
    return any(
        e.get("type") == "unattached" and e.get("attributed_to") == "unresolved"
        for e in events_of(record)
    )


def _is_resolved_forced_effect(record: dict) -> bool:
    if not is_forced_half(record) or payload_of(record).get("outcome") != "resolved":
        return False
    return first_key(record).get("trait_kind") != "trigger"


class ForcedUnattached(TallyCheck):
    """Forced halves where a hosting permanent died state-based, without an `unattached` event."""

    NAMES = ("forced-deaths-without-unattached",)

    def feed(self, record: dict) -> None:
        if not _is_resolved_forced_effect(record):
            return
        hosts = _attached_hosts(record)
        if any(subject in hosts for subject in _unresolved_leaves(record)):
            self.add("forced-deaths-without-unattached", not _has_unattached(record))
