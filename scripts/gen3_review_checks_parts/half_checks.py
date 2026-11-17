"""Effect halves: Equip, counterspells and fizzles."""

from __future__ import annotations

import re

from .model import (
    TallyCheck,
    entities_of,
    events_of,
    first_key,
    is_effect_half,
    payload_of,
    state_of,
)
from .sidecars import Sidecars

STAT_EVENTS = frozenset({"pt_change", "keyword_change", "ability_change", "control_change"})
STACK_APIS = re.compile(
    r"\b(?:SP|AB|DB)\$ (?:Counter|CopySpellAbility|ChangeTargets|ControlSpell)\b"
)


def _is_live_half_of(record: dict, trait_kind: str) -> bool:
    if not is_effect_half(record) or record.get("fork"):
        return False
    return first_key(record).get("trait_kind") == trait_kind


class EquipHalves(TallyCheck):
    """Live keyword effect halves holding `attached`, with and without a stat change."""

    NAMES = ("equip-halves-with-stat-change", "equip-halves-without-stat-change")

    def feed(self, record: dict) -> None:
        if not _is_live_half_of(record, "keyword"):
            return
        types = {e.get("type") for e in events_of(record)}
        if "attached" in types:
            has_stat = bool(types & STAT_EVENTS)
            self.add("equip-halves-with-stat-change", has_stat)
            self.add("equip-halves-without-stat-change", not has_stat)


def _is_random_seat_live_half(record: dict) -> bool:
    if not is_effect_half(record) or record.get("fork"):
        return False
    return bool(record.get("random_seat"))


class CounterHalves(TallyCheck):
    """Random-seat effect halves of stack-targeting scripts, with and without `spell_countered`."""

    NAMES = ("counter-halves-random-seat-countered", "counter-halves-random-seat-not-countered")

    def __init__(self, sidecars: Sidecars) -> None:
        super().__init__()
        self.sidecars = sidecars

    def feed(self, record: dict) -> None:
        if not _is_random_seat_live_half(record):
            return
        if self._targets_the_stack(first_key(record)):
            countered = any(e.get("type") == "spell_countered" for e in events_of(record))
            self.add("counter-halves-random-seat-countered", countered)
            self.add("counter-halves-random-seat-not-countered", not countered)

    def _targets_the_stack(self, key: dict) -> bool:
        rows = self.sidecars.rows_holding(key) if key else None
        return any(STACK_APIS.search(row.get("script_text") or "") for row in rows or ())


class FizzledHalves(TallyCheck):
    """Fizzled effect halves whose acting item shows no targets."""

    NAMES = ("fizzled-halves-with-zero-targets",)

    def feed(self, record: dict) -> None:
        if not is_effect_half(record) or payload_of(record).get("outcome") != "fizzled":
            return
        source = (state_of(record).get("refs") or {}).get("source")
        item = next((e for e in entities_of(record) if e.get("id") == source), None)
        extras = item.get("stack_extras") if item else None
        if extras is not None:
            self.add("fizzled-halves-with-zero-targets", not extras.get("targets"))
