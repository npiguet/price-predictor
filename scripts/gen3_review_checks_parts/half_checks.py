"""Effect halves: Equip and fizzles."""

from __future__ import annotations

from .model import (
    TallyCheck,
    entities_of,
    events_of,
    first_key,
    is_effect_half,
    payload_of,
    state_of,
)

STAT_EVENTS = frozenset({"pt_change", "keyword_change", "ability_change", "control_change"})


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
