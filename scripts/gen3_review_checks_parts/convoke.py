"""Convoke spells among the decision candidates: when they are offered and what they cost."""

from __future__ import annotations

import re

from .model import Line, TallyCheck, payload_of, state_of
from .sidecars import Sidecars

BEFORE_ATTACKS = ("untap", "upkeep", "draw", "main1", "combat_begin")
COST_LINE = re.compile(r"^mana cost: (.*)$", re.MULTILINE)


def is_convoke_row(row: dict) -> bool:
    """A sidecar row for the Convoke keyword: any line kind, keyed as a keyword."""
    if row.get("script_text") != "Convoke":
        return False
    return any(key["trait_kind"] == "keyword" for key in row["provenance"])


def card_has_convoke(sidecars: Sidecars, script_file: str) -> bool:
    data = sidecars.get(script_file)
    return data is not None and any(is_convoke_row(row) for row in data["lines"])


def mana_value(cost: str | None) -> int | None:
    """Mana value of a printed cost such as ``{2}{R}{R}``; ``X`` counts 0, any symbol 1."""
    if not cost or cost == "no cost":
        return None
    symbols = re.findall(r"\{([^}]+)\}", cost)
    return sum(int(s) if s.isdigit() else (0 if s == "X" else 1) for s in symbols)


def printed_cost(sidecars: Sidecars, script_file: str) -> str | None:
    data = sidecars.get(script_file)
    found = COST_LINE.search("\n".join(data["_text"])) if data else None
    return found.group(1).strip() if found else None


def reduction(adjusted: str | None, printed: str | None) -> str:
    after, before = mana_value(adjusted), mana_value(printed)
    if after is None or before is None:
        return "same" if after == before else "other"
    return "reduced" if after < before else ("same" if after == before else "other")


def _is_untapped_creature(entity: dict, player: str | None) -> bool:
    if entity.get("zone") != "battlefield" or entity.get("controller") != player:
        return False
    return "creature" in (entity.get("types") or ()) and not entity.get("tapped")


def _untapped_creatures(record: dict, player: str | None) -> int:
    entities = state_of(record).get("entities") or ()
    return sum(1 for e in entities if _is_untapped_creature(e, player))


def whose_turn(record: dict) -> str:
    state = state_of(record).get("global") or {}
    if state.get("active") != record.get("actor_player"):
        return "opponent-turn"
    return "own-before-attacks" if state.get("phase") in BEFORE_ATTACKS else "own-later"


class ConvokeCandidates(TallyCheck):
    """Convoke spell candidates: by phase and affordability, and by turn, untapped creatures
    and whether the adjusted cost fell below the printed one."""

    def __init__(self, sidecars: Sidecars) -> None:
        super().__init__()
        self.sidecars = sidecars
        self.total = 0

    def feed(self, record: dict) -> None:
        if record.get("subkind") != "decision":
            return
        for candidate in payload_of(record).get("candidates") or ():
            key = (candidate.get("ability") or [{}])[0]
            if self._is_convoke_spell(key):
                self._count(record, candidate, key["script_file"])

    def _is_convoke_spell(self, key: dict) -> bool:
        if key.get("trait_kind") != "spell":
            return False
        return card_has_convoke(self.sidecars, key["script_file"])

    def _count(self, record: dict, candidate: dict, script_file: str) -> None:
        self.total += 1
        phase = (state_of(record).get("global") or {}).get("phase")
        affordable = bool((candidate.get("verdict") or {}).get("affordable"))
        label = "affordable" if affordable else "unaffordable"
        self.add(f"convoke-by-phase[{phase},{label}]", True)
        creatures = _untapped_creatures(record, record.get("actor_player"))
        adjusted = (candidate.get("cost_after_adjustment") or {}).get("mana")
        change = reduction(adjusted, printed_cost(self.sidecars, script_file))
        crew = "untapped-creature" if creatures else "no-untapped-creature"
        self.add(f"convoke-by-turn[{whose_turn(record)},{crew},{change}]", True)

    def lines(self) -> list[Line]:
        found = [Line(name, pair[0], self.total) for name, pair in sorted(self.pairs.items())]
        return [Line("convoke-candidates", self.total, self.total), *found]
