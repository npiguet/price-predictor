"""Converted-card text surgery for the counterfactual probes.

Read and rewrite the fields of a converted card (``mana cost:``, ``types:``,
``power toughness:``, ability lines), plus the keyword and tribe vocabularies
the C battery substitutes. Every edit works on the non-blank lines of a
name-stripped text and re-joins them with ``\\n``, so a line index from
:func:`lines` addresses the same line in every helper.

Design rule inherited from R1/R2: positional placebos are large and negatively
biased, token substitutions are clean. So every contrast is built as a
*paired same-card difference between two arms with identical layout*.
"""

from __future__ import annotations

import re
from typing import Sequence

ABILITY_PREFIXES = ("static:", "spell[", "activated[", "triggered", "replacement")

TYPE_WORDS = frozenset({
    "artifact", "battle", "conspiracy", "creature", "dungeon", "emblem",
    "enchantment", "instant", "kindred", "land", "phenomenon", "plane",
    "planeswalker", "scheme", "sorcery", "tribal", "vanguard",
})
SUPERTYPE_WORDS = frozenset({
    "basic", "elite", "host", "legendary", "ongoing", "snow", "world",
})

# The sixteen keywords the battery substitutes, and the card-table flag of each.
KEYWORD_FLAGS = {
    "flying": "kw_flying", "lifelink": "kw_lifelink",
    "deathtouch": "kw_deathtouch", "vigilance": "kw_vigilance",
    "first strike": "kw_first_strike", "double strike": "kw_double_strike",
    "trample": "kw_trample", "haste": "kw_haste", "reach": "kw_reach",
    "menace": "kw_menace", "defender": "kw_defender",
    "hexproof": "kw_hexproof", "shroud": "kw_shroud",
    "indestructible": "kw_indestructible", "flash": "kw_flash",
    "ward {2}": "kw_ward",
}
KEYWORDS = list(KEYWORD_FLAGS)

TRIBES = [
    "dragon", "angel", "demon", "sphinx", "hydra", "wurm", "giant",
    "zombie", "goblin", "elf", "human", "soldier", "bird", "lizard",
    "spirit", "beast", "wall", "rat",
]

PT_RE = re.compile(r"^power toughness:\s*(-?\d+)/(-?\d+)\s*$")
COST_RE = re.compile(r"^(?:\{(\d+)\})?((?:\{[WUBRG]\})*)$")


def lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def find_line(ls: Sequence[str], prefix: str) -> int:
    for i, ln in enumerate(ls):
        if ln.startswith(prefix):
            return i
    return -1


def get_field(text: str, prefix: str) -> str | None:
    for ln in lines(text):
        if ln.startswith(prefix):
            return ln[len(prefix):].strip()
    return None


def set_field(text: str, prefix: str, value: str) -> str:
    """``text`` with its first ``prefix`` line rewritten; unchanged when it has none."""
    ls = lines(text)
    i = find_line(ls, prefix)
    if i < 0:
        return text
    ls[i] = f"{prefix} {value}"
    return "\n".join(ls)


def set_cost(text: str, cost: str) -> str:
    return set_field(text, "mana cost:", cost)


def set_pt(text: str, power: int, toughness: int) -> str:
    return set_field(text, "power toughness:", f"{power}/{toughness}")


def has_literal_pt(text: str) -> bool:
    """True for a plain ``N/M`` statline (not ``*/1+*``, ``X/X``, ...)."""
    pt = next((ln for ln in lines(text) if ln.startswith("power toughness:")), "")
    return bool(PT_RE.match(pt))


def ability_lines(text: str) -> list[int]:
    return [i for i, ln in enumerate(lines(text)) if ln.startswith(ABILITY_PREFIXES)]


def replace_in_line(text: str, index: int, new_line: str) -> str:
    ls = lines(text)
    ls[index] = new_line
    return "\n".join(ls)


def _insert_slot(ls: Sequence[str], new_line: str) -> int:
    """Canonical insertion index for ``new_line``: statics stay in field order."""
    if new_line.startswith("static:"):
        st = [i for i, ln in enumerate(ls) if ln.startswith("static:")]
        if st:
            return st[-1] + 1
        for pref in ("power toughness:", "types:", "mana cost:"):
            i = find_line(ls, pref)
            if i >= 0:
                return i + 1
    return len(ls)


def add_line(text: str, new_line: str) -> str:
    ls = lines(text)
    ls.insert(_insert_slot(ls, new_line), new_line)
    return "\n".join(ls)


def drop_line(text: str, index: int) -> str:
    ls = lines(text)
    del ls[index]
    return "\n".join(ls)


def subtypes(type_words: Sequence[str]) -> list[str]:
    return [w for w in type_words if w not in TYPE_WORDS and w not in SUPERTYPE_WORDS]


def sole_tribe(text: str) -> tuple[int, str] | None:
    """``(types-line index, tribe)`` for a creature whose only subtype is one of
    :data:`TRIBES`; None otherwise."""
    for i, ln in enumerate(lines(text)):
        if not ln.startswith("types:"):
            continue
        words = ln[6:].split()
        subs = subtypes(words)
        if "creature" in words and len(subs) == 1 and subs[0] in TRIBES:
            return i, subs[0]
        return None
    return None


def parse_cost(cost: str) -> tuple[int, str] | None:
    """``(generic, pips)`` for a plain generic+coloured-pips cost, else None."""
    m = COST_RE.match(cost)
    if not m or (not m.group(1) and not m.group(2)):
        return None
    return int(m.group(1) or 0), m.group(2) or ""


def fmt_cost(k: int, pips: str) -> str:
    if k == 0:
        return pips if pips else "{0}"
    return f"{{{k}}}{pips}"
