"""Gen-3 categories for what became of a stack item and which line acted:
an effect half fizzled, partially fizzled or declined, a forced half, an
ability countered, a saga's chapter II resolving, a Class at level 2 whose
level-2 line acts, mana a pool lost, and colourless mana paid or floating.
Matched first, since each is rare in a pilot."""

from __future__ import annotations

import re
from collections.abc import Callable

from gen3_review_checks_parts.model import is_forced_half

from fixture_records.shapes import (
    acting_key,
    entities,
    events,
    is_effect_resolution,
    lines,
    players,
)
from fixture_records.sidecars import Sidecars

Test = Callable[[dict], bool]

#: The resolution outcomes other than ``resolved`` an effect half writes.
_OUTCOMES = ("fizzled", "partially_fizzled", "declined")
_CHAPTERS = re.compile(r"Chapter\$ ([\d,]+)")
_CLASS_PREFIX = "class:"


def _outcome(outcome: str) -> Test:
    """An observed effect half whose item came to ``outcome``."""
    return lambda record: (is_effect_resolution(record)
                           and record["payload"].get("outcome") == outcome)


def _forced_half(record: dict) -> bool:
    """An intervention's forced play, as the review's forced checks read it."""
    return is_forced_half(record) and record.get("fork_kind") == "intervention"


def _event_of(event_type: str) -> Test:
    return lambda record: any(event["type"] == event_type for event in events(record))


def _rewrites(event_type: str) -> Test:
    return lambda record: (record["kind"] == "rewrite"
                           and record["payload"]["incoming"]["type"] == event_type)


def _chapters(line: dict | None) -> set[str]:
    """The chapter numbers a sidecar's chapter line names (``Chapter$ 2,3``)."""
    if line is None or line.get("line_kind") != "chapter":
        return set()
    found = _CHAPTERS.search(line.get("script_text") or "")
    return set(found.group(1).split(",")) if found else set()


def _saga_chapter(sidecars: Sidecars, chapter: str) -> Test:
    """An observed effect half whose acting key names a chapter line with ``chapter``."""
    def test(record: dict) -> bool:
        key = acting_key(record)
        return (is_effect_resolution(record) and key is not None
                and chapter in _chapters(sidecars.line(key)))
    return test


def _class_level(line: dict) -> int | None:
    """N of a ``Class:N`` keyword line in force, else None."""
    fragment = str(line.get("fragment") or "").lower()
    if line["kind"] != "keyword" or not fragment.startswith(_CLASS_PREFIX):
        return None
    return int(fragment.split(":")[1]) if line.get("active") else None


def _levels(entity: dict) -> dict[int, dict]:
    found = {}
    for line in lines(entity):
        level = _class_level(line)
        if level is not None:
            found[level] = line
    return found


def _acts_at_level(entity: dict, key: dict, level: int) -> bool:
    levels = _levels(entity)
    return bool(levels) and max(levels) == level and key in levels[level]["keys"]


def _class_at_level(level: int) -> Test:
    """A record whose acting key is the ``Class:level`` line of a Class at that level."""
    def test(record: dict) -> bool:
        key = acting_key(record)
        return key is not None and any(_acts_at_level(entity, key, level)
                                       for entity in entities(record))
    return test


def _paid_colourless(record: dict) -> bool:
    """A cost half paying colourless mana (``C``)."""
    if record["kind"] != "resolution" or record.get("moment") != "activation":
        return False
    return (record["payload"]["costs"].get("mana_by_color") or {}).get("C", 0) > 0


def _floating_colourless(record: dict) -> bool:
    return any((player.get("floating_mana") or {}).get("C", 0) > 0
               for player in players(record))


def resolution_categories(sidecars: Sidecars) -> dict[str, Test]:
    """The categories, in the order they are matched."""
    return {
        **{f"outcome-{outcome}": _outcome(outcome) for outcome in _OUTCOMES},
        "forced-half": _forced_half,
        "ability-countered": _event_of("ability_countered"),
        "saga-chapter-2": _saga_chapter(sidecars, "2"),
        "class-level-2": _class_at_level(2),
        "mana-lost": _event_of("mana_lost"),
        "mana-lost-rewrite": _rewrites("mana_lost"),
        "mana-colourless-paid": _paid_colourless,
        "mana-colourless-floating": _floating_colourless,
    }
