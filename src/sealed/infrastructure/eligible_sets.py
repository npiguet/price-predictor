"""Enumerate sealed-legal MTG set codes from MTGJSON AllPrintings.json."""

from __future__ import annotations

import json
from pathlib import Path

MIN_BOOSTER_CARDS = 12
"""Minimum draft-booster size for a set to be considered sealed-viable.

Matches the rule Forge itself uses in
``AdventureEventData.isValidDraftBlock()`` and the mirrored Java filter in
``MatchGenerator.computeEligibleSets()``. Excludes historical sets (DRK, FEM, …)
whose 8-card boosters are too small to yield a real sealed pool.
"""


def eligible_sealed_sets(
    all_printings_path: Path = Path("resources/AllPrintings.json"),
) -> list[str]:
    """Return set codes eligible for sealed-format play.

    Mirrors the criteria used by MTG Forge's
    ``MatchGenerator.computeEligibleSets()``: a set is eligible iff it has a
    draft booster template, is not an un-set (``type == "funny"``), and the
    draft booster contains at least ``MIN_BOOSTER_CARDS`` (=12) cards.

    Args:
        all_printings_path: Path to MTGJSON AllPrintings.json. Defaults to the
            standard location under ``resources/``.

    Returns:
        List of set codes (preserves source ordering).

    Raises:
        FileNotFoundError: If ``all_printings_path`` does not exist.
    """
    with open(all_printings_path, encoding="utf-8") as f:
        data = json.load(f)

    eligible: list[str] = []
    for set_code, set_info in data.get("data", {}).items():
        if not isinstance(set_info, dict):
            continue
        if set_info.get("type") == "funny":
            continue
        booster = set_info.get("booster") or {}
        draft = booster.get("draft") if isinstance(booster, dict) else None
        if not isinstance(draft, dict):
            continue
        if _draft_booster_card_count(draft) < MIN_BOOSTER_CARDS:
            continue
        eligible.append(set_code)

    return eligible


def _draft_booster_card_count(draft: dict) -> int:
    """Return the total card count produced by the draft booster template.

    MTGJSON's ``booster.draft`` has a ``boosters`` list of variants, each with
    a ``contents`` dict mapping sheet names to card counts. All variants of a
    given set produce the same total, so we sum the first one. Returns ``0``
    when the structure is missing or malformed (which disqualifies the set).
    """
    boosters = draft.get("boosters")
    if not isinstance(boosters, list) or not boosters:
        return 0
    contents = boosters[0].get("contents") if isinstance(boosters[0], dict) else None
    if not isinstance(contents, dict):
        return 0
    total = 0
    for value in contents.values():
        if isinstance(value, int):
            total += value
    return total
