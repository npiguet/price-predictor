"""Enumerate sealed-legal MTG set codes from MTGJSON AllPrintings.json."""

from __future__ import annotations

import json
from pathlib import Path


def eligible_sealed_sets(
    all_printings_path: Path = Path("resources/AllPrintings.json"),
) -> list[str]:
    """Return set codes eligible for sealed-format play.

    Mirrors the criteria used by MTG Forge's
    ``MatchGenerator.computeEligibleSets()``: a set is eligible iff it has a
    draft booster template and is not an un-set (``type == "funny"``).

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
        booster = set_info.get("booster") or {}
        if not isinstance(booster, dict) or "draft" not in booster:
            continue
        if set_info.get("type") == "funny":
            continue
        eligible.append(set_code)

    return eligible
