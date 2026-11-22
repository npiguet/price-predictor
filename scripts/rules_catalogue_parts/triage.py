"""The triage sidecars: which rules need a scenario, their validation, and ``premark``."""

import json
from collections import defaultdict

from .locations import RULES
from .ordering import rule_sort_key
from .reading import load_rules
from .vocabulary import CATEGORIES, OBSERVABILITY, RECORD_KINDS, SCENARIO_CATEGORIES

PREMARKED_CHAPTERS = {"8": "multiplayer rules; collection plays two-player games only",
                      "9": "casual variant; collection plays two-player sealed games only"}


def _premarked_by_unit():
    by_unit = defaultdict(list)
    for r in load_rules():
        if r["chapter"] in PREMARKED_CHAPTERS:
            by_unit[r["unit"]].append(r)
    return by_unit


def _write_premarked(unit, rules):
    with open(RULES / "triage" / f"{unit}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rules:
            f.write(json.dumps({"id": r["id"], "category": "out_of_scope",
                                "observability": "none", "record_kinds": [], "outcome": "",
                                "note": PREMARKED_CHAPTERS[r["chapter"]]}) + "\n")


def cmd_premark(_args):
    """Triage chapters 8 and 9 as out_of_scope without an agent."""
    by_unit = _premarked_by_unit()
    (RULES / "triage").mkdir(exist_ok=True)
    for unit, rs in by_unit.items():
        _write_premarked(unit, rs)
    print(f"premarked {len(by_unit)} units")


def needs_scenario(triage_row):
    return (triage_row["category"] in SCENARIO_CATEGORIES
            and triage_row["observability"] != "none")


def wanted_rules(triage):
    """The rules that need a scenario, and so a heads line too."""
    return {r["id"] for r in triage if needs_scenario(r)}


def _row_errors(row):
    rid = row.get("id")
    errs = []
    if row.get("category") not in CATEGORIES:
        errs.append(f"{rid}: category {row.get('category')!r}")
    if row.get("observability") not in OBSERVABILITY:
        errs.append(f"{rid}: observability {row.get('observability')!r}")
    bad = set(row.get("record_kinds") or []) - RECORD_KINDS
    if bad:
        errs.append(f"{rid}: record_kinds {sorted(bad)}")
    return errs


def validate_triage(rows, rule_ids):
    errs = []
    seen = [r.get("id") for r in rows]
    missing = sorted(set(rule_ids) - set(seen), key=rule_sort_key)
    extra = sorted(set(seen) - set(rule_ids))
    if missing:
        errs.append(f"missing rules {missing}")
    if extra:
        errs.append(f"unknown rules {extra}")
    for r in rows:
        errs += _row_errors(r)
    return errs
