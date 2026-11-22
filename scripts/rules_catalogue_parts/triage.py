"""The triage sidecars: which rules need a scenario, their validation, and ``premark``."""

import json
from collections import defaultdict

from .locations import RULES
from .ordering import rule_sort_key
from .reading import load_rules
from .vocabulary import CATEGORIES, OBSERVABILITY, RECORD_KINDS, SCENARIO_CATEGORIES

PREMARKED_CHAPTERS = {"8": "multiplayer rules; collection plays two-player games only",
                      "9": "casual variant; collection plays two-player sealed games only"}


def cmd_premark(_args):
    """Triage chapters 8 and 9 as out_of_scope without an agent."""
    by_unit = defaultdict(list)
    for r in load_rules():
        if r["chapter"] in PREMARKED_CHAPTERS:
            by_unit[r["unit"]].append(r)
    (RULES / "triage").mkdir(exist_ok=True)
    for unit, rs in by_unit.items():
        with open(RULES / "triage" / f"{unit}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps({"id": r["id"], "category": "out_of_scope",
                                    "observability": "none", "record_kinds": [], "outcome": "",
                                    "note": PREMARKED_CHAPTERS[r["chapter"]]}) + "\n")
    print(f"premarked {len(by_unit)} units")


def needs_scenario(triage_row):
    return (triage_row["category"] in SCENARIO_CATEGORIES
            and triage_row["observability"] != "none")


def wanted_rules(triage):
    """The rules that need a scenario, and so a heads line too."""
    return {r["id"] for r in triage if needs_scenario(r)}


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
        if r.get("category") not in CATEGORIES:
            errs.append(f"{r.get('id')}: category {r.get('category')!r}")
        if r.get("observability") not in OBSERVABILITY:
            errs.append(f"{r.get('id')}: observability {r.get('observability')!r}")
        bad = set(r.get("record_kinds") or []) - RECORD_KINDS
        if bad:
            errs.append(f"{r.get('id')}: record_kinds {sorted(bad)}")
    return errs
