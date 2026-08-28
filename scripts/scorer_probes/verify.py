"""Verify synergy_pairs.json: name resolution, set membership, MV/color/type checks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from cardtool import LOC, info, set_cards  # noqa: E402

data = json.loads((HERE / "synergy_pairs.json").read_text(encoding="utf-8"))

problems: list[str] = []
missing_embed: set[str] = set()
not_in_set: list[str] = []

for i, e in enumerate(data):
    code = e["set"]
    tag = f"[{i}] {code} {e['payoff']}"
    cards = set_cards(code)
    if not cards:
        problems.append(f"{tag}: SET {code} has no card list")
        continue
    names = [e["payoff"], e["control"], *e["enablers"]]
    for n in names:
        if LOC.embedding_path(n) is None:
            missing_embed.add(f"{n} (entry {i}, {code})")
        if n not in cards:
            not_in_set.append(f"{n} not in {code} (entry {i})")

    p, c = info(e["payoff"]), info(e["control"])
    if p is None or c is None:
        problems.append(f"{tag}: cannot read text for payoff/control")
        continue
    if abs(p["mv"] - c["mv"]) > 1:
        problems.append(
            f"{tag}: MV mismatch payoff={p['mv']} ({p['mana_cost']}) "
            f"control={e['control']}={c['mv']} ({c['mana_cost']})"
        )
    pc, cc = p["colors"], c["colors"]
    if pc or cc:
        if not (pc & cc):
            problems.append(
                f"{tag}: color mismatch payoff={sorted(pc)} "
                f"control={e['control']}={sorted(cc)}"
            )
    p_is_cre = "creature" in p["types"]
    c_is_cre = "creature" in c["types"]
    if p_is_cre != c_is_cre:
        problems.append(
            f"{tag}: type mismatch payoff='{p['types']}' "
            f"control={e['control']}='{c['types']}'"
        )
    if e["strength"] not in ("strong", "mild"):
        problems.append(f"{tag}: bad strength {e['strength']!r}")
    if not (1 <= len(e["enablers"]) <= 4):
        problems.append(f"{tag}: enabler count {len(e['enablers'])}")
    if e["control"] in e["enablers"] or e["control"] == e["payoff"]:
        problems.append(f"{tag}: control overlaps payoff/enablers")

print(f"entries: {len(data)}")
print(f"sets: {len(sorted({e['set'] for e in data}))} -> {sorted({e['set'] for e in data})}")
print(f"mechanisms: {len({e['mechanism'] for e in data})}")
print()
print("--- MISSING EMBEDDINGS ---")
for m in sorted(missing_embed):
    print(" ", m)
print(f"  ({len(missing_embed)})")
print("--- NOT IN CLAIMED SET ---")
for m in not_in_set:
    print(" ", m)
print(f"  ({len(not_in_set)})")
print("--- OTHER PROBLEMS ---")
for m in problems:
    print(" ", m)
print(f"  ({len(problems)})")
