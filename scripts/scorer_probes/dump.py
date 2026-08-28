"""Dump a set's cards filtered by color / MV / type / rarity.

Usage: dump.py SET [--c W] [--mv 1-3] [--t creature] [--r CU] [--re regex]
"""
from __future__ import annotations

import argparse
import re
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardtool import card_text, info, set_cards, summary  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("set")
p.add_argument("--c", default="")
p.add_argument("--mv", default="")
p.add_argument("--t", default="")
p.add_argument("--r", default="")
p.add_argument("--re", dest="rx", default="")
a = p.parse_args()

lo, hi = 0, 99
if a.mv:
    if "-" in a.mv:
        lo, hi = (int(x) for x in a.mv.split("-"))
    else:
        lo = hi = int(a.mv)

cards = set_cards(a.set)
if not cards:
    print(f"!! set {a.set} not found or empty")
    sys.exit(1)
rx = re.compile(a.rx, re.I) if a.rx else None
n = 0
for name, rar in sorted(cards.items()):
    i = info(name)
    if i is None:
        continue
    if a.c:
        want = set(a.c.upper())
        if want == {"C"}:
            if i["colors"]:
                continue
        elif not (want & i["colors"]):
            continue
    if not (lo <= i["mv"] <= hi):
        continue
    if a.t and a.t.lower() not in i["types"].lower():
        continue
    if a.r and rar not in a.r.upper():
        continue
    if rx and not rx.search(card_text(name) or ""):
        continue
    print(f"{rar} {name:<30} {summary(name)}")
    n += 1
print(f"-- {n} shown of {len(cards)} in {a.set}")
