"""``packets``: units grouped into agent work packets."""

import json

from .locations import RULES
from .ordering import unit_sort_key
from .reading import load_rules, units_with_sizes


def cmd_packets(args):
    """Greedy in-order packing of units up to --max-chars, never splitting a unit or crossing
    a chapter."""
    rules = load_rules()
    sizes, counts = units_with_sizes(rules)
    chapter_of = {r["unit"]: r["chapter"] for r in rules}
    packets, current, chars = [], [], 0
    for unit in sorted(sizes, key=unit_sort_key):
        full = chars + sizes[unit] > args.max_chars
        if current and (full or chapter_of[current[-1]] != chapter_of[unit]):
            packets.append(current)
            current, chars = [], 0
        current.append(unit)
        chars += sizes[unit]
    packets.append(current)
    out = [{"id": f"p{i:03d}", "chapter": chapter_of[p[0]], "units": p,
            "rules": sum(counts[u] for u in p), "chars": sum(sizes[u] for u in p)}
           for i, p in enumerate(packets)]
    (RULES / "packets.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(out)} packets; chars per packet max {max(p['chars'] for p in out)}")
