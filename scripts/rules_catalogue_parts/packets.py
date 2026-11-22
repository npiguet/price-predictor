"""``packets``: units grouped into agent work packets."""

import json

from .locations import RULES
from .ordering import unit_sort_key
from .reading import load_rules, units_with_sizes


def _closes(current, full, chapter, chapter_of):
    """Whether the packet being filled ends before a unit of ``chapter``."""
    if not current:
        return False
    return full or chapter_of[current[-1]] != chapter


def _pack(sizes, chapter_of, max_chars):
    """Greedy in-order packing of units up to ``max_chars``, never splitting a unit or crossing
    a chapter."""
    packets, current, chars = [], [], 0
    for unit in sorted(sizes, key=unit_sort_key):
        full = chars + sizes[unit] > max_chars
        if _closes(current, full, chapter_of[unit], chapter_of):
            packets.append(current)
            current, chars = [], 0
        current.append(unit)
        chars += sizes[unit]
    packets.append(current)
    return packets


def cmd_packets(args):
    rules = load_rules()
    sizes, counts = units_with_sizes(rules)
    chapter_of = {r["unit"]: r["chapter"] for r in rules}
    out = [{"id": f"p{i:03d}", "chapter": chapter_of[p[0]], "units": p,
            "rules": sum(counts[u] for u in p), "chars": sum(sizes[u] for u in p)}
           for i, p in enumerate(_pack(sizes, chapter_of, args.max_chars))]
    (RULES / "packets.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(out)} packets; chars per packet max {max(p['chars'] for p in out)}")
