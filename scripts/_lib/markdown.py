"""Markdown tables for the reports the probe scripts print or write."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def md_table(headers: Sequence, rows: Iterable[Sequence], align: str = "") -> str:
    """A pipe table; ``align`` holds one of ``l``/``r`` per column (default: left)."""
    rule = {"l": "---", "r": "---:"}
    marks = [rule[align[i]] if i < len(align) else "---" for i in range(len(headers))]
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(marks) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def print_table(headers: Sequence, rows: Iterable[Sequence], align: str = "") -> None:
    print(md_table(headers, rows, align))
