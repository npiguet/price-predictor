"""Markdown tables and console text for the reports the probe scripts print or write."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence


def utf8_console() -> None:
    """Print UTF-8 whatever the console's code page: the reports carry glyphs
    (², →, ≥) a Windows cp1252 console cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")


def num(x, digits: int = 4, missing: str = "-") -> str:
    """One number as a table cell: ``missing`` for None, ``nan`` for NaN, and
    anything that is not a number as its own text."""
    if x is None:
        return missing
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    return "nan" if f != f else f"{f:.{digits}f}"


def banner(title: str, width: int = 78) -> None:
    """Print ``title`` between two rules, opening a section of a console report."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def _cell(value) -> str:
    return str(value).replace("|", "\\|")


def md_table(headers: Sequence, rows: Iterable[Sequence], align: str = "") -> str:
    """A pipe table; ``align`` holds one of ``l``/``r`` per column (default: left).
    A ``|`` inside a cell is escaped, so card text cannot split a column."""
    rule = {"l": "---", "r": "---:"}
    marks = [rule[align[i]] if i < len(align) else "---" for i in range(len(headers))]
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |",
           "|" + "|".join(marks) + "|"]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def print_table(headers: Sequence, rows: Iterable[Sequence], align: str = "") -> None:
    print(md_table(headers, rows, align))
