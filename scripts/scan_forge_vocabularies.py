"""Write ``src/effects/domain/forge_vocabularies.py``: Forge's closed lists the input reads.

The effect head's card slot reads three closed lists Forge defines (root spec
section 9.4, section 9.6):

- **Subtypes** (``forge-gui/res/lists/TypeLists.txt``): every section's subtypes,
  basic land, land, creature, artifact, enchantment, spell, planeswalker, dungeon, battle and
  planar, each one learned embedding summed into the card slot and shared with a
  chosen creature type.
- **Named counters** (``forge-game/.../card/CounterEnumType.java``): each counter type
  as a record spells it (its name upper-cased, then
  ``effects.domain.state_snapshot.normalize_counter_type``), one scalar input each.
- **Keyword counters** (``CounterKeywordType.java``'s rule 122.1b list): a counter that
  grants a keyword, read as a granted keyword line rather than as a scalar.

The lists are written in Forge's own order, not sorted, so a Forge merge that adds a
subtype or a counter at the end of its list leaves every earlier row where it was.
A checkpoint records the lists it trained on and a loader refuses another.

Usage: python scripts/scan_forge_vocabularies.py [--forge DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _lib.paths import FORGE_RES, REPO

from effects.domain.keyword_names import normalize_keyword
from effects.domain.state_snapshot import normalize_counter_type

FORGE = FORGE_RES.parents[1]
OUTPUT = REPO / "src" / "effects" / "domain" / "forge_vocabularies.py"
TYPE_LISTS = Path("forge-gui/res/lists/TypeLists.txt")
COUNTER_ENUM = Path("forge-game/src/main/java/forge/game/card/CounterEnumType.java")
KEYWORD_COUNTERS = Path("forge-game/src/main/java/forge/game/card/CounterKeywordType.java")

_ENUM_ENTRY = re.compile(r"^\s+([A-Z][A-Z0-9_]*)\((.*)\),?\s*$", re.M)
_LEADING_STRINGS = re.compile(r'^\s*"([^"]*)"\s*,\s*"([^"]*)"')
_KEYWORD_COUNTER_LIST = re.compile(r"keywordCounter\s*=\s*ImmutableList\.of\((.*?)\);", re.S)


def subtype_spelling(name: str) -> str:
    """A subtype in the input's spelling: lower case, spaces as underscores."""
    return name.strip().lower().replace(" ", "_")


def subtypes(type_lists: str) -> tuple[str, ...]:
    """Every subtype ``TypeLists.txt`` names, in file order, each once: a line is
    ``Singular`` or ``Singular:Plural``, and every section lists subtypes, the basic
    land types' included."""
    found: dict[str, None] = {}
    for raw in type_lists.splitlines():
        line = raw.strip()
        if not line or (line.startswith("[") and line.endswith("]")):
            continue
        found.setdefault(subtype_spelling(line.split(":", 1)[0]), None)
    return tuple(found)


def named_counters(counter_enum: str) -> tuple[str, ...]:
    """Every counter type of ``CounterEnumType``, as a record spells it, in enum order.

    A constant's name is its first argument where two strings lead its arguments
    (``M1M1("-1/-1", "-1/-1", ...)``), else the constant's own name, which Forge
    capitalizes; the record upper-cases either.
    """
    found: dict[str, None] = {}
    for constant, arguments in _ENUM_ENTRY.findall(counter_enum):
        named = _LEADING_STRINGS.match(arguments)
        name = named.group(1) if named else constant
        found.setdefault(normalize_counter_type(name.upper()), None)
    return tuple(found)


def keyword_counters(counter_keyword_type: str) -> tuple[str, ...]:
    """The keywords a counter can grant (rule 122.1b), in the package's spelling."""
    match = _KEYWORD_COUNTER_LIST.search(counter_keyword_type)
    if match is None:
        raise ValueError("CounterKeywordType.java lists no keywordCounter")
    return tuple(normalize_keyword(name) for name in re.findall(r'"([^"]*)"', match.group(1)))


def scan(forge: Path) -> dict[str, tuple[str, ...]]:
    def read(path: Path) -> str:
        return (forge / path).read_text(encoding="utf-8", errors="replace")

    return {
        "SUBTYPES": subtypes(read(TYPE_LISTS)),
        "NAMED_COUNTERS": named_counters(read(COUNTER_ENUM)),
        "KEYWORD_COUNTERS": keyword_counters(read(KEYWORD_COUNTERS)),
    }


_COMMENTS = {
    "SUBTYPES": "Subtypes, in ``TypeLists.txt`` order, in the input's spelling.",
    "NAMED_COUNTERS": "Counter types, in ``CounterEnumType`` order, as a record spells them.",
    "KEYWORD_COUNTERS": "The keywords a counter grants (rule 122.1b), in the package's spelling.",
}


def render(lists: dict[str, tuple[str, ...]]) -> str:
    """The checked-in module's text."""
    lines = [
        '"""Forge\'s closed lists the effect head\'s input reads: generated, do not edit.',
        "",
        "Written by ``python scripts/scan_forge_vocabularies.py`` from Forge's type lists and",
        "counter enums, in Forge's own order, so a Forge merge that appends to a list keeps",
        "every earlier row. Rerun it after a Forge merge; its test fails when the module",
        "differs from a fresh scan.",
        '"""',
    ]
    for name, values in lists.items():
        lines += ["", f"#: {_COMMENTS[name]}", f"{name}: tuple[str, ...] = ("]
        lines += [f"    {json.dumps(value, ensure_ascii=False)}," for value in values]
        lines.append(")")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forge", type=Path, default=FORGE,
                        help="the Forge checkout to scan (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="the module to write (default: %(default)s)")
    args = parser.parse_args(argv)
    lists = scan(args.forge)
    args.output.write_text(render(lists), encoding="utf-8", newline="\n")
    print(", ".join(f"{len(values)} {name}" for name, values in lists.items())
          + f"; wrote {args.output}")


if __name__ == "__main__":
    main()
