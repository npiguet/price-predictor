"""Write ``src/effects/domain/kept_keyword_flags.py``: the keywords that keep a flag (M5).

A keyword keeps its flag where a selector in Forge's card scripts names it, or where
Forge's engine checks it by name; protection and split second keep theirs whatever
the scan finds (root spec section 9.6, FR-044a). Every other keyword is read through its
line alone.

Two scans, both read-only on the sibling ``../forge`` checkout:

- **Card scripts** (``forge-gui/res/cardsfolder``): a selector property naming a
  keyword, ``with<Keyword>``, ``without<Keyword>`` or ``hasKeyword<Keyword>``, as in
  ``ValidTgts$ Creature.withFlying`` or ``Count$Valid Creature.withoutFirst Strike``.
- **Engine code** (``forge-game/src/main/java``): a card queried for one keyword,
  ``hasKeyword(Keyword.TRAMPLE)``, ``getKeywords(Keyword.LANDWALK)``,
  ``getKeywordMagnitude(Keyword.TOXIC)``, ``hasKeyword("Trample:Planeswalker")``; a
  keyword instance compared with one, ``inst.getKeyword() == Keyword.WARD``; and a
  selector written into the engine's own strings, such as the static flying generates,
  ``ValidBlocker$ Creature.withoutFlying+withoutReach``.

The keyword names come from Forge's ``Keyword`` enum, so a selector is matched against
the display names Forge itself parses (``First Strike``, ``Split second``). A selector
names the longest display name it starts with that ends where a word ends:
``withFlashback`` names flashback, not flash. Names are written in this package's
spelling (``effects.domain.keyword_names.normalize_keyword``: ``first_strike``).

Usage: python scripts/scan_keyword_flags.py [--forge DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from _lib.paths import FORGE_RES, REPO

from effects.domain.keyword_names import normalize_keyword

FORGE = FORGE_RES.parents[1]
OUTPUT = REPO / "src" / "effects" / "domain" / "kept_keyword_flags.py"
KEYWORD_ENUM = Path("forge-game/src/main/java/forge/game/keyword/Keyword.java")
CARD_SCRIPTS = Path("forge-gui/res/cardsfolder")
ENGINE = Path("forge-game/src/main/java")

#: Kept whatever the scan finds (root spec section 9.6).
ALWAYS_KEPT = ("protection", "split_second")

_ENUM_ENTRY = re.compile(r'^\s+([A-Z][A-Z0-9_]*)\("([^"]*)",', re.M)
#: A selector property naming a keyword: what follows the prefix, up to a selector
#: separator. Display names hold spaces, apostrophes and hyphens.
_SELECTOR = re.compile(r"(?<![A-Za-z])(?:without|with|hasKeyword)([A-Z][A-Za-z' -]*)")
#: A card asked about one keyword, by the enum or by its text.
_QUERIES = ("hasKeyword", "hasAnyKeyword", "getKeywords", "getAmountOfKeyword",
            "getKeywordMagnitude", "hasStartOfKeyword", "hasStartOfUnHiddenKeyword",
            "hasIntrinsicKeyword")
#: A call on another object or on ``this`` (``Card`` asks itself ``hasKeyword(...)``).
_CALL = rf"(?<![\w])(?:{'|'.join(_QUERIES)})\(\s*"
_QUERY_ENUM = re.compile(_CALL + r"Keyword\.([A-Z][A-Z0-9_]*)\b")
_QUERY_TEXT = re.compile(_CALL + r'"([^"\\]*)"')
_COMPARED = re.compile(
    r"getKeyword\(\)\s*(?:==|!=|\.equals\()\s*Keyword\.([A-Z][A-Z0-9_]*)\b"
    r"|Keyword\.([A-Z][A-Z0-9_]*)\s*(?:==|!=)\s*\w+(?:\.\w+\(\))*\.getKeyword\(\)")
_STRING = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
#: Engine sources that decide no outcome, beside the ``*View.java`` the UI reads.
_NOT_RULES = frozenset({"AchievementTracker.java"})
#: Card-script lines that hold prose or a name, never a selector.
_PROSE_LINE = re.compile(r"^(?:Name|Oracle|FlavorName|AltName):")


def keyword_names(forge: Path) -> dict[str, str]:
    """``enum constant -> display name`` from Forge's ``Keyword`` enum."""
    text = (forge / KEYWORD_ENUM).read_text(encoding="utf-8")
    return {name: display for name, display in _ENUM_ENTRY.findall(text)
            if name != "UNDEFINED" and display}


def named_keyword(text: str, displays: list[str]) -> str | None:
    """The longest display name ``text`` starts with that ends at a word's end."""
    lowered = text.lower()
    for display in displays:  # longest first
        if lowered.startswith(display.lower()):
            rest = text[len(display):]
            if not rest or not rest[0].isalpha():
                return display
    return None


def selector_keywords(text: str, displays: list[str]) -> Counter:
    """The keywords the selectors in ``text`` name, with how often."""
    found = Counter()
    for match in _SELECTOR.finditer(text):
        display = named_keyword(match.group(1), displays)
        if display is not None:
            found[display] += 1
    return found


def card_script_keywords(forge: Path, displays: list[str]) -> Counter:
    """Keyword -> selectors naming it over every card script."""
    found = Counter()
    for path in sorted((forge / CARD_SCRIPTS).rglob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not _PROSE_LINE.match(line):
                found += selector_keywords(line, displays)
    return found


def engine_keywords(source: str, names: dict[str, str], displays: list[str]) -> Counter:
    """Keyword -> checks naming it in one Java source file."""
    found = Counter()
    for name in _QUERY_ENUM.findall(source):
        if name in names:
            found[names[name]] += 1
    for text in _QUERY_TEXT.findall(source):
        display = named_keyword(text, displays)
        if display is not None:
            found[display] += 1
    for pair in _COMPARED.findall(source):
        name = pair[0] or pair[1]
        if name in names:
            found[names[name]] += 1
    for literal in _STRING.findall(source):
        found += selector_keywords(literal, displays)
    return found


def rules_code(path: Path) -> bool:
    """Whether a source file plays the rules: the trackable views the UI reads
    (``CardView.java`` flags ward for display) and the achievement tracker decide
    no outcome, so a keyword they name keeps no flag for it."""
    return not path.name.endswith("View.java") and path.name not in _NOT_RULES


def engine_code_keywords(forge: Path, names: dict[str, str], displays: list[str]) -> Counter:
    """Keyword -> checks naming it over the engine's Java sources."""
    found = Counter()
    for path in sorted((forge / ENGINE).rglob("*.java")):
        if not rules_code(path):
            continue
        found += engine_keywords(path.read_text(encoding="utf-8", errors="replace"),
                                 names, displays)
    return found


def kept_flags(cards: Counter, engine: Counter) -> dict[str, tuple[int, int]]:
    """``keyword -> (card-script selectors, engine checks)``, in this package's
    spelling, sorted, with :data:`ALWAYS_KEPT` added."""
    kept: dict[str, list[int]] = {}
    for display in set(cards) | set(engine):
        counts = kept.setdefault(normalize_keyword(display), [0, 0])
        counts[0] += cards[display]
        counts[1] += engine[display]
    for name in ALWAYS_KEPT:
        kept.setdefault(name, [0, 0])
    return {name: (counts[0], counts[1]) for name, counts in sorted(kept.items())}


def render(kept: dict[str, tuple[int, int]]) -> str:
    """The checked-in module's text."""
    lines = [
        '"""The keywords that keep a flag (M5, FR-044a): generated, do not edit.',
        "",
        "Written by ``python scripts/scan_keyword_flags.py`` from Forge's card scripts and",
        "engine code. A keyword keeps its flag where a card-script selector or an engine check",
        "names it; protection and split second keep theirs whatever the scan finds. Every",
        "other keyword is read through its line alone. Each name's comment gives the",
        "card-script selectors and engine checks the scan found naming it.",
        '"""',
        "",
        "KEPT_KEYWORD_FLAGS: tuple[str, ...] = (",
    ]
    for name, (cards, engine) in kept.items():
        why = "always kept" if cards == engine == 0 else f"scripts {cards}, engine {engine}"
        lines.append(f'    "{name}",  # {why}')
    lines.append(")")
    return "\n".join(lines) + "\n"


def scan(forge: Path) -> dict[str, tuple[int, int]]:
    names = keyword_names(forge)
    displays = sorted(set(names.values()), key=len, reverse=True)
    return kept_flags(card_script_keywords(forge, displays),
                      engine_code_keywords(forge, names, displays))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--forge", type=Path, default=FORGE,
                        help="the Forge checkout to scan (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="the module to write (default: %(default)s)")
    args = parser.parse_args(argv)
    kept = scan(args.forge)
    args.output.write_text(render(kept), encoding="utf-8", newline="\n")
    print(f"{len(kept)} keywords keep a flag; wrote {args.output}")


if __name__ == "__main__":
    main()
