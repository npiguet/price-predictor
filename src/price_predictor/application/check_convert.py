"""Check converted card files against Oracle text from Forge scripts."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Lines in converted output that are metadata, not ability text
_HEADER_KEYS = frozenset({
    "name", "mana cost", "types", "power toughness",
    "loyalty", "defense", "colors", "layout",
    # "text" is intentionally excluded: text: values represent oracle-relevant card content
    # (casting restrictions, conspiracy abilities, etc.) and must be counted as ability lines.
})

_REMINDER_TEXT = re.compile(r"\s*\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9{} ]+")
# Oracle spells out the additional-cost preamble; our converter strips it and uses the
# "additional cost:" key instead.  Remove it from oracle text before comparison.
_ADDITIONAL_COST_PREFIX = re.compile(
    r"^as an additional cost to cast this spell[,\s]+"
)
# Class/Talent cards have "{cost}: Level N" lines in oracle that the converter drops.
_CLASS_LEVEL_LINE = re.compile(r"^(\{[^}]+\})+:\s*Level\s+\d+$", re.IGNORECASE)
# text: lines that are purely a [Developer's note: …] bracket — Forge-internal metadata.
_DEVELOPER_NOTE = re.compile(r"^\[developer's note:[^\]]*\]$", re.IGNORECASE)
# Oracle: "Protection from creatures/artifacts/…" (with "from" + plural type)
# Converter internal format: "protection:creature" → normalised "protection creature"
# Two-step: first strip plural 's' from "protection from Xs", then strip bare "from".
_PROTECTION_PLURAL = re.compile(r"\bprotection from (\w+)s\b")
_PROTECTION_FROM = re.compile(r"\bprotection from\b")

# Words that indicate a line fragment is a sentence clause, not a bare keyword.
# Used by _split_keyword_line to avoid splitting non-keyword comma lists.
_KEYWORD_STOP_WORDS = frozenset({
    "a", "an", "the", "you", "your", "it", "its", "they", "their", "them",
    "then", "if", "when", "whenever", "until", "unless", "instead", "where",
    "while", "but", "as", "at", "in", "on", "into", "onto",
    "each", "all", "any", "target", "that", "this", "these", "those",
    "with", "for", "to", "of", "by", "do", "not", "no", "may", "can",
})

# Oracle uses portmanteau landwalk names ("swampwalk") but our converter outputs
# the split form ("landwalk swamp"). Normalise oracle text to the split form so
# both sides compare equal.  Order matters: longer phrases before shorter ones.
_LANDWALK_MAP: list[tuple[str, str]] = [
    ("legendary landwalk", "landwalk legendary land"),
    ("nonbasic landwalk",  "landwalk nonbasic land"),
    ("snow landwalk",      "landwalk snow land"),
    ("swampwalk",          "landwalk swamp"),
    ("forestwalk",         "landwalk forest"),
    ("islandwalk",         "landwalk island"),
    ("mountainwalk",       "landwalk mountain"),
    ("plainswalk",         "landwalk plains"),
    ("desertwalk",         "landwalk desert"),
]

# Cards whose Oracle text uses shorthand card-name references (unofficial sets that don't
# follow standard MTG conventions) rather than rules text. Our converter outputs the full
# ability text, which is correct; the oracle mismatch is unfixable.
_SKIP_ORACLE_SHORTHAND: frozenset[str] = frozenset({
    "g/growth_charm.txt",
    "i/innistrad_charm.txt",
    "k/kamigawa_charm.txt",
    "t/tarkir_charm.txt",
    "t/theros_charm.txt",
    "u/ulgrotha_charm.txt",
})

# Cards where Oracle groups multiple protection/hexproof keywords (or same-effect static
# abilities for different creature types) into a single compound line, but the converter
# correctly emits each Forge keyword as its own separate static ability.  The word-bag
# comparison flags these due to the extra "from"/"and" conjunction words in the oracle
# compound line.  The converter behaviour is intentional and correct.
_SKIP_ORACLE_AGGREGATION: frozenset[str] = frozenset({
    "c/caterwauling_boggart.txt",   # two menace statics (Goblin / Elemental) → one oracle line
    "e/elite_inquisitor.txt",       # three protection keywords → one oracle compound line
    "j/jaheira_harper_emissary.txt",# two hexproof keywords → one oracle compound line
    "o/oversoul_of_dusk.txt",       # three color-protection keywords → one oracle compound line
})

# Cards where the Forge script has no SpellDescription for the main spell effect —
# the converter correctly emits the additional cost but cannot produce a spell-effect line.
_SKIP_MISSING_DESCRIPTION: frozenset[str] = frozenset({
    "c/crashing_wave.txt",  # A:SP$ Tap + DBPutCounter SVars have no description
})

# Cards whose Forge implementation uses unofficial errata that diverges fundamentally
# from the printed oracle text; mismatch is not fixable without altering the oracle.
_SKIP_UNOFFICIAL_ERRATA: frozenset[str] = frozenset({
    "n/1996_world_champion.txt",  # EDH Silver errata: ETB choose-opponent + emblem not in oracle
})

_SKIP_FILES: frozenset[str] = (
    _SKIP_ORACLE_SHORTHAND
    | _SKIP_ORACLE_AGGREGATION
    | _SKIP_MISSING_DESCRIPTION
    | _SKIP_UNOFFICIAL_ERRATA
)

# Mapping from basic land subtypes to their intrinsic mana ability text
_LAND_TYPE_MANA: dict[str, str] = {
    "Plains": "{W}",
    "Island": "{U}",
    "Swamp": "{B}",
    "Mountain": "{R}",
    "Forest": "{G}",
}


@dataclass
class CardCheckResult:
    """Result of checking one converted card against its Oracle text."""

    filename: str
    card_name: str
    similarity: float
    oracle_lines: int
    converted_lines: int
    duplicate_lines: list[str]
    empty_lines: bool
    has_oracle: bool


def _is_keyword_token(token: str) -> bool:
    """Return True if token looks like a standalone MTG keyword phrase."""
    token = token.strip().rstrip(".").strip()
    if not token:
        return False
    words = token.lower().split()
    if len(words) > 6:
        return False
    return not any(w in _KEYWORD_STOP_WORDS for w in words)


def _split_keyword_line(line: str) -> list[str]:
    """If line is a comma- or semicolon-separated list of MTG keywords, return them split.

    Oracle text like "Flying, reach, trample." or "Trample; banding" is one line but
    the converter emits separate static: lines.  Splitting here lets line counts match.
    Returns [line] unchanged if the line doesn't look like a keyword list.
    """
    # Normalise semicolons to commas then split
    sep = "," if "," in line else (";" if ";" in line else None)
    if sep is None:
        return [line]
    parts = [p.strip() for p in line.split(sep)]
    if len(parts) < 2:
        return [line]
    # Strip trailing period from the last part only
    parts[-1] = parts[-1].rstrip(".")
    if all(_is_keyword_token(p) for p in parts):
        return [p for p in parts if p]
    return [line]


def _normalize(text: str, card_name: str | None = None) -> str:
    """Normalize text for comparison: lowercase, strip reminder text,
    replace card name, collapse whitespace/punctuation."""
    text = text.lower()
    text = text.replace("nickname", "cardname")
    # Strip reminder text (parenthesized)
    text = _REMINDER_TEXT.sub("", text)
    # Strip "as an additional cost to cast this spell, " prefix: oracle includes it but
    # our converter drops it (using the "additional cost:" key instead).
    text = _ADDITIONAL_COST_PREFIX.sub("", text)
    # Normalise landwalk portmanteaus: oracle says "swampwalk", converter outputs
    # "landwalk swamp". Map oracle form to converter form before further processing.
    for oracle_form, converted_form in _LANDWALK_MAP:
        text = text.replace(oracle_form, converted_form)
    # Normalise "protection from Xs" → "protection X" (oracle plural + "from" vs
    # converter singular internal format "protection:X" → "protection X").
    text = _PROTECTION_PLURAL.sub(r"protection \1", text)
    text = _PROTECTION_FROM.sub("protection", text)
    # Replace card name with placeholder
    if card_name:
        text = text.replace(card_name.lower(), "cardname")
    # Strip punctuation (keep braces for mana symbols)
    text = _NON_ALNUM.sub(" ", text)
    # Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _build_land_mana(types_line: str) -> str | None:
    """Build implicit mana ability text from basic land subtypes, or None."""
    symbols = [s for t, s in _LAND_TYPE_MANA.items() if t in types_line]
    if not symbols:
        return None
    if len(symbols) == 1:
        return f"{{T}}: Add {symbols[0]}."
    return "{T}: Add " + ", ".join(symbols[:-1]) + " or " + symbols[-1] + "."


def _strip_reminder(text: str) -> str:
    """Strip reminder text (parenthesized) from a string."""
    return _REMINDER_TEXT.sub("", text).strip()


def _extract_oracle(forge_text: str) -> tuple[str | None, str | None]:
    """Extract Oracle text and card name from a Forge script string.

    Strips reminder text from Oracle lines and appends implicit mana
    abilities for lands with basic land subtypes.
    """
    card_name = None
    oracle = None
    types_line = None
    for line in forge_text.splitlines():
        line = line.strip()
        if line == "ALTERNATE":
            break  # Only compare front face; back-face oracle is a separate card face
        if line.startswith("Name:") and card_name is None:
            card_name = line[5:].strip()
        elif line.startswith("Types:") and types_line is None:
            types_line = line[6:].strip()
        elif line.startswith("Oracle:") and oracle is None:
            oracle = line[7:].strip()
    if oracle:
        oracle = oracle.replace("\\n", "\n")
        # Strip reminder text from each oracle line
        oracle_lines = [_strip_reminder(ln) for ln in oracle.split("\n")]
        oracle_lines = [ln for ln in oracle_lines if ln]
        # Drop Class/Talent "{cost}: Level N" lines — converter omits them
        oracle_lines = [ln for ln in oracle_lines if not _CLASS_LEVEL_LINE.match(ln)]
        # Replace card name and NICKNAME with CARDNAME before splitting, so card names
        # containing commas (e.g., "Silvos, Rogue Elemental") do not cause incorrect splits.
        if card_name:
            oracle_lines = [
                ln.replace(card_name, "CARDNAME").replace("NICKNAME", "CARDNAME")
                for ln in oracle_lines
            ]
        # Split comma-separated keyword-only lines so line counts match the converter
        expanded: list[str] = []
        for ln in oracle_lines:
            expanded.extend(_split_keyword_line(ln))
        oracle_lines = expanded
        # Append implicit land mana ability if applicable
        if types_line:
            land_mana = _build_land_mana(types_line)
            if land_mana:
                oracle_lines.append(land_mana)
        oracle = "\n".join(oracle_lines) if oracle_lines else None
    elif types_line:
        # No oracle text, but may have implicit land mana
        land_mana = _build_land_mana(types_line)
        if land_mana:
            oracle = land_mana
    return card_name, oracle


def _extract_ability_text(converted_text: str) -> tuple[list[str], list[str]]:
    """Extract ability description lines from a converted output file.

    Returns (ability_lines, duplicate_lines).
    Only processes the first face (stops at first ALTERNATE line).
    """
    lines: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []

    for raw_line in converted_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        if raw_line == "ALTERNATE":
            break

        # Split on first ':'
        colon = raw_line.find(":")
        if colon < 0:
            continue
        key = raw_line[:colon].strip()
        # Strip action number suffix like "spell[1]" -> "spell"
        base_key = re.sub(r"\[\d+\]$", "", key)
        if base_key in _HEADER_KEYS:
            continue

        value = raw_line[colon + 1:].strip()
        # Skip text: lines that are purely a [Developer's note: …] — Forge-internal metadata,
        # not oracle content; the Java converter should strip these, but guard here too.
        if base_key == "text" and _DEVELOPER_NOTE.match(value):
            continue
        # Class level lines include the upgrade cost (e.g. "{2}{R}: At the beginning...")
        # which oracle drops.  Strip the leading mana-cost prefix before comparison.
        if base_key == "level" and value:
            cost_stripped = re.sub(r"^(\{[^}]+\})+:\s*", "", value)
            if cost_stripped:
                value = cost_stripped
        lines.append(value)
        if value in seen:
            duplicates.append(value)
        seen.add(value)

    return lines, duplicates


def check_card(converted_text: str, forge_text: str) -> CardCheckResult:
    """Check a single converted card against its Forge source."""
    card_name, oracle = _extract_oracle(forge_text)
    ability_lines, duplicates = _extract_ability_text(converted_text)

    # Get the name from converted output for display
    display_name = card_name or "unknown"
    for line in converted_text.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            display_name = line[5:].strip()
            break

    has_oracle = oracle is not None and oracle.strip() != ""
    empty_lines = any(not ln.strip() for ln in ability_lines)

    if not has_oracle or not ability_lines:
        # Can't compute similarity without both sides
        return CardCheckResult(
            filename="",
            card_name=display_name,
            similarity=1.0 if not has_oracle else 0.0,
            oracle_lines=len(oracle.split("\n")) if oracle else 0,
            converted_lines=len(ability_lines),
            duplicate_lines=duplicates,
            empty_lines=empty_lines,
            has_oracle=has_oracle,
        )

    # Compare using word-bag (multiset) Jaccard similarity.
    # This is order-independent, line-grouping-independent, and robust to
    # cases where oracle splits text differently than the converter (e.g.
    # one oracle paragraph vs several sub-ability lines in the output).
    oracle_words: Counter[str] = Counter()
    for ln in oracle.split("\n"):
        oracle_words.update(_normalize(ln, card_name).split())

    converted_words: Counter[str] = Counter()
    for ln in ability_lines:
        converted_words.update(_normalize(ln, card_name).split())

    intersection = sum((oracle_words & converted_words).values())
    union = sum((oracle_words | converted_words).values())
    similarity = intersection / union if union > 0 else 1.0

    return CardCheckResult(
        filename="",
        card_name=display_name,
        similarity=similarity,
        oracle_lines=len(oracle.split("\n")),
        converted_lines=len(ability_lines),
        duplicate_lines=duplicates,
        empty_lines=empty_lines,
        has_oracle=has_oracle,
    )


def check_all(
    output_dir: Path,
    cards_dir: Path,
    *,
    threshold: float = 0.5,
) -> list[CardCheckResult]:
    """Check all converted files against their Forge sources.

    Returns results sorted by similarity ascending (worst first).
    Only includes cards below the threshold or with structural issues.
    """
    results: list[CardCheckResult] = []

    for converted_path in sorted(output_dir.rglob("*.txt")):
        rel = converted_path.relative_to(output_dir)
        forge_path = cards_dir / rel

        if not forge_path.exists():
            continue

        # Skip cards with known-unfixable oracle mismatches
        rel_posix = rel.as_posix()
        if rel_posix in _SKIP_FILES:
            continue

        try:
            converted_text = converted_path.read_text(encoding="utf-8", errors="replace")
            forge_text = forge_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        result = check_card(converted_text, forge_text)
        result.filename = str(rel)

        has_issues = (
            result.similarity < threshold
            # Only flag duplicate lines when similarity is not perfect: if oracle and
            # converter both emit N identical lines (e.g. Bounty of Might), the
            # word-bag similarity is 100% and the duplicates are intentional.
            or (result.duplicate_lines and result.similarity < 1.0)
            or result.empty_lines
            or (result.has_oracle and result.converted_lines == 0)
        )
        if has_issues:
            results.append(result)

    results.sort(key=lambda r: r.similarity)
    return results


def format_report(results: list[CardCheckResult], *, limit: int = 0) -> str:
    """Format check results as a human-readable report."""
    if not results:
        return "All cards passed checks."

    lines: list[str] = []
    shown = results[:limit] if limit > 0 else results
    for r in shown:
        flags: list[str] = []
        if r.duplicate_lines:
            flags.append(f"duplicates={len(r.duplicate_lines)}")
        if r.empty_lines:
            flags.append("has_empty_lines")
        if r.has_oracle and r.converted_lines == 0:
            flags.append("no_ability_lines")

        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"{r.similarity:.2%}  {r.card_name:<40s}  "
            f"oracle={r.oracle_lines} converted={r.converted_lines}"
            f"{flag_str}  ({r.filename})"
        )

    header = f"Found {len(results)} cards with issues"
    if limit > 0 and len(results) > limit:
        header += f" (showing top {limit})"
    header += ":\n"

    return header + "\n".join(lines)
