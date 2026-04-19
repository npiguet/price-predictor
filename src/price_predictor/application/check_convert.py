"""Check converted card files against Oracle text from Forge scripts."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from price_predictor.domain.card_text import ForgeCardScript

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

class SkipReason(Enum):
    """Why a converted card is intentionally excluded from the similarity check."""

    ORACLE_SHORTHAND = (
        "Oracle text uses shorthand card-name references rather than rules text "
        "(unofficial sets that don't follow standard MTG conventions)"
    )
    ORACLE_AGGREGATION = (
        "Oracle groups multiple protection/hexproof keywords into one compound "
        "line; the converter correctly emits each Forge keyword separately"
    )
    MISSING_DESCRIPTION = (
        "Forge script has no SpellDescription for the main spell effect"
    )
    UNOFFICIAL_ERRATA = (
        "Forge implementation uses errata that diverges fundamentally from oracle"
    )


_SKIPPED_CARDS: dict[str, SkipReason] = {
    "g/growth_charm.txt": SkipReason.ORACLE_SHORTHAND,
    "i/innistrad_charm.txt": SkipReason.ORACLE_SHORTHAND,
    "k/kamigawa_charm.txt": SkipReason.ORACLE_SHORTHAND,
    "t/tarkir_charm.txt": SkipReason.ORACLE_SHORTHAND,
    "t/theros_charm.txt": SkipReason.ORACLE_SHORTHAND,
    "u/ulgrotha_charm.txt": SkipReason.ORACLE_SHORTHAND,
    "c/caterwauling_boggart.txt": SkipReason.ORACLE_AGGREGATION,
    "e/elite_inquisitor.txt": SkipReason.ORACLE_AGGREGATION,
    "j/jaheira_harper_emissary.txt": SkipReason.ORACLE_AGGREGATION,
    "o/oversoul_of_dusk.txt": SkipReason.ORACLE_AGGREGATION,
    "c/crashing_wave.txt": SkipReason.MISSING_DESCRIPTION,
    "n/1996_world_champion.txt": SkipReason.UNOFFICIAL_ERRATA,
}

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

    def has_issues(self, threshold: float) -> bool:
        """Return True if this card should appear in the issues report.

        Duplicate lines are only flagged when similarity is not perfect: if
        oracle and converter both emit N identical lines (e.g. Bounty of
        Might), the word-bag similarity is 100% and the duplicates are
        intentional.
        """
        return (
            self.similarity < threshold
            or (bool(self.duplicate_lines) and self.similarity < 1.0)
            or self.empty_lines
            or (self.has_oracle and self.converted_lines == 0)
        )


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


@dataclass(frozen=True)
class _NormalizationRule:
    """One step in the oracle-vs-converted text normalization pipeline.

    Each rule rewrites text the same way for oracle and converter input so the
    downstream word-bag comparison is fair. ``card_name`` is supplied so rules
    that need to mask the literal card name can do so before punctuation
    stripping shreds multi-word names.
    """

    name: str
    apply: Callable[[str, str | None], str]


def _replace_landwalk(text: str, _card_name: str | None) -> str:
    for oracle_form, converted_form in _LANDWALK_MAP:
        text = text.replace(oracle_form, converted_form)
    return text


def _replace_card_name(text: str, card_name: str | None) -> str:
    if not card_name:
        return text
    return text.replace(card_name.lower(), "cardname")


# Rules run top-to-bottom. Order matters: lowercase first so subsequent rules
# can be lowercase-only; reminder/prefix strips before token rewrites; card-name
# replacement before punctuation strip; whitespace collapse last.
_NORMALIZATION_RULES: tuple[_NormalizationRule, ...] = (
    _NormalizationRule("lowercase", lambda t, _: t.lower()),
    _NormalizationRule("nickname_alias", lambda t, _: t.replace("nickname", "cardname")),
    _NormalizationRule("strip_reminder_text", lambda t, _: _REMINDER_TEXT.sub("", t)),
    _NormalizationRule(
        "strip_additional_cost_prefix",
        lambda t, _: _ADDITIONAL_COST_PREFIX.sub("", t),
    ),
    _NormalizationRule("expand_landwalk_portmanteau", _replace_landwalk),
    _NormalizationRule(
        "protection_from_plural",
        lambda t, _: _PROTECTION_PLURAL.sub(r"protection \1", t),
    ),
    _NormalizationRule(
        "protection_from_bare",
        lambda t, _: _PROTECTION_FROM.sub("protection", t),
    ),
    _NormalizationRule("mask_card_name", _replace_card_name),
    _NormalizationRule("strip_punctuation", lambda t, _: _NON_ALNUM.sub(" ", t)),
    _NormalizationRule(
        "collapse_whitespace",
        lambda t, _: _WHITESPACE.sub(" ", t).strip(),
    ),
)


def _normalize(text: str, card_name: str | None = None) -> str:
    """Run text through the normalization pipeline."""
    for rule in _NORMALIZATION_RULES:
        text = rule.apply(text, card_name)
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


def _parse_forge_header(
    forge_text: str,
) -> tuple[str | None, str | None, str | None]:
    """Walk the front face of a Forge script and pull out (name, types, oracle)."""
    card_name = oracle = types_line = None
    for line in forge_text.splitlines():
        line = line.strip()
        if line == "ALTERNATE":
            break
        if line.startswith("Name:") and card_name is None:
            card_name = line[5:].strip()
        elif line.startswith("Types:") and types_line is None:
            types_line = line[6:].strip()
        elif line.startswith("Oracle:") and oracle is None:
            oracle = line[7:].strip()
    return card_name, types_line, oracle


def _strip_oracle_reminders(lines: list[str]) -> list[str]:
    """Strip parenthesized reminder text from each line and drop blanks."""
    return [ln for ln in (_strip_reminder(ln) for ln in lines) if ln]


def _drop_class_level_lines(lines: list[str]) -> list[str]:
    """Drop Class/Talent "{cost}: Level N" lines — the converter omits them."""
    return [ln for ln in lines if not _CLASS_LEVEL_LINE.match(ln)]


def _mask_card_name_in_lines(lines: list[str], card_name: str | None) -> list[str]:
    """Replace literal card name and NICKNAME with CARDNAME so name commas don't split.

    Done before keyword-list splitting so multi-word names containing commas
    (e.g. "Silvos, Rogue Elemental") never cause an incorrect split.
    """
    if not card_name:
        return lines
    return [
        ln.replace(card_name, "CARDNAME").replace("NICKNAME", "CARDNAME")
        for ln in lines
    ]


def _split_keyword_lines_in_list(lines: list[str]) -> list[str]:
    """Expand comma-separated keyword lines so counts match the converter's per-line output."""
    expanded: list[str] = []
    for ln in lines:
        expanded.extend(_split_keyword_line(ln))
    return expanded


def _maybe_append_land_mana(lines: list[str], types_line: str | None) -> list[str]:
    """Append the implicit basic-land mana ability when applicable."""
    if not types_line:
        return lines
    land_mana = _build_land_mana(types_line)
    if not land_mana:
        return lines
    return [*lines, land_mana]


def _extract_oracle(forge_text: str) -> tuple[str | None, str | None]:
    """Extract Oracle text and card name from a Forge script string.

    Strips reminder text from Oracle lines and appends implicit mana
    abilities for lands with basic land subtypes.
    """
    card_name, types_line, oracle = _parse_forge_header(forge_text)

    if oracle:
        raw_lines = oracle.replace("\\n", "\n").split("\n")
        lines = _strip_oracle_reminders(raw_lines)
        lines = _drop_class_level_lines(lines)
        lines = _mask_card_name_in_lines(lines, card_name)
        lines = _split_keyword_lines_in_list(lines)
        lines = _maybe_append_land_mana(lines, types_line)
        oracle = "\n".join(lines) if lines else None
    elif types_line:
        oracle = _build_land_mana(types_line)

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


def _word_bag_jaccard(
    oracle: str, ability_lines: list[str], card_name: str | None,
) -> float:
    """Multiset Jaccard similarity over normalized oracle vs converted lines."""
    oracle_words: Counter[str] = Counter()
    for ln in oracle.split("\n"):
        oracle_words.update(_normalize(ln, card_name).split())

    converted_words: Counter[str] = Counter()
    for ln in ability_lines:
        converted_words.update(_normalize(ln, card_name).split())

    intersection = sum((oracle_words & converted_words).values())
    union = sum((oracle_words | converted_words).values())
    return intersection / union if union > 0 else 1.0


def _display_name_from_converted(converted_text: str, fallback: str | None) -> str:
    for line in converted_text.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            return line[5:].strip()
    return fallback or "unknown"


def check_card(converted_text: str, forge_text: ForgeCardScript) -> CardCheckResult:
    """Check a single converted card against its Forge source."""
    card_name, oracle = _extract_oracle(forge_text)
    ability_lines, duplicates = _extract_ability_text(converted_text)

    display_name = _display_name_from_converted(converted_text, card_name)
    has_oracle = oracle is not None and oracle.strip() != ""
    empty_lines = any(not ln.strip() for ln in ability_lines)

    # Word-bag (multiset) Jaccard similarity is order-independent, robust to
    # different line groupings between oracle and converter, and degenerates
    # to 1.0 when there's nothing to compare on either side.
    if not has_oracle:
        similarity = 1.0
    elif not ability_lines:
        similarity = 0.0
    else:
        similarity = _word_bag_jaccard(oracle, ability_lines, card_name)

    return CardCheckResult(
        filename="",
        card_name=display_name,
        similarity=similarity,
        oracle_lines=len(oracle.split("\n")) if oracle else 0,
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

        if rel.as_posix() in _SKIPPED_CARDS:
            continue

        try:
            converted_text = converted_path.read_text(encoding="utf-8", errors="replace")
            forge_text = ForgeCardScript(
                forge_path.read_text(encoding="utf-8", errors="replace")
            )
        except OSError:
            continue

        result = check_card(converted_text, forge_text)
        result.filename = str(rel)

        if result.has_issues(threshold):
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
