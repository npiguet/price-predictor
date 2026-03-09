"""Check converted card files against Oracle text from Forge scripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

# Lines in converted output that are metadata, not ability text
_HEADER_KEYS = frozenset({
    "name", "mana cost", "types", "power toughness",
    "loyalty", "defense", "colors", "layout", "text",
})

_REMINDER_TEXT = re.compile(r"\s*\([^)]*\)")
_WHITESPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9{} ]+")


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


def _normalize(text: str, card_name: str | None = None) -> str:
    """Normalize text for comparison: lowercase, strip reminder text,
    replace card name, collapse whitespace/punctuation."""
    text = text.lower()
    # Strip reminder text (parenthesized)
    text = _REMINDER_TEXT.sub("", text)
    # Replace card name with placeholder
    if card_name:
        text = text.replace(card_name.lower(), "cardname")
    # Strip punctuation (keep braces for mana symbols)
    text = _NON_ALNUM.sub(" ", text)
    # Collapse whitespace
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _extract_oracle(forge_text: str) -> tuple[str | None, str | None]:
    """Extract Oracle text and card name from a Forge script string."""
    card_name = None
    oracle = None
    for line in forge_text.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            card_name = line[5:].strip()
        elif line.startswith("Oracle:"):
            oracle = line[7:].strip()
    if oracle:
        oracle = oracle.replace("\\n", "\n")
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

    # Normalize and compare
    norm_oracle = _normalize(oracle, card_name)
    norm_converted = _normalize("\n".join(ability_lines), card_name)

    similarity = SequenceMatcher(
        None, norm_oracle, norm_converted,
    ).ratio()

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

        try:
            converted_text = converted_path.read_text(encoding="utf-8", errors="replace")
            forge_text = forge_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        result = check_card(converted_text, forge_text)
        result.filename = str(rel)

        has_issues = (
            result.similarity < threshold
            or result.duplicate_lines
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
