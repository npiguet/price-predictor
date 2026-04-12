"""Parse converted card text format into Card domain entities."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from price_predictor.domain.card_taxonomy import KNOWN_CARD_TYPES, KNOWN_SUPERTYPES
from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost

logger = logging.getLogger(__name__)

# Lines whose values contribute to oracle_text and ability_count.
# Matches prefixes like spell[1]:, triggered:, activated:, planeswalker[2]:,
# replacement:, keyword[1]:, static:, etc.
_ABILITY_LINE_RE = re.compile(
    r"^(spell\[\d+\]|triggered|activated|planeswalker\[\d+\]"
    r"|replacement|keyword\[\d+\]|static|etb|ltb|mana\[\d+\]"
    r"|cost reduction|mode\[\d+\]|chapter \w+|level \S+):",
    re.IGNORECASE,
)

# Pattern for the {X}{Y}{Z} mana cost format.
_MANA_BRACE_RE = re.compile(r"\{([^}]+)\}")


def convert_mana_cost(raw: str) -> str:
    """Convert brace-delimited mana cost to space-separated Forge format.

    E.g. ``{2}{U}{U}`` becomes ``2 U U``.
    """
    shards = _MANA_BRACE_RE.findall(raw)
    return " ".join(shards)


def extract_mana_cost_line(text: str) -> str | None:
    """Return the raw value of the ``mana cost:`` line, or None when missing/empty."""
    for line in text.splitlines():
        if line.strip().lower().startswith("mana cost:"):
            return line.split(":", 1)[1].strip() or None
    return None


def _classify_types(types_str: str) -> tuple[list[str], list[str], list[str]]:
    """Split a converted-format types line into (supertypes, card_types, subtypes).

    The converted format uses lowercase tokens, so each token is title-cased
    before classification against the known sets.
    """
    tokens = types_str.split()
    supertypes: list[str] = []
    card_types: list[str] = []
    subtypes: list[str] = []
    for token in tokens:
        titled = token.title()
        if titled in KNOWN_SUPERTYPES:
            supertypes.append(titled)
        elif titled in KNOWN_CARD_TYPES:
            card_types.append(titled)
        else:
            subtypes.append(titled)
    return supertypes, card_types, subtypes


def parse_converted_text(text: str) -> Card:
    """Parse a single converted card text string into a Card entity.

    Raises ValueError if the text is empty or missing required fields
    (``name:`` or ``types:``).
    """
    if not text or not text.strip():
        raise ValueError("Card text is empty")

    fields: dict[str, str] = {}
    ability_lines: list[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Check for ability lines first — they contribute to oracle_text
        ability_match = _ABILITY_LINE_RE.match(line)
        if ability_match:
            # Value is everything after the matched prefix and colon
            colon_idx = line.index(":", ability_match.start())
            value = line[colon_idx + 1:].strip()
            if value:
                ability_lines.append(value)
            continue

        # Regular key:value extraction
        colon_idx = line.find(":")
        if colon_idx > 0:
            key = line[:colon_idx].strip().lower()
            value = line[colon_idx + 1:].strip()
            if key not in fields:
                fields[key] = value

    # --- Required fields ---
    name = fields.get("name", "").strip()
    if not name:
        raise ValueError("No 'name:' field found in converted card text")

    types_str = fields.get("types", "").strip()
    if not types_str:
        raise ValueError("No 'types:' field found in converted card text")

    supertypes, card_types, subtypes = _classify_types(types_str)
    if not card_types:
        raise ValueError(f"No recognized card types in types line: {types_str}")

    # --- Mana cost ---
    mana_cost_raw = fields.get("mana cost", "").strip()
    mana_cost: ManaCost | None = None
    if mana_cost_raw:
        forge_mana = convert_mana_cost(mana_cost_raw)
        mana_cost = ManaCost.parse(forge_mana)

    # --- Power / Toughness ---
    power: str | None = None
    toughness: str | None = None
    pt_raw = fields.get("power toughness", "").strip()
    if pt_raw and "/" in pt_raw:
        pt_parts = pt_raw.split("/", maxsplit=1)
        power = pt_parts[0].strip()
        toughness = pt_parts[1].strip()

    # --- Loyalty ---
    loyalty = fields.get("loyalty", "").strip() or None

    # --- Oracle text & ability count ---
    oracle_text: str | None = None
    ability_count = len(ability_lines)
    if ability_lines:
        oracle_text = "\n".join(ability_lines)

    return Card(
        name=name,
        types=card_types,
        supertypes=supertypes,
        subtypes=subtypes,
        mana_cost=mana_cost,
        oracle_text=oracle_text,
        keywords=[],
        power=power,
        toughness=toughness,
        loyalty=loyalty,
        layout="normal",
        ability_count=ability_count,
    )


def parse_converted_file(path: Path) -> Card | None:
    """Parse a single converted card text file into a Card entity.

    Returns None if the file cannot be read or parsed.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    try:
        return parse_converted_text(text)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return None


def parse_converted_cards(
    cards_dir: Path,
) -> tuple[list[Card], list[str]]:
    """Parse all converted card .txt files in a directory tree.

    The directory is expected to contain sub-folders organised by first letter.
    Returns ``(cards, errors)`` where *errors* is a list of error messages.

    Raises ValueError if *cards_dir* does not exist or contains no .txt files.
    """
    if not cards_dir.exists():
        raise ValueError(f"Cards directory does not exist: {cards_dir}")

    txt_files = sorted(cards_dir.rglob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in: {cards_dir}")

    cards: list[Card] = []
    errors: list[str] = []

    logger.info("Parsing converted card texts...")
    processed = 0
    for txt_file in txt_files:
        card = parse_converted_file(txt_file)
        if card is not None:
            cards.append(card)
        else:
            errors.append(f"Failed to parse: {txt_file}")
        processed += 1
        if processed % 5000 == 0:
            logger.info("Parsing converted cards... %d parsed", processed)

    logger.info("Parsed %d cards total, %d errors", len(cards), len(errors))
    return cards, errors
