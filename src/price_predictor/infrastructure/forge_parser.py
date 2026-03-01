"""Parse Forge card scripts into Card domain entities."""

from __future__ import annotations

import logging
from pathlib import Path

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost

logger = logging.getLogger(__name__)

KNOWN_SUPERTYPES = frozenset({"Legendary", "Basic", "Snow", "World", "Ongoing", "Host"})
KNOWN_CARD_TYPES = frozenset({
    "Creature", "Instant", "Sorcery", "Enchantment", "Artifact",
    "Planeswalker", "Land", "Battle", "Tribal", "Kindred",
    "Scheme", "Plane", "Conspiracy", "Vanguard", "Phenomenon",
})

ALTERNATE_MODE_TO_LAYOUT = {
    "DoubleFaced": "doublefaced",
    "Split": "split",
    "Adventure": "adventure",
    "Modal": "modal",
    "Flip": "flip",
}


def _classify_types(types_str: str) -> tuple[list[str], list[str], list[str]]:
    """Split a Forge Types line into (supertypes, card_types, subtypes)."""
    tokens = types_str.split()
    supertypes: list[str] = []
    card_types: list[str] = []
    subtypes: list[str] = []
    for token in tokens:
        if token in KNOWN_SUPERTYPES:
            supertypes.append(token)
        elif token in KNOWN_CARD_TYPES:
            card_types.append(token)
        else:
            subtypes.append(token)
    return supertypes, card_types, subtypes


def parse_forge_text(text: str) -> Card:
    """Parse a Forge card script string into a Card entity.

    Raises ValueError if the text is empty or cannot be parsed into a valid Card.
    """
    if not text or not text.strip():
        raise ValueError("Card script text is empty")

    # Split on ALTERNATE section — use only front face
    parts = text.split("\nALTERNATE\n", maxsplit=1)
    front_text = parts[0]

    fields: dict[str, str] = {}
    keywords: list[str] = []
    ability_count = 0
    alternate_mode: str | None = None

    for line in front_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Count abilities
        if line.startswith("A:") or line.startswith("T:") or line.startswith("S:"):
            ability_count += 1

        # Extract keywords (K: lines)
        if line.startswith("K:"):
            kw_text = line[2:].strip()
            # Strip parameters after first ':'
            kw_name = kw_text.split(":")[0].strip()
            if kw_name:
                keywords.append(kw_name)
            continue

        # Key-value extraction
        colon_idx = line.find(":")
        if colon_idx > 0:
            key = line[:colon_idx].strip()
            value = line[colon_idx + 1:].strip()
            if key == "AlternateMode":
                alternate_mode = value
            elif key not in fields:
                fields[key] = value

    name = fields.get("Name", "").strip()
    if not name:
        raise ValueError("No Name field found in card script")

    types_str = fields.get("Types", "")
    if not types_str:
        raise ValueError("No Types field found in card script")

    supertypes, card_types, subtypes = _classify_types(types_str)
    if not card_types:
        raise ValueError(f"No recognized card types in Types line: {types_str}")

    # Mana cost
    mana_cost_raw = fields.get("ManaCost", "").strip()
    mana_cost = ManaCost.parse(mana_cost_raw) if mana_cost_raw else None

    # Power/Toughness
    power: str | None = None
    toughness: str | None = None
    pt_raw = fields.get("PT", "").strip()
    if pt_raw and "/" in pt_raw:
        pt_parts = pt_raw.split("/", maxsplit=1)
        power = pt_parts[0].strip()
        toughness = pt_parts[1].strip()

    # Loyalty
    loyalty = fields.get("Loyalty", "").strip() or None

    # Oracle text
    oracle_text = fields.get("Oracle", "").strip() or None
    if oracle_text:
        oracle_text = oracle_text.replace("\\n", "\n")

    # Layout
    layout = "normal"
    if alternate_mode and alternate_mode in ALTERNATE_MODE_TO_LAYOUT:
        layout = ALTERNATE_MODE_TO_LAYOUT[alternate_mode]

    return Card(
        name=name,
        types=card_types,
        supertypes=supertypes,
        subtypes=subtypes,
        mana_cost=mana_cost,
        oracle_text=oracle_text,
        keywords=keywords,
        power=power,
        toughness=toughness,
        loyalty=loyalty,
        layout=layout,
        ability_count=ability_count,
    )


def parse_forge_file(path: Path) -> Card | None:
    """Parse a single Forge card script file into a Card entity.

    Returns None if the file cannot be parsed.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None

    try:
        return parse_forge_text(text)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return None


def parse_forge_cards(
    cards_dir: Path,
) -> tuple[list[Card], list[str]]:
    """Parse all Forge card scripts in a directory tree.

    Returns (cards, errors) where errors is a list of error messages.
    """
    cards: list[Card] = []
    errors: list[str] = []

    if not cards_dir.exists():
        return cards, errors

    logger.info("Parsing Forge card scripts...")
    processed = 0
    for txt_file in sorted(cards_dir.rglob("*.txt")):
        card = parse_forge_file(txt_file)
        if card is not None:
            cards.append(card)
        else:
            errors.append(f"Failed to parse: {txt_file}")
        processed += 1
        if processed % 5000 == 0:
            logger.info("Parsing Forge cards... %d parsed", processed)

    logger.info("Parsed %d cards total, %d errors", len(cards), len(errors))
    return cards, errors
