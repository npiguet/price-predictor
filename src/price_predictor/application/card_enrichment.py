"""Card text enrichment: append printing data fields, extract, auto-fill, default."""

from __future__ import annotations

from price_predictor.domain.value_objects import RECOGNIZED_FORMATS, PrintingData

# The 6 printing data field keys as they appear in card text
_FIELD_KEYS = ("reserved", "rarity", "printings", "set", "legalities", "abu")


def enrich_card_text(text: str, printing_data: PrintingData) -> str:
    """Append the 5 printing data lines at the end of card text."""
    legalities_str = ", ".join(printing_data.legalities)
    lines = [
        f"reserved: {'true' if printing_data.is_reserved else 'false'}",
        f"rarity: {printing_data.rarity}",
        f"printings: {printing_data.printings_count}",
        f"set: {printing_data.set_code}",
        f"legalities: {legalities_str}",
        f"abu: {'true' if printing_data.is_abu else 'false'}",
    ]
    stripped = text.rstrip()
    return stripped + "\n" + "\n".join(lines)


def extract_printing_fields_from_text(text: str) -> dict[str, str]:
    """Detect which printing data keys are already present in the text.

    Returns a dict of field_key -> raw_value for each found field.
    """
    found: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        colon_idx = line.find(":")
        if colon_idx <= 0:
            continue
        key = line[:colon_idx].strip().lower()
        if key in _FIELD_KEYS:
            found[key] = line[colon_idx + 1:].strip()
    return found


def extract_printing_data_from_text(text: str) -> PrintingData | None:
    """Parse all 5 printing data fields from the text into a PrintingData.

    Returns None if no printing data fields are found in the text.
    """
    fields = extract_printing_fields_from_text(text)
    if not fields:
        return None

    is_reserved = fields.get("reserved", "false").lower() == "true"

    rarity = fields.get("rarity", "rare").lower()
    if rarity not in ("common", "uncommon", "rare", "mythic", "special", "bonus"):
        rarity = "rare"

    try:
        printings_count = max(int(fields.get("printings", "1")), 1)
    except ValueError:
        printings_count = 1

    set_code = fields.get("set", "ukn").lower()
    if not set_code:
        set_code = "ukn"

    legalities_raw = fields.get("legalities", "")
    if legalities_raw:
        legalities = sorted(
            fmt.strip().lower()
            for fmt in legalities_raw.split(",")
            if fmt.strip().lower() in RECOGNIZED_FORMATS
        )
    else:
        legalities = []

    is_abu = fields.get("abu", "false").lower() == "true"

    return PrintingData(
        is_reserved=is_reserved,
        rarity=rarity,
        printings_count=printings_count,
        set_code=set_code,
        legalities=legalities,
        is_abu=is_abu,
    )


def _extract_card_name(text: str) -> str | None:
    """Extract the card name from a name: line in card text."""
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("name:"):
            return line[len("name:"):].strip()
    return None


def enrich_or_default(
    text: str,
    metadata_map: dict[str, PrintingData],
) -> str:
    """Enrich card text with printing data: auto-fill, default, or merge.

    1. Parse existing printing data fields from the text.
    2. Look up card name in metadata_map for auto-fill values.
    3. Merge: client-provided fields override auto-fill/defaults.
    4. Return text with all 5 fields present.
    """
    existing = extract_printing_fields_from_text(text)
    card_name = _extract_card_name(text)

    # Determine base printing data: from metadata map or defaults
    lower_map = {k.lower(): v for k, v in metadata_map.items()}
    base: PrintingData | None = None
    if card_name:
        base = lower_map.get(card_name.lower())
    if base is None:
        base = PrintingData.defaults()

    # Merge: client-provided fields override base values
    is_reserved = (
        existing["reserved"].lower() == "true"
        if "reserved" in existing
        else base.is_reserved
    )

    rarity = existing.get("rarity", base.rarity).lower()
    if rarity not in ("common", "uncommon", "rare", "mythic", "special", "bonus"):
        rarity = base.rarity

    try:
        printings_count = (
            max(int(existing["printings"]), 1)
            if "printings" in existing
            else base.printings_count
        )
    except ValueError:
        printings_count = base.printings_count

    set_code = existing.get("set", base.set_code).lower()
    if not set_code:
        set_code = base.set_code

    if "legalities" in existing:
        raw = existing["legalities"]
        if raw:
            legalities = sorted(
                fmt.strip().lower()
                for fmt in raw.split(",")
                if fmt.strip().lower() in RECOGNIZED_FORMATS
            )
        else:
            legalities = []
    else:
        legalities = list(base.legalities)

    is_abu = (
        existing["abu"].lower() == "true"
        if "abu" in existing
        else base.is_abu
    )

    merged = PrintingData(
        is_reserved=is_reserved,
        rarity=rarity,
        printings_count=printings_count,
        set_code=set_code,
        legalities=legalities,
        is_abu=is_abu,
    )

    # Strip any existing printing data lines from text before re-appending
    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            colon_idx = stripped.find(":")
            if colon_idx > 0:
                key = stripped[:colon_idx].strip().lower()
                if key in _FIELD_KEYS:
                    continue
        clean_lines.append(line)

    clean_text = "\n".join(clean_lines)
    return enrich_card_text(clean_text, merged)
