"""Load and process MTGJSON AllPrintings and AllPricesToday data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from price_predictor.domain.value_objects import RECOGNIZED_FORMATS, PrintingData

logger = logging.getLogger(__name__)

# Set codes for Alpha, Beta, and Unlimited — the three original core sets.
# A card whose only tournament-legal printings are in these sets gets is_abu=True.
_ABU_SET_CODES = frozenset({"lea", "leb", "2ed"})


def _is_constructed_legal(legalities: dict) -> bool:
    """Return True if the card is Legal or Restricted in at least one constructed format."""
    return any(
        legalities.get(fmt, "") in ("Legal", "Restricted")
        for fmt in RECOGNIZED_FORMATS
    )


def build_name_to_uuids(
    allprintings_path: Path,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    """Build a mapping of card name -> list of UUIDs from AllPrintings.json.

    Filters to paper-available, English, non-funny, non-online-only cards.

    Returns:
        (name_to_uuids, uuid_metadata) where uuid_metadata maps UUID to
        a dict with keys: rarity, setCode, isReserved, legalities, printings.
    """
    logger.info("Loading AllPrintings.json — building name-to-UUID mapping...")
    mapping: dict[str, list[str]] = {}
    uuid_meta: dict[str, dict[str, Any]] = {}

    with open(allprintings_path, encoding="utf-8") as f:
        data = json.load(f)

    sets_data = data.get("data", {})
    for set_code, set_info in sets_data.items():
        cards = set_info if isinstance(set_info, list) else set_info.get("cards", [])

        # Extract release year from set-level data; fallback to 1993
        release_year = 1993
        if isinstance(set_info, dict):
            raw_date = set_info.get("releaseDate", "")
            try:
                release_year = int(str(raw_date)[:4])
                if release_year < 1993:
                    release_year = 1993
            except (ValueError, TypeError):
                release_year = 1993

        for card in cards:
            # Apply filters
            availability = card.get("availability", [])
            if "paper" not in availability:
                continue
            if card.get("isFunny", False):
                continue
            if card.get("isOnlineOnly", False):
                continue
            if card.get("language", "English") != "English":
                continue

            name = card.get("name", "")
            uuid = card.get("uuid", "")
            if name and uuid:
                if name not in mapping:
                    mapping[name] = []
                mapping[name].append(uuid)
                raw_legalities = card.get("legalities", {})
                uuid_meta[uuid] = {
                    "rarity": card.get("rarity", ""),
                    "setCode": card.get("setCode", set_code),
                    "isReserved": card.get("isReserved", False),
                    "legalities": raw_legalities,
                    "printings": card.get("printings", []),
                    "is_constructed_legal": _is_constructed_legal(raw_legalities),
                    "releaseYear": release_year,
                }

    logger.info("Built name-to-UUID mapping (%d card names)", len(mapping))
    return mapping, uuid_meta


def get_cheapest_price(
    allprices_path: Path, uuids: list[str]
) -> float | None:
    """Get the cheapest Cardmarket EUR price across all given UUIDs.

    Checks both normal and foil finishes. Applies a €0.01 price floor.
    Returns None if no price found.
    """
    with open(allprices_path, encoding="utf-8") as f:
        data = json.load(f)

    prices_data = data.get("data", {})
    all_prices: list[float] = []

    for uuid in uuids:
        uuid_data = prices_data.get(uuid, {})
        paper = uuid_data.get("paper", {})
        cardmarket = paper.get("cardmarket", {})
        retail = cardmarket.get("retail", {})

        for finish in ("normal", "foil"):
            finish_data = retail.get(finish, {})
            if finish_data:
                # Get the most recent date's price
                latest_date = max(finish_data.keys())
                price = finish_data[latest_date]
                if price is not None and price >= 0:
                    all_prices.append(price)

    return max(min(all_prices), 0.01) if all_prices else None


def build_price_map(
    allprices_path: Path, name_to_uuids: dict[str, list[str]]
) -> dict[str, float]:
    """Build a mapping of card name -> cheapest EUR price.

    Loads the prices file once and looks up all cards.
    Applies a €0.01 price floor to all prices.
    Returns only cards with a valid price >= 0.
    """
    price_map, _ = build_price_map_with_uuids(allprices_path, name_to_uuids)
    return price_map


def build_price_map_with_uuids(
    allprices_path: Path,
    name_to_uuids: dict[str, list[str]],
    legal_uuids: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Build price map AND track which UUID produced the cheapest price.

    Args:
        legal_uuids: When provided, only UUIDs in this set are considered for
            price selection. Used to exclude non-constructed-legal printings
            (Collector's Edition, 30th Anniversary, etc.) whose prices do not
            reflect the actual tournament card market.

    Returns:
        (price_map, cheapest_uuid_map) where price_map is name->price
        and cheapest_uuid_map is name->uuid of the cheapest printing.
    """
    logger.info("Loading AllPricesToday.json — building price map...")
    with open(allprices_path, encoding="utf-8") as f:
        data = json.load(f)

    prices_data = data.get("data", {})
    result: dict[str, float] = {}
    cheapest_uuids: dict[str, str] = {}
    multi_printing_count = 0

    for name, uuids in name_to_uuids.items():
        best_price: float | None = None
        best_uuid: str | None = None

        candidate_uuids = [u for u in uuids if legal_uuids is None or u in legal_uuids]
        for uuid in candidate_uuids:
            uuid_data = prices_data.get(uuid, {})
            paper = uuid_data.get("paper", {})
            cardmarket = paper.get("cardmarket", {})
            retail = cardmarket.get("retail", {})

            for finish in ("normal", "foil"):
                finish_data = retail.get(finish, {})
                if finish_data:
                    latest_date = max(finish_data.keys())
                    price = finish_data[latest_date]
                    if price is not None and price >= 0:
                        if best_price is None or price < best_price:
                            best_price = price
                            best_uuid = uuid

        if best_price is not None and best_uuid is not None:
            selected_price = max(best_price, 0.01)
            result[name] = selected_price
            cheapest_uuids[name] = best_uuid
            if len(candidate_uuids) > 1:
                multi_printing_count += 1
                logger.debug(
                    "  %s: selected €%.2f from uuid %s",
                    name,
                    selected_price,
                    best_uuid,
                )

    logger.info(
        "Price selection summary: %d of %d cards had multiple printings",
        multi_printing_count,
        len(result),
    )
    excluded_count = len(name_to_uuids) - len(result)
    logger.info(
        "Price exclusion: %d of %d cards excluded (no CardMarket price available)",
        excluded_count,
        len(name_to_uuids),
    )
    logger.info("Loaded price data (%d cards with prices)", len(result))
    return result, cheapest_uuids


def build_metadata_map(
    allprintings_path: Path,
    allprices_path: Path,
) -> tuple[dict[str, PrintingData], dict[str, float]]:
    """Build a metadata map: card name -> PrintingData from cheapest printing.

    Also returns the price map for convenience (avoids re-loading prices).

    Returns:
        (metadata_map, price_map) where metadata_map is name->PrintingData
        derived from the cheapest printing's data.
    """
    name_to_uuids, uuid_meta = build_name_to_uuids(allprintings_path)

    legal_uuids = {
        uuid for uuid, meta in uuid_meta.items()
        if meta.get("is_constructed_legal", False)
    }
    logger.info(
        "Constructed-legal UUIDs: %d of %d total",
        len(legal_uuids), len(uuid_meta),
    )

    price_map, cheapest_uuids = build_price_map_with_uuids(
        allprices_path, name_to_uuids, legal_uuids=legal_uuids,
    )

    metadata_map: dict[str, PrintingData] = {}
    for name, cheapest_uuid in cheapest_uuids.items():
        meta = uuid_meta.get(cheapest_uuid, {})

        # isReserved: absent means False
        is_reserved = bool(meta.get("isReserved", False))

        # rarity: fallback to "rare" for missing/unknown
        raw_rarity = meta.get("rarity", "").lower()
        if raw_rarity in ("common", "uncommon", "rare", "mythic"):
            rarity = raw_rarity
        else:
            rarity = "rare"

        # printings count
        printings_list = meta.get("printings", [])
        printings_count = max(len(printings_list), 1)

        # release year
        release_year = meta.get("releaseYear", 1993)
        if not isinstance(release_year, int) or release_year < 1993:
            release_year = 1993

        # legalities: filter to recognized formats, only "Legal" status
        raw_legalities = meta.get("legalities", {})
        # MTGJSON uses lowercase format keys; RECOGNIZED_FORMATS is also lowercase
        legalities = sorted(
            fmt
            for fmt in RECOGNIZED_FORMATS
            if raw_legalities.get(fmt, "") == "Legal"
        )

        # is_abu: all tournament-legal printings of this card are in Alpha/Beta/Unlimited
        card_legal_uuids = [u for u in name_to_uuids.get(name, []) if u in legal_uuids]
        legal_set_codes = {
            uuid_meta[u]["setCode"].lower()
            for u in card_legal_uuids
            if u in uuid_meta
        }
        is_abu = bool(legal_set_codes) and legal_set_codes.issubset(_ABU_SET_CODES)

        metadata_map[name] = PrintingData(
            is_reserved=is_reserved,
            rarity=rarity,
            printings_count=printings_count,
            release_year=release_year,
            legalities=legalities,
            is_abu=is_abu,
        )

    logger.info("Built metadata map (%d cards)", len(metadata_map))
    return metadata_map, price_map
