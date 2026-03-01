"""Load and process MTGJSON AllPrintings and AllPricesToday data."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_name_to_uuids(allprintings_path: Path) -> dict[str, list[str]]:
    """Build a mapping of card name -> list of UUIDs from AllPrintings.json.

    Filters to paper-available, English, non-funny, non-online-only cards.
    """
    logger.info("Loading AllPrintings.json \u2014 building name-to-UUID mapping...")
    mapping: dict[str, list[str]] = {}

    with open(allprintings_path, encoding="utf-8") as f:
        data = json.load(f)

    sets_data = data.get("data", {})
    for set_code, set_info in sets_data.items():
        cards = set_info if isinstance(set_info, list) else set_info.get("cards", [])
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

    logger.info("Built name-to-UUID mapping (%d card names)", len(mapping))
    return mapping


def get_cheapest_price(
    allprices_path: Path, uuids: list[str]
) -> float | None:
    """Get the cheapest Cardmarket EUR price across all given UUIDs.

    Checks both normal and foil finishes. Returns None if no price found.
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
                if price and price > 0:
                    all_prices.append(price)

    return min(all_prices) if all_prices else None


def build_price_map(
    allprices_path: Path, name_to_uuids: dict[str, list[str]]
) -> dict[str, float]:
    """Build a mapping of card name -> cheapest EUR price.

    Loads the prices file once and looks up all cards.
    Returns only cards with a valid price > 0.
    """
    logger.info("Loading AllPricesToday.json \u2014 building price map...")
    with open(allprices_path, encoding="utf-8") as f:
        data = json.load(f)

    prices_data = data.get("data", {})
    result: dict[str, float] = {}

    for name, uuids in name_to_uuids.items():
        all_prices: list[float] = []
        for uuid in uuids:
            uuid_data = prices_data.get(uuid, {})
            paper = uuid_data.get("paper", {})
            cardmarket = paper.get("cardmarket", {})
            retail = cardmarket.get("retail", {})

            for finish in ("normal", "foil"):
                finish_data = retail.get(finish, {})
                if finish_data:
                    latest_date = max(finish_data.keys())
                    price = finish_data[latest_date]
                    if price and price > 0:
                        all_prices.append(price)

        if all_prices:
            result[name] = min(all_prices)

    logger.info("Loaded price data (%d cards with prices)", len(result))
    return result
