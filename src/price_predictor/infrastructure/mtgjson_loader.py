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
    logger.info("Loading AllPricesToday.json \u2014 building price map...")
    with open(allprices_path, encoding="utf-8") as f:
        data = json.load(f)

    prices_data = data.get("data", {})
    result: dict[str, float] = {}
    multi_printing_count = 0

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
                    if price is not None and price >= 0:
                        all_prices.append(price)

        if all_prices:
            selected_price = max(min(all_prices), 0.01)
            result[name] = selected_price
            if len(all_prices) > 1:
                multi_printing_count += 1
                logger.info(
                    "  %s: selected \u20ac%.2f from %d prices",
                    name,
                    selected_price,
                    len(all_prices),
                )

    logger.info(
        "Price selection summary: %d of %d cards had multiple price points",
        multi_printing_count,
        len(result),
    )
    logger.info("Loaded price data (%d cards with prices)", len(result))
    return result
