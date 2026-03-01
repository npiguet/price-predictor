"""Tests for MTGJSON data loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from price_predictor.infrastructure.mtgjson_loader import (
    build_name_to_uuids,
    get_cheapest_price,
)


@pytest.fixture
def allprintings_path() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "allprintings_sample.json"


@pytest.fixture
def allprices_path() -> Path:
    return Path(__file__).parents[2] / "fixtures" / "allprices_sample.json"


class TestBuildNameToUuids:
    def test_builds_mapping(self, allprintings_path: Path) -> None:
        mapping = build_name_to_uuids(allprintings_path)
        assert "Grizzly Bears" in mapping
        assert len(mapping["Grizzly Bears"]) == 2  # 10E + 7ED

    def test_filters_funny_cards(self, allprintings_path: Path) -> None:
        mapping = build_name_to_uuids(allprintings_path)
        assert "AWOL" not in mapping  # isFunny=true

    def test_filters_online_only(self, allprintings_path: Path) -> None:
        mapping = build_name_to_uuids(allprintings_path)
        assert "Swords to Plowshares" not in mapping  # isOnlineOnly=true

    def test_paper_available_cards_included(self, allprintings_path: Path) -> None:
        mapping = build_name_to_uuids(allprintings_path)
        assert "Lightning Bolt" in mapping
        assert "Jace, the Mind Sculptor" in mapping
        assert "Sol Ring" in mapping
        assert "Matter Reshaper" in mapping

    def test_multiple_printings_aggregated(self, allprintings_path: Path) -> None:
        mapping = build_name_to_uuids(allprintings_path)
        assert len(mapping["Lightning Bolt"]) == 2  # A25 + 2XM
        assert len(mapping["Sol Ring"]) == 2  # C21 + C20


class TestGetCheapestPrice:
    def test_cheapest_across_printings(self, allprices_path: Path) -> None:
        # Grizzly Bears: 10E normal = 0.15, 7ED normal = 0.10, 7ED foil = 0.30
        uuids = [
            "aaaaaaaa-1111-1111-1111-111111111111",
            "bbbbbbbb-1111-1111-1111-111111111111",
        ]
        price = get_cheapest_price(allprices_path, uuids)
        assert price == 0.10  # cheapest across all printings/finishes

    def test_cheapest_includes_normal_and_foil(self, allprices_path: Path) -> None:
        # Lightning Bolt A25: normal = 2.50, foil = 5.30
        # Lightning Bolt 2XM: normal = 1.50 (cheapest)
        uuids = [
            "aaaaaaaa-3333-3333-3333-333333333333",
            "bbbbbbbb-3333-3333-3333-333333333333",
        ]
        price = get_cheapest_price(allprices_path, uuids)
        assert price == 1.50

    def test_missing_price_returns_none(self, allprices_path: Path) -> None:
        # Swords to Plowshares: empty price data
        uuids = ["eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"]
        price = get_cheapest_price(allprices_path, uuids)
        assert price is None

    def test_unknown_uuid_returns_none(self, allprices_path: Path) -> None:
        price = get_cheapest_price(allprices_path, ["nonexistent-uuid"])
        assert price is None

    def test_missing_cardmarket_section(self, allprices_path: Path) -> None:
        price = get_cheapest_price(allprices_path, ["eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee"])
        assert price is None
