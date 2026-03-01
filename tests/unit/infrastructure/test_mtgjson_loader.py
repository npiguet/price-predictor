"""Tests for MTGJSON data loader."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from price_predictor.infrastructure.mtgjson_loader import (
    build_name_to_uuids,
    build_price_map,
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

    def test_zero_price_clamped_to_floor(self, allprices_path: Path) -> None:
        # Zero Price Card: normal = 0.00 → should be clamped to 0.01
        uuids = ["ffffffff-0000-0000-0000-000000000000"]
        price = get_cheapest_price(allprices_path, uuids)
        assert price == 0.01

    def test_sub_cent_price_clamped_to_floor(self, allprices_path: Path) -> None:
        # Sub Cent Card: normal = 0.005 → should be clamped to 0.01
        uuids = ["ffffffff-0001-0001-0001-000000000001"]
        price = get_cheapest_price(allprices_path, uuids)
        assert price == 0.01

    def test_normal_price_above_floor_unchanged(self, allprices_path: Path) -> None:
        # Island: normal = 0.05 → above floor, should be unchanged
        uuids = ["aaaaaaaa-cccc-cccc-cccc-cccccccccccc"]
        price = get_cheapest_price(allprices_path, uuids)
        assert price == 0.05


class TestBuildPriceMapFloor:
    def test_zero_price_card_gets_floor(
        self, allprintings_path: Path, allprices_path: Path
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        price_map = build_price_map(allprices_path, name_to_uuids)
        assert price_map["Zero Price Card"] == 0.01

    def test_sub_cent_card_gets_floor(
        self, allprintings_path: Path, allprices_path: Path
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        price_map = build_price_map(allprices_path, name_to_uuids)
        assert price_map["Sub Cent Card"] == 0.01

    def test_existing_prices_unchanged(
        self, allprintings_path: Path, allprices_path: Path
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        price_map = build_price_map(allprices_path, name_to_uuids)
        # Grizzly Bears cheapest = 0.10 (unchanged, above floor)
        assert price_map["Grizzly Bears"] == 0.10
        # Lightning Bolt cheapest = 1.50 (unchanged, above floor)
        assert price_map["Lightning Bolt"] == 1.50


class TestBuildPriceMapPerCardLogging:
    """FR-008: Per-card price selection logging."""

    def test_multi_printing_card_emits_log(
        self,
        allprintings_path: Path,
        allprices_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        with caplog.at_level(
            logging.INFO, logger="price_predictor.infrastructure.mtgjson_loader"
        ):
            build_price_map(allprices_path, name_to_uuids)

        # Grizzly Bears has 2 printings with 3 prices (10E normal, 7ED normal, 7ED foil)
        bear_msgs = [m for m in caplog.messages if "Grizzly Bears" in m]
        assert len(bear_msgs) == 1
        assert "0.10" in bear_msgs[0]  # selected price
        assert "3" in bear_msgs[0]  # total price count

    def test_single_printing_card_no_per_card_log(
        self,
        allprintings_path: Path,
        allprices_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        with caplog.at_level(
            logging.INFO, logger="price_predictor.infrastructure.mtgjson_loader"
        ):
            build_price_map(allprices_path, name_to_uuids)

        # Serra Angel has only 1 printing with 2 prices (normal + foil) → still multi-price
        # But single-price cards like Delver of Secrets (1 printing, 1 finish) → no log
        delver_msgs = [m for m in caplog.messages if "Delver of Secrets" in m]
        assert len(delver_msgs) == 0


class TestBuildPriceMapSummaryLogging:
    """FR-007: Summary logging for multi-printing cards."""

    def test_summary_log_emitted(
        self,
        allprintings_path: Path,
        allprices_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        with caplog.at_level(
            logging.INFO, logger="price_predictor.infrastructure.mtgjson_loader"
        ):
            build_price_map(allprices_path, name_to_uuids)

        summary_msgs = [m for m in caplog.messages if "summary" in m.lower()]
        assert len(summary_msgs) == 1

    def test_summary_count_matches_multi_price_cards(
        self,
        allprintings_path: Path,
        allprices_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        with caplog.at_level(
            logging.INFO, logger="price_predictor.infrastructure.mtgjson_loader"
        ):
            price_map = build_price_map(allprices_path, name_to_uuids)

        summary_msgs = [m for m in caplog.messages if "summary" in m.lower()]
        assert len(summary_msgs) == 1
        # Multi-price cards in fixtures: Grizzly Bears (3 prices), Lightning Bolt (3),
        # Jace (4), Sol Ring (3), Serra Angel (2), Breeding Pool (2), Ragavan (2)
        # Count per-card log lines to cross-check
        per_card_msgs = [
            m
            for m in caplog.messages
            if "selected" in m.lower() and "summary" not in m.lower()
        ]
        count = len(per_card_msgs)
        assert str(count) in summary_msgs[0]


class TestBuildPriceMapExclusion:
    """FR-006/FR-008: Missing CardMarket price exclusion and reporting."""

    def test_no_price_card_excluded_from_map(
        self, allprintings_path: Path, allprices_path: Path
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        assert "No Price Card" in name_to_uuids  # passes name-to-UUID filter
        price_map = build_price_map(allprices_path, name_to_uuids)
        assert "No Price Card" not in price_map  # excluded — no price data

    def test_exclusion_count_logged(
        self,
        allprintings_path: Path,
        allprices_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        name_to_uuids = build_name_to_uuids(allprintings_path)
        with caplog.at_level(
            logging.INFO, logger="price_predictor.infrastructure.mtgjson_loader"
        ):
            price_map = build_price_map(allprices_path, name_to_uuids)

        exclusion_msgs = [m for m in caplog.messages if "exclusion" in m.lower()]
        assert len(exclusion_msgs) == 1
        excluded_count = len(name_to_uuids) - len(price_map)
        assert str(excluded_count) in exclusion_msgs[0]
        assert str(len(name_to_uuids)) in exclusion_msgs[0]
