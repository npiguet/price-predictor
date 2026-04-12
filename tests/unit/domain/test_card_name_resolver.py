"""Tests for CardNameResolver."""

from __future__ import annotations

import pytest

from price_predictor.domain.card_name_resolver import CardNameResolver, ResolvedCard
from price_predictor.domain.value_objects import PrintingData


def _pd(rarity: str = "rare") -> PrintingData:
    return PrintingData(rarity=rarity, printings_count=1, release_year=2020)


class TestResolve:
    def test_resolves_exact_name(self) -> None:
        resolver = CardNameResolver(
            price_map={"Lightning Bolt": 1.50},
            metadata_map={"Lightning Bolt": _pd("common")},
        )

        result = resolver.resolve("Lightning Bolt")

        assert isinstance(result, ResolvedCard)
        assert result.canonical_name == "Lightning Bolt"
        assert result.price_eur == 1.50
        assert result.printing_data.rarity == "common"

    def test_resolves_lowercase_name(self) -> None:
        resolver = CardNameResolver(
            price_map={"Lightning Bolt": 1.50},
            metadata_map={"Lightning Bolt": _pd()},
        )

        result = resolver.resolve("lightning bolt")

        assert result is not None
        assert result.canonical_name == "Lightning Bolt"

    def test_resolves_split_card_front_face(self) -> None:
        resolver = CardNameResolver(
            price_map={"Fire // Ice": 0.75},
            metadata_map={"Fire // Ice": _pd()},
        )

        result = resolver.resolve("Fire")

        assert result is not None
        assert result.canonical_name == "Fire // Ice"
        assert result.price_eur == 0.75

    def test_returns_none_when_unknown(self) -> None:
        resolver = CardNameResolver(price_map={"Lightning Bolt": 1.50})

        assert resolver.resolve("Black Lotus") is None

    def test_defaults_printing_data_when_missing(self) -> None:
        resolver = CardNameResolver(
            price_map={"Lightning Bolt": 1.50}, metadata_map={}
        )

        result = resolver.resolve("Lightning Bolt")

        assert result is not None
        assert result.printing_data == PrintingData.defaults()

    def test_defaults_printing_data_when_metadata_map_is_none(self) -> None:
        resolver = CardNameResolver(price_map={"Lightning Bolt": 1.50})

        result = resolver.resolve("Lightning Bolt")

        assert result is not None
        assert result.printing_data == PrintingData.defaults()

    def test_split_prefix_does_not_match_substring(self) -> None:
        resolver = CardNameResolver(price_map={"Fire // Ice": 0.75})

        assert resolver.resolve("Fir") is None

    def test_resolve_without_price_map_returns_none(self) -> None:
        resolver = CardNameResolver(metadata_map={"Lightning Bolt": _pd()})

        # The name canonicalizes, but no price is recorded for it.
        assert resolver.resolve("Lightning Bolt") is None


class TestLookupPrintingData:
    def test_returns_metadata_for_known_card(self) -> None:
        pd = _pd("common")
        resolver = CardNameResolver(metadata_map={"Lightning Bolt": pd})

        assert resolver.lookup_printing_data("lightning bolt") is pd

    def test_returns_defaults_for_unknown_card(self) -> None:
        resolver = CardNameResolver(metadata_map={"Lightning Bolt": _pd()})

        assert resolver.lookup_printing_data("Black Lotus") == PrintingData.defaults()

    def test_split_card_lookup(self) -> None:
        pd = _pd()
        resolver = CardNameResolver(metadata_map={"Fire // Ice": pd})

        assert resolver.lookup_printing_data("Fire") is pd


class TestConstruction:
    def test_empty_maps_resolve_to_none(self) -> None:
        # Both maps empty is allowed — every lookup just yields nothing.
        resolver = CardNameResolver(price_map={}, metadata_map={})

        assert resolver.canonicalize("Lightning Bolt") is None
        assert resolver.lookup_printing_data("Lightning Bolt") == PrintingData.defaults()
