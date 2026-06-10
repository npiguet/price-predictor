"""Composition-stats engine + sealed CLI wiring (analyze-generated-decks)."""

from __future__ import annotations

from dataclasses import dataclass

from sealed.application.analyze_generated_decks import (
    _bucket_mv,
    compute_deck_stats,
)
from sealed.domain.deck import Deck
from sealed.infrastructure.cli import build_parser, run_analyze_generated_decks
from sealed.infrastructure.pool_file_reader import GeneratedDeck


@dataclass
class _Mana:
    total_mana_value: float
    w: int = 0
    u: int = 0
    b: int = 0
    r: int = 0
    g: int = 0


class _Card:
    def __init__(self, types, mana=None) -> None:
        self.types = types
        self.mana_cost = mana

    def is_land(self) -> bool:
        return "Land" in self.types


def _deck(*names) -> GeneratedDeck:
    return GeneratedDeck(label="gen-1", set_code="TST", deck=Deck.of(names))


def test_compute_deck_stats_splits_lands_colors_and_curve() -> None:
    cache = {
        "Bear": _Card(["Creature"], _Mana(2.0, w=1, g=1)),
        "Bolt": _Card(["Instant"], _Mana(1.0, r=1)),
        "Tower": _Card(["Land"]),  # nonbasic land
    }
    deck = _deck("Forest", "Bear", "Bolt", "Tower")  # Forest is a basic land
    stats = compute_deck_stats(deck, cache, metadata=None)

    assert stats.basic_land_count == 1          # Forest
    assert stats.nonbasic_land_count == 1       # Tower
    assert stats.nonland_count == 2             # Bear, Bolt
    assert stats.colors_present == {"W", "G", "R"}
    assert stats.pip_counts["W"] == 1 and stats.pip_counts["G"] == 1
    assert stats.pip_counts["R"] == 1
    assert stats.mv_bucket_counts["1"] == 1 and stats.mv_bucket_counts["2"] == 1
    assert stats.type_counts["Land"] == 2 and stats.type_counts["Creature"] == 1
    assert stats.rarity_counts is None          # metadata not provided


def test_unresolved_cards_are_counted_not_crashing() -> None:
    deck = _deck("Mystery", "Plains")
    stats = compute_deck_stats(deck, {"Mystery": None}, metadata=None)
    assert stats.unresolved_count == 1          # Mystery has no card entry
    assert stats.basic_land_count == 1          # Plains still counts


def test_bucket_mv_edges() -> None:
    assert _bucket_mv(0) == "0"
    assert _bucket_mv(3.0) == "3"
    assert _bucket_mv(7) == "7+"
    assert _bucket_mv(12) == "7+"


def test_sealed_subcommand_dispatches() -> None:
    args = build_parser().parse_args(["analyze-generated-decks"])
    assert args.func is run_analyze_generated_decks
    assert args.decks_files == []               # defaults to generated-decks.txt
    assert args.no_rarity is False
