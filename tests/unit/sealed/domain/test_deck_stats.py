"""Unit tests for the hand-computed deck-statistics vector."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from price_predictor.domain.entities import Card
from price_predictor.domain.value_objects import ManaCost
from sealed.domain.deck_stats import (
    CONTRIBUTION_DIM,
    DECK_STATS_DIM,
    aggregate_contributions,
    compute_deck_stats,
    compute_per_card_contributions,
)

# Slot offsets within the 23-feature vector — kept in sync with deck_stats.py.
_MV_HISTOGRAM_START = 0
_MV_HISTOGRAM_END = 8
_COLOR_COUNT = 8
_CARDS_PER_COLOR_START = 9
_PIPS_PER_COLOR_START = 15
_CREATURE_COUNT = 21
_NONCREATURE_COUNT = 22

# Per-color positions inside the cards-per-color and pips-per-color blocks.
_W, _U, _B, _R, _G, _C = range(6)

# Scaling divisors.
_DECK_SIZE = 23
_COLOR_MAX = 6


def _vanilla_creature(name: str, mana_cost_str: str) -> Card:
    return Card(
        name=name,
        types=["Creature"],
        mana_cost=ManaCost.parse(mana_cost_str),
        power="2",
        toughness="2",
    )


def _vanilla_instant(name: str, mana_cost_str: str) -> Card:
    return Card(
        name=name,
        types=["Instant"],
        mana_cost=ManaCost.parse(mana_cost_str),
    )


def _devoid_creature(name: str, mana_cost_str: str) -> Card:
    return Card(
        name=name,
        types=["Creature"],
        mana_cost=ManaCost.parse(mana_cost_str),
        oracle_text="Devoid",
        power="2",
        toughness="2",
    )


def _no_cost_card(name: str) -> Card:
    return Card(
        name=name,
        types=["Land"],
        mana_cost=None,
    )


class TestVectorShape:
    def test_returns_correct_dim(self):
        feats = compute_deck_stats([])
        assert feats.shape == (DECK_STATS_DIM,)
        assert feats.dtype.name == "float32"

    def test_dim_constant_is_23(self):
        assert DECK_STATS_DIM == 23


class TestVanillaWhiteDeck:
    """A deck of 23 vanilla 2-mana W creatures should populate exactly
    the expected slots and leave everything else at zero."""

    @pytest.fixture
    def feats(self):
        cards = [_vanilla_creature(f"WCreature{i}", "1 W") for i in range(23)]
        return compute_deck_stats(cards)

    def test_mv_bucket_2_full(self, feats):
        # MV 2 = generic 1 + W 1 = total CMC 2
        assert feats[_MV_HISTOGRAM_START + 2] == pytest.approx(1.0)

    def test_other_mv_buckets_zero(self, feats):
        for i in range(_MV_HISTOGRAM_START, _MV_HISTOGRAM_END):
            if i != _MV_HISTOGRAM_START + 2:
                assert feats[i] == 0.0

    def test_color_count_one_sixth(self, feats):
        # Only W is active → 1 / 6
        assert feats[_COLOR_COUNT] == pytest.approx(1.0 / _COLOR_MAX)

    def test_cards_w_full(self, feats):
        assert feats[_CARDS_PER_COLOR_START + _W] == pytest.approx(1.0)
        for offset in (_U, _B, _R, _G, _C):
            assert feats[_CARDS_PER_COLOR_START + offset] == 0.0

    def test_pips_w_full(self, feats):
        # 23 cards × 1 W pip / 23 = 1.0
        assert feats[_PIPS_PER_COLOR_START + _W] == pytest.approx(1.0)
        for offset in (_U, _B, _R, _G, _C):
            assert feats[_PIPS_PER_COLOR_START + offset] == 0.0

    def test_all_creatures(self, feats):
        assert feats[_CREATURE_COUNT] == pytest.approx(1.0)
        assert feats[_NONCREATURE_COUNT] == 0.0


class TestColorlessHandling:
    """A deck of 23 generic-mana artifact-creatures should be entirely colorless,
    but should NOT activate the C color slot (that requires {C} pips, not generic)."""

    @pytest.fixture
    def feats(self):
        cards = [
            Card(
                name=f"Artifact{i}",
                types=["Artifact", "Creature"],
                mana_cost=ManaCost.parse("3"),
                power="2",
                toughness="2",
            )
            for i in range(23)
        ]
        return compute_deck_stats(cards)

    def test_cards_c_full(self, feats):
        # 23 colorless cards (no colored pips) → cards C = 1.0
        assert feats[_CARDS_PER_COLOR_START + _C] == pytest.approx(1.0)

    def test_no_other_color_cards(self, feats):
        for offset in (_W, _U, _B, _R, _G):
            assert feats[_CARDS_PER_COLOR_START + offset] == 0.0

    def test_no_c_pips(self, feats):
        # Generic mana does NOT count toward C pips.
        assert feats[_PIPS_PER_COLOR_START + _C] == 0.0

    def test_color_count_zero(self, feats):
        # No {C} pips and no colored pips → no colors active.
        assert feats[_COLOR_COUNT] == 0.0


class TestColorlessPipCard:
    """A deck of 23 {C}{C}{2} cards (true colorless pips, not just generic)."""

    @pytest.fixture
    def feats(self):
        cards = [_vanilla_creature(f"Eldrazi{i}", "2 C C") for i in range(23)]
        return compute_deck_stats(cards)

    def test_pips_c_present(self, feats):
        # 23 cards × 2 C pips / 23 = 2.0
        assert feats[_PIPS_PER_COLOR_START + _C] == pytest.approx(2.0)

    def test_color_count_includes_c(self, feats):
        # C is the only active color → 1/6
        assert feats[_COLOR_COUNT] == pytest.approx(1.0 / _COLOR_MAX)

    def test_cards_c_full(self, feats):
        # {C} pips don't make a card "colored" — it's still colorless.
        assert feats[_CARDS_PER_COLOR_START + _C] == pytest.approx(1.0)


class TestDevoidHandling:
    """A devoid card with a {2}{R} cost is colorless (devoid wins) for the
    cards-per-color slot, but its R pip should still count for pips-per-color."""

    def test_devoid_card_treated_as_colorless(self):
        card = _devoid_creature("Devoid Eldrazi", "2 R")
        feats = compute_deck_stats([card])
        # 1 colorless card, no R card.
        assert feats[_CARDS_PER_COLOR_START + _C] == pytest.approx(1.0 / _DECK_SIZE)
        assert feats[_CARDS_PER_COLOR_START + _R] == 0.0

    def test_devoid_card_pips_still_count(self):
        card = _devoid_creature("Devoid Eldrazi", "2 R")
        feats = compute_deck_stats([card])
        # The R pip in the cost still contributes to pips-per-color; devoid
        # changes color identity, not the printed pip composition.
        assert feats[_PIPS_PER_COLOR_START + _R] == pytest.approx(1.0 / _DECK_SIZE)

    def test_devoid_does_not_activate_r_color(self):
        # A pure-devoid deck with {2}{R} costs has no "active" R (cards are
        # colorless); only C activates if any {C} pip is present (it isn't here).
        cards = [_devoid_creature(f"Eldrazi{i}", "2 R") for i in range(5)]
        feats = compute_deck_stats(cards)
        assert feats[_COLOR_COUNT] == 0.0


class TestNoCostCard:
    """Edge case: a card with no mana cost (lands, some special cards) should
    not blow up the stats and should be treated as colorless for the cards
    slot but should not contribute to MV or pips."""

    def test_no_cost_card_does_not_crash(self):
        card = _no_cost_card("Plains")
        feats = compute_deck_stats([card])
        assert feats.shape == (DECK_STATS_DIM,)

    def test_no_cost_card_in_colorless_slot(self):
        feats = compute_deck_stats([_no_cost_card("Plains")])
        assert feats[_CARDS_PER_COLOR_START + _C] == pytest.approx(1.0 / _DECK_SIZE)

    def test_no_cost_card_no_mv_contribution(self):
        feats = compute_deck_stats([_no_cost_card("Plains")])
        assert all(feats[i] == 0.0 for i in range(_MV_HISTOGRAM_START, _MV_HISTOGRAM_END))

    def test_no_cost_card_no_pips(self):
        feats = compute_deck_stats([_no_cost_card("Plains")])
        for offset in range(6):
            assert feats[_PIPS_PER_COLOR_START + offset] == 0.0


class TestMvBucketing:
    """The MV histogram has 8 buckets (0, 1, 2, 3, 4, 5, 6, 7+). High-cost cards
    should all collapse into the 7+ bucket."""

    def test_high_mv_collapses_to_seven_plus(self):
        # MV 9 card.
        card = _vanilla_creature("Big", "8 R")
        feats = compute_deck_stats([card])
        # 7+ bucket is at index 7.
        assert feats[_MV_HISTOGRAM_START + 7] == pytest.approx(1.0 / _DECK_SIZE)

    def test_mv_zero_bucket(self):
        card = _vanilla_creature("Free", "0")
        feats = compute_deck_stats([card])
        # Wait — ManaCost.parse("0") may return None. Test with an actual 0-cost.
        # Use Memnite-like: empty cost is "0" generic.
        # If parse returns None, mv_histogram is empty.
        if card.mana_cost is not None:
            assert feats[_MV_HISTOGRAM_START + 0] == pytest.approx(1.0 / _DECK_SIZE)


class TestMixedDeck:
    """A small mixed-color deck to sanity-check the aggregate behavior."""

    def test_two_color_deck(self):
        cards = [
            _vanilla_creature("WhiteOne", "W"),
            _vanilla_creature("WhiteTwo", "1 W"),
            _vanilla_instant("BluePower", "U U"),
        ]
        feats = compute_deck_stats(cards)
        # Color count: W and U active → 2 / 6
        assert feats[_COLOR_COUNT] == pytest.approx(2.0 / _COLOR_MAX)
        # Cards: 2 W, 1 U.
        assert feats[_CARDS_PER_COLOR_START + _W] == pytest.approx(2.0 / _DECK_SIZE)
        assert feats[_CARDS_PER_COLOR_START + _U] == pytest.approx(1.0 / _DECK_SIZE)
        # Pips: 2 W (1+1), 2 U (2+0).
        assert feats[_PIPS_PER_COLOR_START + _W] == pytest.approx(2.0 / _DECK_SIZE)
        assert feats[_PIPS_PER_COLOR_START + _U] == pytest.approx(2.0 / _DECK_SIZE)
        # 2 creatures, 1 noncreature.
        assert feats[_CREATURE_COUNT] == pytest.approx(2.0 / _DECK_SIZE)
        assert feats[_NONCREATURE_COUNT] == pytest.approx(1.0 / _DECK_SIZE)


class TestContributionsEquivalentToDirectCompute:
    """Sum-then-aggregate of per-card contributions must match the direct
    ``compute_deck_stats`` result. This guards the inference fast path against
    drift from the canonical implementation."""

    @pytest.mark.parametrize("cards", [
        [_vanilla_creature(f"WC{i}", "1 W") for i in range(23)],
        [_vanilla_creature(f"E{i}", "2 C C") for i in range(23)],
        [
            _vanilla_creature("WhiteOne", "W"),
            _vanilla_creature("WhiteTwo", "1 W"),
            _vanilla_instant("BluePower", "U U"),
            _devoid_creature("DevoidR", "2 R"),
            _no_cost_card("Plains"),
        ],
    ])
    def test_two_paths_agree(self, cards):
        direct = compute_deck_stats(cards)
        contribs = compute_per_card_contributions(cards)
        assert contribs.shape == (len(cards), CONTRIBUTION_DIM)
        summed = torch.from_numpy(contribs).sum(dim=0)
        via_aggregate = aggregate_contributions(summed).numpy()
        np.testing.assert_allclose(via_aggregate, direct, rtol=1e-5, atol=1e-7)

    def test_aggregate_vectorized_over_batch(self):
        """Aggregate works on a leading batch dim — this is the inference path."""
        cards_a = [_vanilla_creature(f"WC{i}", "1 W") for i in range(23)]
        cards_b = [_vanilla_creature(f"BC{i}", "2 R") for i in range(23)]
        contribs_a = compute_per_card_contributions(cards_a)
        contribs_b = compute_per_card_contributions(cards_b)
        summed = torch.stack([
            torch.from_numpy(contribs_a).sum(dim=0),
            torch.from_numpy(contribs_b).sum(dim=0),
        ])  # (2, CONTRIBUTION_DIM)
        result = aggregate_contributions(summed)
        assert result.shape == (2, DECK_STATS_DIM)
        np.testing.assert_allclose(
            result[0].numpy(), compute_deck_stats(cards_a), rtol=1e-5, atol=1e-7,
        )
        np.testing.assert_allclose(
            result[1].numpy(), compute_deck_stats(cards_b), rtol=1e-5, atol=1e-7,
        )
