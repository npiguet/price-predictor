"""T008-T011 — Unit tests for mana_scorer domain module."""
from __future__ import annotations

import pytest

from sealed.domain.mana_scorer import (
    PipCounts,
    IdealDistribution,
    ActualSourceCounts,
    ManaScore,
    count_pips,
    compute_ideal_distribution,
    count_actual_sources,
    compute_mana_score,
    compute_mana_value,
    count_generic,
    count_x,
    count_mana_produced,
    extract_mana_features,
    normalize_mana_features,
)

# ─────────────────────────────────────────────────────────────────────────────
# T008 — count_pips()
# ─────────────────────────────────────────────────────────────────────────────

class TestCountPips:
    # Single-color pips
    def test_single_white_pip(self):
        text = "name: plains wanderer\nmana cost: {W}\ntypes: creature\n"
        result = count_pips([text])
        assert result.counts.get("W", 0.0) == pytest.approx(1.0)
        assert result.counts.get("U", 0.0) == 0.0

    def test_single_blue_pip(self):
        text = "name: sea sprite\nmana cost: {U}\ntypes: creature\n"
        result = count_pips([text])
        assert result.counts.get("U", 0.0) == pytest.approx(1.0)

    def test_multiple_same_color_pips(self):
        text = "name: serra angel\nmana cost: {3}{W}{W}\ntypes: creature\n"
        result = count_pips([text])
        assert result.counts.get("W", 0.0) == pytest.approx(2.0)

    # Phyrexian mana
    def test_phyrexian_mana_white(self):
        # {W/P} → +0.5 to W
        text = "name: apostle blessing\nmana cost: {1}{W/P}\ntypes: instant\n"
        result = count_pips([text])
        assert result.counts.get("W", 0.0) == pytest.approx(0.5)

    def test_phyrexian_mana_red_double(self):
        # {3}{R/P}{R/P} — actual Act of Aggression format
        text = "name: act of aggression\nmana cost: {3}{R/P}{R/P}\ntypes: instant\n"
        result = count_pips([text])
        assert result.counts.get("R", 0.0) == pytest.approx(1.0)

    # Hybrid mana
    def test_hybrid_mana_two_colors(self):
        # {G/R} → +0.5 G and +0.5 R
        text = "name: hybrid card\nmana cost: {G/R}\ntypes: creature\n"
        result = count_pips([text])
        assert result.counts.get("G", 0.0) == pytest.approx(0.5)
        assert result.counts.get("R", 0.0) == pytest.approx(0.5)

    def test_hybrid_mana_white_blue(self):
        # {3}{W/U} — Aethertow format
        text = "name: aethertow\nmana cost: {3}{W/U}\ntypes: instant\n"
        result = count_pips([text])
        assert result.counts.get("W", 0.0) == pytest.approx(0.5)
        assert result.counts.get("U", 0.0) == pytest.approx(0.5)

    # Generic mana — ignored
    def test_generic_mana_ignored(self):
        text = "name: generic card\nmana cost: {2}\ntypes: artifact\n"
        result = count_pips([text])
        for c in ("W", "U", "B", "R", "G", "C"):
            assert result.counts.get(c, 0.0) == 0.0

    def test_variable_mana_x_ignored(self):
        text = "name: fireball\nmana cost: {X}{R}\ntypes: sorcery\n"
        result = count_pips([text])
        # X is ignored; R counts
        assert result.counts.get("R", 0.0) == pytest.approx(1.0)
        assert result.counts.get("W", 0.0) == 0.0

    # Colorless mana
    def test_colorless_pip(self):
        text = "name: devoid creature\nmana cost: {1}{C}\ntypes: creature\n"
        result = count_pips([text])
        assert result.counts.get("C", 0.0) == pytest.approx(1.0)

    # Multi-face card (split)
    def test_multi_face_split_card_both_faces_counted(self):
        # Fire // Ice: {1}{R} and {1}{U}
        text = (
            "layout: split\n"
            "name: fire\n"
            "mana cost: {1}{R}\n"
            "types: instant\n"
            "\n"
            "ALTERNATE\n"
            "\n"
            "name: ice\n"
            "mana cost: {1}{U}\n"
            "types: instant\n"
        )
        result = count_pips([text])
        assert result.counts.get("R", 0.0) == pytest.approx(1.0)
        assert result.counts.get("U", 0.0) == pytest.approx(1.0)

    # No mana cost
    def test_no_mana_cost_line(self):
        text = "name: land\ntypes: basic land forest\nactivated[1]: {T}: add {G}\n"
        result = count_pips([text])
        for c in ("W", "U", "B", "R", "G", "C"):
            assert result.counts.get(c, 0.0) == 0.0

    # Multiple cards accumulate
    def test_multiple_cards_accumulate(self):
        text_w = "name: w card\nmana cost: {W}\ntypes: creature\n"
        text_u = "name: u card\nmana cost: {U}\ntypes: creature\n"
        result = count_pips([text_w, text_u])
        assert result.counts.get("W", 0.0) == pytest.approx(1.0)
        assert result.counts.get("U", 0.0) == pytest.approx(1.0)

    def test_same_card_twice_doubles_pips(self):
        text = "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
        result = count_pips([text, text])
        assert result.counts.get("R", 0.0) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# T009 — compute_ideal_distribution()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeIdealDistribution:
    def test_single_color_all_17_to_one_color(self):
        # n_colors=1, total=5.0, ideal[W] = 2 + (17-2)*5/5 = 17.0
        pips = PipCounts(counts={"W": 5.0})
        result = compute_ideal_distribution(pips)
        assert result.ideal.get("W", 0.0) == pytest.approx(17.0)
        assert result.ideal.get("U", 0.0) == 0.0

    def test_two_colors_proportional_with_floor(self):
        # n_colors=2, total=8.0, each=4.0
        # ideal[W] = 2 + (17-4)*4/8 = 2 + 6.5 = 8.5
        pips = PipCounts(counts={"W": 4.0, "G": 4.0})
        result = compute_ideal_distribution(pips)
        assert result.ideal.get("W", 0.0) == pytest.approx(8.5)
        assert result.ideal.get("G", 0.0) == pytest.approx(8.5)

    def test_two_colors_unequal_proportional(self):
        # W=6, G=3, n_colors=2, total=9
        # ideal[W] = 2 + (17-4)*6/9 = 2 + 13*2/3 ≈ 2 + 8.667 = 10.667
        # ideal[G] = 2 + (17-4)*3/9 = 2 + 13*1/3 ≈ 2 + 4.333 = 6.333
        pips = PipCounts(counts={"W": 6.0, "G": 3.0})
        result = compute_ideal_distribution(pips)
        assert result.ideal.get("W", 0.0) == pytest.approx(2 + 13 * 6 / 9)
        assert result.ideal.get("G", 0.0) == pytest.approx(2 + 13 * 3 / 9)

    def test_three_colors_proportional(self):
        # W=6, U=3, G=6, n_colors=3, total=15
        # remaining = 17 - 2*3 = 11
        # ideal[W] = 2 + 11*6/15 = 2 + 4.4 = 6.4
        # ideal[U] = 2 + 11*3/15 = 2 + 2.2 = 4.2
        # ideal[G] = 2 + 11*6/15 = 2 + 4.4 = 6.4
        pips = PipCounts(counts={"W": 6.0, "U": 3.0, "G": 6.0})
        result = compute_ideal_distribution(pips)
        remaining = 17 - 2 * 3
        total = 15.0
        assert result.ideal.get("W", 0.0) == pytest.approx(2 + remaining * 6.0 / total)
        assert result.ideal.get("U", 0.0) == pytest.approx(2 + remaining * 3.0 / total)
        assert result.ideal.get("G", 0.0) == pytest.approx(2 + remaining * 6.0 / total)

    def test_zero_pip_edge_case_returns_empty(self):
        pips = PipCounts(counts={})
        result = compute_ideal_distribution(pips)
        # No colors present → empty ideal
        for c in ("W", "U", "B", "R", "G", "C"):
            assert result.ideal.get(c, 0.0) == 0.0

    def test_colors_not_in_pips_not_in_ideal(self):
        pips = PipCounts(counts={"R": 3.0})
        result = compute_ideal_distribution(pips)
        assert "W" not in result.ideal
        assert "U" not in result.ideal
        assert result.ideal.get("R", 0.0) == pytest.approx(17.0)


# ─────────────────────────────────────────────────────────────────────────────
# T010 — count_actual_sources()
# ─────────────────────────────────────────────────────────────────────────────

class TestCountActualSources:
    # Basic lands
    def test_island_adds_blue(self):
        text = "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n"
        result = count_actual_sources([text])
        assert result.sources.get("U", 0.0) == pytest.approx(1.0)

    def test_plains_adds_white(self):
        text = "name: plains\ntypes: basic land plains\nactivated[1]: {T}: add {W}\n"
        result = count_actual_sources([text])
        assert result.sources.get("W", 0.0) == pytest.approx(1.0)

    def test_wastes_adds_colorless(self):
        text = "name: wastes\ntypes: basic land\nactivated[1]: {T}: add {C}.\n"
        result = count_actual_sources([text])
        assert result.sources.get("C", 0.0) == pytest.approx(1.0)

    # Dual lands
    def test_breeding_pool_adds_two_colors(self):
        # add {U} or {G}
        text = (
            "name: breeding pool\n"
            "types: land forest island\n"
            "activated[1]: {T}: add {U} or {G}\n"
        )
        result = count_actual_sources([text])
        assert result.sources.get("U", 0.0) == pytest.approx(1.0)
        assert result.sources.get("G", 0.0) == pytest.approx(1.0)

    # Tri-lands
    def test_jungle_shrine_adds_three_colors(self):
        # add {R}, {G}, or {W}.
        text = (
            "name: jungle shrine\n"
            "types: land\n"
            "activated[1]: {T}: add {R}, {G}, or {W}.\n"
        )
        result = count_actual_sources([text])
        assert result.sources.get("R", 0.0) == pytest.approx(1.0)
        assert result.sources.get("G", 0.0) == pytest.approx(1.0)
        assert result.sources.get("W", 0.0) == pytest.approx(1.0)

    # Sol Ring: {C}{C} → +1 C (set deduplication)
    def test_sol_ring_colorless_deduplication(self):
        # add {C}{C} → set({C, C}) = {C} → +1 C
        text = "name: sol ring\nmana cost: {1}\ntypes: artifact\nactivated[1]: {T}: add {C}{C}.\n"
        result = count_actual_sources([text])
        assert result.sources.get("C", 0.0) == pytest.approx(1.0)

    # Non-mana activated abilities are filtered out
    def test_non_mana_ability_filtered(self):
        # activated[1]: {3}{B}, {T}: seek — NOT a mana ability
        # activated[2]: {T}: add {B} — IS a mana ability
        text = (
            "name: gate of the black dragon\n"
            "types: land swamp gate\n"
            "activated[1]: {3}{B}, {T}: seek a nonland card. activate only once.\n"
            "activated[2]: {T}: add {B}\n"
        )
        result = count_actual_sources([text])
        assert result.sources.get("B", 0.0) == pytest.approx(1.0)
        # no other colors
        for c in ("W", "U", "R", "G", "C"):
            assert result.sources.get(c, 0.0) == 0.0

    def test_non_tap_only_ability_filtered(self):
        # Cost that is not purely {T}: should be ignored
        text = (
            "name: some tapper\n"
            "types: land\n"
            "activated[1]: {1}, {T}: add {G}\n"
        )
        result = count_actual_sources([text])
        # {1},{T}: add {G} — has extra cost → filtered
        assert result.sources.get("G", 0.0) == 0.0

    # "add one mana of any color" → +0
    def test_any_color_mana_contributes_zero(self):
        text = (
            "name: mana confluence\n"
            "types: land\n"
            "activated[1]: {T}: add one mana of any color\n"
        )
        result = count_actual_sources([text])
        for c in ("W", "U", "B", "R", "G", "C"):
            assert result.sources.get(c, 0.0) == 0.0

    # Multiple lands accumulate
    def test_multiple_lands_accumulate(self):
        island = "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n"
        plains = "name: plains\ntypes: basic land plains\nactivated[1]: {T}: add {W}\n"
        result = count_actual_sources([island, plains])
        assert result.sources.get("U", 0.0) == pytest.approx(1.0)
        assert result.sources.get("W", 0.0) == pytest.approx(1.0)

    def test_two_islands_stack(self):
        island = "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n"
        result = count_actual_sources([island, island])
        assert result.sources.get("U", 0.0) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────────────────────
# T011 — compute_mana_score()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeManaScore:
    def test_perfect_match_score_one(self):
        # l1=0, n_lands=17 → score=1.0, reward=1.0
        ideal = IdealDistribution(ideal={"W": 8.5, "G": 8.5})
        actual = ActualSourceCounts(sources={"W": 8.5, "G": 8.5})
        result = compute_mana_score(ideal, actual, n_lands=17)
        assert result.score == pytest.approx(1.0)
        assert result.reward == pytest.approx(1.0)
        assert result.l1_error == pytest.approx(0.0)
        assert result.n_lands == 17

    def test_land_count_deviation_penalty(self):
        # l1=0, n_lands=15 → score = 1 - (0+2)/17
        ideal = IdealDistribution(ideal={"W": 8.5, "G": 8.5})
        actual = ActualSourceCounts(sources={"W": 8.5, "G": 8.5})
        result = compute_mana_score(ideal, actual, n_lands=15)
        expected_score = 1.0 - (0.0 + abs(15 - 17)) / 17.0
        assert result.score == pytest.approx(expected_score)
        assert result.n_lands == 15

    def test_distribution_mismatch(self):
        # l1=4 (|6.5-8.5| + |10.5-8.5| = 2+2), n_lands=17
        ideal = IdealDistribution(ideal={"W": 8.5, "G": 8.5})
        actual = ActualSourceCounts(sources={"W": 6.5, "G": 10.5})
        result = compute_mana_score(ideal, actual, n_lands=17)
        l1 = abs(6.5 - 8.5) + abs(10.5 - 8.5)
        expected = max(0.0, 1.0 - l1 / 17.0)
        assert result.score == pytest.approx(expected)
        assert result.l1_error == pytest.approx(l1)

    def test_combined_errors(self):
        # ideal={W:17}, actual={W:12}, n_lands=14 → l1=5, score=(1-(5+3)/17)
        ideal = IdealDistribution(ideal={"W": 17.0})
        actual = ActualSourceCounts(sources={"W": 12.0})
        result = compute_mana_score(ideal, actual, n_lands=14)
        l1 = abs(12.0 - 17.0)
        expected = max(0.0, 1.0 - (l1 + abs(14 - 17)) / 17.0)
        assert result.score == pytest.approx(expected)

    def test_score_floor_at_zero(self):
        # ideal={W:17}, actual={W:0}, n_lands=17 → l1=17, score=max(0, 1-17/17)=0
        ideal = IdealDistribution(ideal={"W": 17.0})
        actual = ActualSourceCounts(sources={})
        result = compute_mana_score(ideal, actual, n_lands=17)
        assert result.score == pytest.approx(0.0)
        assert result.reward == pytest.approx(-1.0)

    def test_reward_mapping(self):
        # score=0.5 → reward=0.0
        ideal = IdealDistribution(ideal={"W": 17.0})
        # l1 + land_dev = 17.0 * 0.5 = 8.5 → score = 1 - 0.5 = 0.5
        actual = ActualSourceCounts(sources={"W": 17.0 - 8.5})  # actual=8.5, l1=8.5
        result = compute_mana_score(ideal, actual, n_lands=17)
        assert result.score == pytest.approx(0.5)
        assert result.reward == pytest.approx(0.0)

    def test_reward_in_range_negative_one_to_one(self):
        ideal = IdealDistribution(ideal={"W": 10.0, "U": 7.0})
        actual = ActualSourceCounts(sources={"W": 8.0, "U": 5.0})
        result = compute_mana_score(ideal, actual, n_lands=18)
        assert -1.0 <= result.reward <= 1.0
        assert 0.0 <= result.score <= 1.0

    def test_edge_all_lands_zero_spells(self):
        # 40 picks are all lands → no spells → pips={} → ideal={} → l1=0
        # n_lands=40, score = max(0, 1 - (0+23)/17) = 0.0
        ideal = IdealDistribution(ideal={})
        actual = ActualSourceCounts(sources={})
        result = compute_mana_score(ideal, actual, n_lands=40)
        assert result.score == pytest.approx(0.0)
        assert result.reward == pytest.approx(-1.0)

    def test_edge_all_spells_zero_lands(self):
        # 40 picks are all spells → n_lands=0
        # ideal={W:17}, actual={}, l1=17, score = max(0, 1-(17+17)/17) = 0.0
        ideal = IdealDistribution(ideal={"W": 17.0})
        actual = ActualSourceCounts(sources={})
        result = compute_mana_score(ideal, actual, n_lands=0)
        assert result.score == pytest.approx(0.0)

    def test_edge_colorless_only_deck(self):
        # All spells are colorless: ideal={C:17}, actual={C:17}, n_lands=17
        ideal = IdealDistribution(ideal={"C": 17.0})
        actual = ActualSourceCounts(sources={"C": 17.0})
        result = compute_mana_score(ideal, actual, n_lands=17)
        assert result.score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# T002 — compute_mana_value()
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeManaValue:
    def test_single_colored_pip(self):
        assert compute_mana_value("{W}") == pytest.approx(1.0)

    def test_generic_plus_colored(self):
        assert compute_mana_value("{2}{W}{W}") == pytest.approx(4.0)

    def test_hybrid_counts_as_one(self):
        assert compute_mana_value("{G/R}") == pytest.approx(1.0)

    def test_phyrexian_counts_as_one(self):
        assert compute_mana_value("{W/P}") == pytest.approx(1.0)

    def test_variable_x_contributes_zero(self):
        assert compute_mana_value("{X}{R}") == pytest.approx(1.0)

    def test_colorless_pip(self):
        assert compute_mana_value("{C}") == pytest.approx(1.0)

    def test_multi_generic_plus_colored(self):
        assert compute_mana_value("{4}{B}{B}{B}") == pytest.approx(7.0)

    def test_empty_string(self):
        assert compute_mana_value("") == pytest.approx(0.0)

    def test_pure_generic(self):
        assert compute_mana_value("{5}") == pytest.approx(5.0)

    def test_x_only(self):
        assert compute_mana_value("{X}") == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Explicit mana feature extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestCountGeneric:
    def test_generic_from_mixed_cost(self):
        assert count_generic("{3}{W}{W}") == pytest.approx(3.0)

    def test_no_generic(self):
        assert count_generic("{R}") == pytest.approx(0.0)

    def test_empty_string(self):
        assert count_generic("") == pytest.approx(0.0)

    def test_multiple_generic_symbols(self):
        assert count_generic("{2}{2}") == pytest.approx(4.0)

    def test_x_not_counted_as_generic(self):
        assert count_generic("{X}{R}") == pytest.approx(0.0)

    def test_colorless_pip_not_counted_as_generic(self):
        assert count_generic("{C}{C}") == pytest.approx(0.0)

    def test_hybrid_not_counted_as_generic(self):
        assert count_generic("{G/R}") == pytest.approx(0.0)


class TestCountX:
    def test_single_x(self):
        assert count_x("{X}{R}") == 1

    def test_double_x(self):
        assert count_x("{X}{X}{G}") == 2

    def test_no_x(self):
        assert count_x("{3}{W}") == 0

    def test_empty_string(self):
        assert count_x("") == 0

    def test_x_lowercase_not_matched(self):
        # Only uppercase {X} is a valid mana symbol; lowercase won't appear in practice
        assert count_x("{x}") == 0


class TestCountManaProduced:
    def test_forest_produces_green(self):
        text = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}.\n"
        result = count_mana_produced(text)
        assert result.get("G", 0.0) == pytest.approx(1.0)

    def test_sol_ring_produces_two_colorless(self):
        # Sol Ring: {T}: add {C}{C}
        text = "name: sol ring\nmana cost: {1}\ntypes: artifact\nactivated[1]: {T}: add {C}{C}.\n"
        result = count_mana_produced(text)
        assert result.get("C", 0.0) == pytest.approx(2.0)

    def test_non_mana_card_empty(self):
        text = "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
        result = count_mana_produced(text)
        assert result == {}

    def test_dual_land_produces_two_colors(self):
        text = "name: tundra\ntypes: land plains island\nactivated[1]: {T}: add {W} or {U}.\n"
        result = count_mana_produced(text)
        assert result.get("W", 0.0) == pytest.approx(1.0)
        assert result.get("U", 0.0) == pytest.approx(1.0)


class TestExtractManaFeatures:
    def test_returns_15_elements(self):
        result = extract_mana_features("name: lightning bolt\nmana cost: {R}\ntypes: instant\n")
        assert len(result) == 15

    def test_red_spell_features(self):
        # {R}: pip_W=0, pip_U=0, pip_B=0, pip_R=1, pip_G=0, pip_C=0,
        #      generic=0, x=0, mv=1, no mana produced
        result = extract_mana_features("name: lightning bolt\nmana cost: {R}\ntypes: instant\n")
        assert result[3] == pytest.approx(1.0)  # pip_R
        assert result[8] == pytest.approx(1.0)  # mana_value
        for i in [0, 1, 2, 4, 5, 6, 7]:
            assert result[i] == pytest.approx(0.0)

    def test_land_no_mana_cost(self):
        # Land has no mana cost: all pip/generic/x/mv are 0
        text = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}.\n"
        result = extract_mana_features(text)
        for i in range(9):
            assert result[i] == pytest.approx(0.0)
        assert result[13] == pytest.approx(1.0)  # mana_produced_G (index 9+4=13)

    def test_generic_mana_counted(self):
        # {3}{W}{W}: generic=3, pip_W=2, mv=5
        text = "name: baneslayer\nmana cost: {3}{W}{W}\ntypes: creature angel\n"
        result = extract_mana_features(text)
        assert result[6] == pytest.approx(3.0)   # generic
        assert result[0] == pytest.approx(2.0)   # pip_W
        assert result[8] == pytest.approx(5.0)   # mana_value

    def test_x_counted(self):
        text = "name: fireball\nmana cost: {X}{R}\ntypes: instant\n"
        result = extract_mana_features(text)
        assert result[7] == pytest.approx(1.0)   # x_count
        assert result[3] == pytest.approx(1.0)   # pip_R
        assert result[8] == pytest.approx(1.0)   # mana_value ({X} = 0, {R} = 1)


class TestNormalizeManaFeatures:
    def test_returns_15_elements(self):
        raw = [0.0] * 15
        assert len(normalize_mana_features(raw)) == 15

    def test_all_zeros_stays_zero(self):
        raw = [0.0] * 15
        for v in normalize_mana_features(raw):
            assert v == pytest.approx(0.0)

    def test_values_in_0_1_range(self):
        # Use large values to test clamping
        raw = [8.0, 8.0, 8.0, 8.0, 8.0, 3.0, 15.0, 3.0, 16.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        for v in normalize_mana_features(raw):
            assert 0.0 <= v <= 1.0, f"Value {v} out of [0, 1]"

    def test_max_values_normalize_to_one(self):
        raw = [8.0, 8.0, 8.0, 8.0, 8.0, 3.0, 15.0, 3.0, 16.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
        norm = normalize_mana_features(raw)
        for v in norm:
            assert v == pytest.approx(1.0)

    def test_pip_r_normalization(self):
        # pip_R (index 3) ÷ 8.0
        raw = [0.0] * 15
        raw[3] = 4.0  # 4 R pips
        norm = normalize_mana_features(raw)
        assert norm[3] == pytest.approx(0.5)

    def test_mana_value_normalization(self):
        # mana_value (index 8) ÷ 16.0
        raw = [0.0] * 15
        raw[8] = 8.0
        norm = normalize_mana_features(raw)
        assert norm[8] == pytest.approx(0.5)

    def test_complex_multicolor(self):
        # {2}{W}{U}{B} → 2+1+1+1 = 5
        assert compute_mana_value("{2}{W}{U}{B}") == pytest.approx(5.0)
