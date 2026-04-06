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


# ─────────────────────────────────────────────────────────────────────────────
# T002 — compute_recoverability_ratio()  (feature 016)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRecoverabilityRatio:
    """Tests for compute_recoverability_ratio() — pure mana-urgency math."""

    # Helper: pip_counts + actual_sources that give imbalance = 4.0
    # pip_counts = {W: 4} → ideal = {W: 17} (single-color, 1+remaining*pip/total)
    # actual_sources = {W: 13}  → |17 - 13| = 4.0
    @staticmethod
    def _pip4w():
        return PipCounts({"W": 4.0})

    @staticmethod
    def _actual13w():
        return {"W": 13.0}

    def test_ratio_early_step_remaining_35_exponent_2(self):
        """Step 5 of 40 picks: remaining=35, imbalance=4.0, exponent=2 → 4/35²≈0.003265."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        result = compute_recoverability_ratio(
            self._pip4w(), self._actual13w(), remaining_picks=35, exponent=2.0
        )
        assert result == pytest.approx(4.0 / 35 ** 2, rel=1e-5)

    def test_ratio_late_step_remaining_5_exponent_2(self):
        """Step 35 of 40 picks: remaining=5, imbalance=4.0, exponent=2 → 4/25=0.16."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        result = compute_recoverability_ratio(
            self._pip4w(), self._actual13w(), remaining_picks=5, exponent=2.0
        )
        assert result == pytest.approx(4.0 / 5 ** 2, rel=1e-5)

    def test_ratio_no_spells_picked_is_zero(self):
        """No spells, no lands → ideal is empty → imbalance = 0 → ratio = 0."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        result = compute_recoverability_ratio(
            PipCounts({}), {}, remaining_picks=20, exponent=2.0
        )
        assert result == pytest.approx(0.0)

    def test_ratio_perfect_balance_is_zero(self):
        """Perfect balance (actual = ideal) → imbalance = 0 → ratio = 0."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        # pip = {W: 4} → ideal = {W: 17}; actual = {W: 17} → imbalance = 0
        result = compute_recoverability_ratio(
            PipCounts({"W": 4.0}), {"W": 17.0}, remaining_picks=10, exponent=2.0
        )
        assert result == pytest.approx(0.0)

    def test_ratio_remaining_zero_equals_raw_imbalance(self):
        """remaining_picks=0 → denominator = max(0,1)^exp = 1 → ratio = imbalance."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        result = compute_recoverability_ratio(
            self._pip4w(), self._actual13w(), remaining_picks=0, exponent=2.0
        )
        # imbalance = 4.0, denominator = max(0,1)^2 = 1 → ratio = 4.0
        assert result == pytest.approx(4.0)

    def test_ratio_custom_exponent_3(self):
        """Custom exponent=3: ratio = 4.0/5³ = 4/125 = 0.032."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        result = compute_recoverability_ratio(
            self._pip4w(), self._actual13w(), remaining_picks=5, exponent=3.0
        )
        assert result == pytest.approx(4.0 / 5 ** 3, rel=1e-5)

    def test_ratio_custom_exponent_1(self):
        """Custom exponent=1: ratio = 4.0/5 = 0.8."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        result = compute_recoverability_ratio(
            self._pip4w(), self._actual13w(), remaining_picks=5, exponent=1.0
        )
        assert result == pytest.approx(4.0 / 5.0, rel=1e-5)

    def test_ratio_multicolor_imbalance(self):
        """Two-color pip counts: imbalance sums over both colors."""
        from sealed.domain.mana_scorer import compute_recoverability_ratio
        # pip = {W: 2, U: 2} → n_colors=2, total=4
        # ideal W: 2 + (17-2*2)*2/4 = 2+13*0.5 = 8.5, ideal U: 8.5
        # actual = {W: 5, U: 5} → |8.5-5| + |8.5-5| = 7.0
        result = compute_recoverability_ratio(
            PipCounts({"W": 2.0, "U": 2.0}),
            {"W": 5.0, "U": 5.0},
            remaining_picks=10,
            exponent=2.0,
        )
        expected_imbalance = 7.0
        assert result == pytest.approx(expected_imbalance / 10 ** 2, rel=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# T004 + T008 — compute_per_step_rewards()  (feature 016)
# ─────────────────────────────────────────────────────────────────────────────

class _MockCardPort:
    """Minimal CardEmbeddingPort stub for reward-function tests."""
    def __init__(self, land_names: set, card_texts: dict):
        self._land_names = land_names
        self._card_texts = card_texts

    def get_embedding(self, card_name: str):
        import numpy as _np
        return _np.zeros(8, dtype=_np.float32)

    def is_land(self, card_name: str) -> bool:
        return card_name in self._land_names

    def get_card_text(self, card_name: str) -> str:
        return self._card_texts.get(card_name, "")


_WHITE_SPELL_TEXT = "name: white knight\nmana cost: {1}{W}\ntypes: creature\n"
_PLAINS_TEXT = "name: plains\ntypes: basic land\nactivated[1]: {T}: add {W}\n"


def _make_port_ws_plains() -> _MockCardPort:
    """port for pool 'WhiteSpell;Plains' (index 0 = WhiteSpell, 1 = Plains)."""
    return _MockCardPort(
        land_names={"Plains"},
        card_texts={"WhiteSpell": _WHITE_SPELL_TEXT, "Plains": _PLAINS_TEXT},
    )


def _invoke_per_step(actions, pool_names, port, budget_rewards=None,
                     exponent=2.0, temperature=1.0):
    from sealed.domain.mana_scorer import compute_per_step_rewards
    import numpy as _np
    n = len(actions)
    if budget_rewards is None:
        budget_rewards = _np.ones(n, dtype=_np.float32)
    return compute_per_step_rewards(
        actions=_np.array(actions, dtype=_np.int32),
        pool_names=pool_names,
        card_port=port,
        budget_rewards=_np.array(budget_rewards, dtype=_np.float32),
        urgency_exponent=exponent,
        temperature=temperature,
    )


class TestComputePerStepRewards:
    """T004 — unit tests for compute_per_step_rewards()."""

    def test_returns_float32_array_length_40(self):
        """Result.step_rewards must be float32 with length == len(actions)."""
        import numpy as _np
        port = _make_port_ws_plains()
        result = _invoke_per_step([1] * 40, "WhiteSpell;Plains", port)
        assert result.step_rewards.dtype == _np.float32
        assert result.step_rewards.shape == (40,)

    def test_step_rewards_bounded_open_minus2_to_2(self):
        """step_rewards[t] must be strictly in (-2, 2) for all t."""
        import numpy as _np
        port = _make_port_ws_plains()
        budget = _np.tile([1.0, -1.0], 20).astype(_np.float32)
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=budget)
        assert all(-2.0 < v < 2.0 for v in result.step_rewards), (
            f"step_rewards out of (-2,2): {result.step_rewards}"
        )

    def test_plains_into_undersourced_white_positive_shaping(self):
        """Plains pick when under-sourced for white → step_reward > budget_reward."""
        # 1 WhiteSpell then 39 Plains; budget all +1
        # At step 10 (10th Plains), actual={W:9}, ideal≈{W:17} → under-sourced
        import numpy as _np
        port = _make_port_ws_plains()
        budget = _np.ones(40, dtype=_np.float32)
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=budget)
        # step 10: Plains into under-sourced → positive shaping → reward > budget
        assert result.step_rewards[10] > budget[10], (
            f"Expected step_rewards[10] > 1.0, got {result.step_rewards[10]:.6f}"
        )

    def test_plains_into_oversourced_white_negative_shaping(self):
        """Plains pick when over-sourced for white → step_reward < budget_reward."""
        # 1 WhiteSpell then 39 Plains; after step ~17, ideal≈17 is reached
        import numpy as _np
        port = _make_port_ws_plains()
        budget = _np.ones(40, dtype=_np.float32)
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=budget)
        # step 25: actual ≈ 24 > ideal ≈ 17 → over-sourced → negative shaping
        assert result.step_rewards[25] < budget[25], (
            f"Expected step_rewards[25] < 1.0, got {result.step_rewards[25]:.6f}"
        )

    def test_shaping_signals_not_all_identical(self):
        """Per-step shaping signals must vary (not all the same value)."""
        import numpy as _np
        port = _make_port_ws_plains()
        budget = _np.ones(40, dtype=_np.float32)
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=budget)
        unique_values = set(round(float(v), 8) for v in result.step_rewards)
        assert len(unique_values) > 1, (
            "step_rewards should have distinct per-step values, not be uniform"
        )

    def test_mean_shaping_diagnostic_is_finite_float(self):
        """mean_shaping must be a finite float."""
        import math
        port = _make_port_ws_plains()
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port)
        assert isinstance(result.mean_shaping, float)
        assert math.isfinite(result.mean_shaping)

    def test_final_imbalance_diagnostic_nonnegative(self):
        """final_imbalance must be a non-negative finite float."""
        import math
        port = _make_port_ws_plains()
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port)
        assert isinstance(result.final_imbalance, float)
        assert math.isfinite(result.final_imbalance)
        assert result.final_imbalance >= 0.0

    def test_multiface_adventure_card_counts_pips_from_all_faces(self):
        """FR-010: Multi-face card counts pips from ALL faces (both W and U here)."""
        import numpy as _np
        # Adventure card: face 1 = {1}{W} creature, face 2 = {2}{U} instant
        # count_pips sums W+U across faces
        adventure_text = (
            "name: brave adventurer\nmana cost: {1}{W}\ntypes: creature\n"
            "ALTERNATE\n"
            "name: heroic adventure\nmana cost: {2}{U}\ntypes: instant\n"
        )
        port = _MockCardPort(
            land_names={"Plains"},
            card_texts={
                "AdventureCard": adventure_text,
                "Plains": _PLAINS_TEXT,
            },
        )
        # 1 AdventureCard + 39 Plains (all add W source)
        result = _invoke_per_step([0] + [1] * 39, "AdventureCard;Plains", port)
        # pip_counts={W:1, U:1} → n_colors=2 → ideal={W:8.5, U:8.5}
        # actual_sources={W:39} → imbalance = |8.5-39| + |8.5-0| = 30.5 + 8.5 = 39.0
        assert result.final_imbalance == pytest.approx(39.0, abs=1.0)


class TestPerStepRewardBudgetPreservation:
    """T008 — budget signal is preserved at full strength (additive, unscaled)."""

    def test_shaping_component_bounded_minus1_to_1(self):
        """step_rewards[t] - budget_rewards[t] must be in [-1, 1] for all t (tanh output)."""
        import numpy as _np
        port = _make_port_ws_plains()
        budget = _np.tile([1.0, -1.0], 20).astype(_np.float32)
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=budget)
        shaping = result.step_rewards - budget
        for t, s in enumerate(shaping):
            assert -1.0 <= s <= 1.0, f"shaping[{t}]={s:.6f} not in [-1,1]"

    def test_budget_input_array_is_not_mutated(self):
        """The budget_rewards array passed in must not be modified by the function."""
        import numpy as _np
        port = _make_port_ws_plains()
        budget = _np.ones(40, dtype=_np.float32)
        budget_copy = budget.copy()
        _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port, budget_rewards=budget)
        _np.testing.assert_array_equal(budget, budget_copy,
                                       err_msg="budget_rewards array was mutated")

    def test_total_reward_bounded_minus2_to_2_positive_budget(self):
        """With budget=+1, step_rewards are in [-2, 2]."""
        import numpy as _np
        port = _make_port_ws_plains()
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=_np.ones(40, dtype=_np.float32))
        assert all(-2.0 <= v <= 2.0 for v in result.step_rewards)

    def test_total_reward_bounded_minus2_to_2_negative_budget(self):
        """With budget=-1, step_rewards are in [-2, 2]."""
        import numpy as _np
        port = _make_port_ws_plains()
        result = _invoke_per_step([0] + [1] * 39, "WhiteSpell;Plains", port,
                                  budget_rewards=_np.full(40, -1.0, dtype=_np.float32))
        assert all(-2.0 <= v <= 2.0 for v in result.step_rewards)
