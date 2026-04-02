"""T003/T004 — Unit tests for embedding_probe domain module."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from sealed.domain.embedding_probe import (
    CardData,
    ProbeSpec,
    ProbeResult,
    ValidationResult,
    extract_is_land,
    extract_card_color,
    extract_pip_counts,
    extract_mana_value,
    extract_mana_produced,
    build_default_probes,
    run_probes,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _card(text: str, embed_dim: int = 4) -> CardData:
    """Create a CardData with a deterministic dummy embedding."""
    rng = np.random.default_rng(abs(hash(text)) % (2**31))
    return CardData(
        name="test_card",
        embedding=rng.standard_normal(embed_dim).astype(np.float32),
        text=text,
    )


LAND_TEXT = "name: forest\ntypes: basic land forest\nactivated[1]: {T}: add {G}.\n"
SPELL_TEXT = "name: lightning bolt\nmana cost: {R}\ntypes: instant\n"
WUBR_TEXT = "name: multicolor\nmana cost: {W}{U}{B}{R}\ntypes: creature\n"
ARTIFACT_LAND_TEXT = (
    "name: sol ring\nmana cost: {1}\ntypes: artifact\n"
    "activated[1]: {T}: add {C}{C}.\n"
)
MANA_DORK_TEXT = (
    "name: llanowar elves\nmana cost: {G}\ntypes: creature elf druid\n"
    "activated[1]: {T}: add {G}.\n"
)
NON_MANA_TEXT = "name: goblin guide\nmana cost: {R}\ntypes: creature goblin scout\n"
HYBRID_TEXT = "name: hybrid card\nmana cost: {G/R}\ntypes: creature\n"
MV4_TEXT = "name: baneslayer angel\nmana cost: {3}{W}{W}\ntypes: creature angel\n"


# ─────────────────────────────────────────────────────────────────────────────
# T003 — Ground truth extraction functions
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractIsLand:
    def test_land_returns_one(self):
        cards = [_card(LAND_TEXT)]
        result = extract_is_land(cards)
        assert result[0] == pytest.approx(1.0)

    def test_non_land_returns_zero(self):
        cards = [_card(SPELL_TEXT)]
        result = extract_is_land(cards)
        assert result[0] == pytest.approx(0.0)

    def test_mixed_cards(self):
        cards = [_card(LAND_TEXT), _card(SPELL_TEXT), _card(LAND_TEXT)]
        result = extract_is_land(cards)
        assert result.tolist() == pytest.approx([1.0, 0.0, 1.0])

    def test_returns_ndarray(self):
        assert isinstance(extract_is_land([_card(LAND_TEXT)]), np.ndarray)

    def test_creature_land_is_land(self):
        text = "name: dryad arbor\ntypes: land creature forest dryad\nactivated[1]: {T}: add {G}.\n"
        result = extract_is_land([_card(text)])
        assert result[0] == pytest.approx(1.0)


class TestExtractCardColor:
    def test_single_color_red(self):
        cards = [_card(SPELL_TEXT)]
        assert extract_card_color(cards, "R")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "W")[0] == pytest.approx(0.0)

    def test_multicolor_card(self):
        cards = [_card(WUBR_TEXT)]
        for color in ("W", "U", "B", "R"):
            assert extract_card_color(cards, color)[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "G")[0] == pytest.approx(0.0)

    def test_land_no_color(self):
        cards = [_card(LAND_TEXT)]
        for color in ("W", "U", "B", "R", "G", "C"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0)

    def test_hybrid_card_counts_both_colors(self):
        cards = [_card(HYBRID_TEXT)]
        # {G/R} → 0.5 G and 0.5 R — both > 0 so both are colored
        assert extract_card_color(cards, "G")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "R")[0] == pytest.approx(1.0)


class TestExtractPipCounts:
    def test_single_red_pip(self):
        cards = [_card(SPELL_TEXT)]
        assert extract_pip_counts(cards, "R")[0] == pytest.approx(1.0)
        assert extract_pip_counts(cards, "W")[0] == pytest.approx(0.0)

    def test_hybrid_fractional(self):
        cards = [_card(HYBRID_TEXT)]
        # {G/R} → 0.5 each
        assert extract_pip_counts(cards, "G")[0] == pytest.approx(0.5)
        assert extract_pip_counts(cards, "R")[0] == pytest.approx(0.5)

    def test_land_zero_pips(self):
        cards = [_card(LAND_TEXT)]
        for color in ("W", "U", "B", "R", "G", "C"):
            assert extract_pip_counts(cards, color)[0] == pytest.approx(0.0)

    def test_double_colored_pips(self):
        cards = [_card(MV4_TEXT)]  # {3}{W}{W}
        assert extract_pip_counts(cards, "W")[0] == pytest.approx(2.0)


class TestExtractManaValue:
    def test_lightning_bolt_mv1(self):
        cards = [_card(SPELL_TEXT)]  # {R}
        assert extract_mana_value(cards)[0] == pytest.approx(1.0)

    def test_baneslayer_mv5(self):
        cards = [_card(MV4_TEXT)]  # {3}{W}{W}
        assert extract_mana_value(cards)[0] == pytest.approx(5.0)

    def test_land_mv0(self):
        cards = [_card(LAND_TEXT)]
        assert extract_mana_value(cards)[0] == pytest.approx(0.0)

    def test_hybrid_mv1(self):
        cards = [_card(HYBRID_TEXT)]  # {G/R}
        assert extract_mana_value(cards)[0] == pytest.approx(1.0)


class TestExtractManaProduced:
    def test_basic_land_produces_color(self):
        cards = [_card(LAND_TEXT)]  # Forest → {G}
        assert extract_mana_produced(cards, "G")[0] == pytest.approx(1.0)
        assert extract_mana_produced(cards, "W")[0] == pytest.approx(0.0)

    def test_mana_rock_produces_colorless(self):
        cards = [_card(ARTIFACT_LAND_TEXT)]  # Sol Ring → {C}{C} deduped → {C}
        assert extract_mana_produced(cards, "C")[0] == pytest.approx(1.0)

    def test_mana_dork_produces_green(self):
        cards = [_card(MANA_DORK_TEXT)]  # Llanowar Elves
        assert extract_mana_produced(cards, "G")[0] == pytest.approx(1.0)

    def test_non_mana_card_zero(self):
        cards = [_card(NON_MANA_TEXT)]
        for color in ("W", "U", "B", "R", "G", "C"):
            assert extract_mana_produced(cards, color)[0] == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# T004 — Probe runner
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDefaultProbes:
    def test_returns_20_probes(self):
        # 1 is-land + 6 card-color + 6 pip-counts + 1 mana-value + 6 mana-produced = 20
        probes = build_default_probes()
        assert len(probes) == 20

    def test_probe_types_correct(self):
        probes = build_default_probes()
        classification = [p for p in probes if p.probe_type == "classification"]
        regression = [p for p in probes if p.probe_type == "regression"]
        # is-land:1, card-color:6, mana-produced:6 = 13 classification
        assert len(classification) == 13
        # pip-counts:6, mana-value:1 = 7 regression
        assert len(regression) == 7

    def test_is_land_threshold_floor(self):
        # is-land uses max(threshold_accuracy, 0.99)
        probes = build_default_probes(threshold_accuracy=0.80)
        is_land = next(p for p in probes if p.feature_name == "Is land")
        assert is_land.threshold == pytest.approx(0.99)

    def test_is_land_threshold_above_floor(self):
        probes = build_default_probes(threshold_accuracy=0.999)
        is_land = next(p for p in probes if p.feature_name == "Is land")
        assert is_land.threshold == pytest.approx(0.999)

    def test_mana_value_threshold_floor(self):
        # mana value uses max(threshold_r2, 0.90)
        probes = build_default_probes(threshold_r2=0.70)
        mv = next(p for p in probes if p.feature_name == "Mana value")
        assert mv.threshold == pytest.approx(0.90)

    def test_mana_value_threshold_above_floor(self):
        probes = build_default_probes(threshold_r2=0.95)
        mv = next(p for p in probes if p.feature_name == "Mana value")
        assert mv.threshold == pytest.approx(0.95)

    def test_card_color_threshold_overrideable(self):
        probes = build_default_probes(threshold_accuracy=0.80)
        color_w = next(p for p in probes if p.feature_name == "Card color (W)")
        assert color_w.threshold == pytest.approx(0.80)

    def test_pip_counts_threshold_overrideable(self):
        probes = build_default_probes(threshold_r2=0.70)
        pip_w = next(p for p in probes if p.feature_name == "Pip counts (W)")
        assert pip_w.threshold == pytest.approx(0.70)

    def test_all_six_colors_present_in_card_color(self):
        probes = build_default_probes()
        names = {p.feature_name for p in probes}
        for color in ("W", "U", "B", "R", "G", "C"):
            assert f"Card color ({color})" in names

    def test_all_six_colors_present_in_mana_produced(self):
        probes = build_default_probes()
        names = {p.feature_name for p in probes}
        for color in ("W", "U", "B", "R", "G", "C"):
            assert f"Mana produced ({color})" in names


class TestRunProbes:
    def _make_cards(self, n: int = 20) -> list[CardData]:
        rng = np.random.default_rng(0)
        cards = []
        for i in range(n):
            cards.append(CardData(
                name=f"card_{i}",
                embedding=rng.standard_normal(4).astype(np.float32),
                text=LAND_TEXT if i % 2 == 0 else SPELL_TEXT,
            ))
        return cards

    def test_returns_one_result_per_probe(self):
        cards = self._make_cards(20)
        probes = build_default_probes()

        fake_scores = np.array([0.9, 0.9, 0.9, 0.9, 0.9])
        with patch("sealed.domain.embedding_probe.cross_val_score", return_value=fake_scores):
            results = run_probes(cards, probes)

        assert len(results) == len(probes)

    def test_passed_when_score_above_threshold(self):
        cards = self._make_cards(20)
        probes = [build_default_probes()[0]]  # is-land, threshold 0.99

        with patch("sealed.domain.embedding_probe.cross_val_score", return_value=np.array([1.0] * 5)):
            results = run_probes(cards, probes)

        assert results[0].passed is True
        assert results[0].score == pytest.approx(1.0)

    def test_failed_when_score_below_threshold(self):
        cards = self._make_cards(20)
        probes = [build_default_probes()[0]]  # is-land, threshold 0.99

        with patch("sealed.domain.embedding_probe.cross_val_score", return_value=np.array([0.5] * 5)):
            results = run_probes(cards, probes)

        assert results[0].passed is False
        assert results[0].score == pytest.approx(0.5)

    def test_n_samples_correct(self):
        cards = self._make_cards(20)
        probes = [build_default_probes()[0]]

        with patch("sealed.domain.embedding_probe.cross_val_score", return_value=np.array([0.9] * 5)):
            results = run_probes(cards, probes)

        assert results[0].n_samples == 20
