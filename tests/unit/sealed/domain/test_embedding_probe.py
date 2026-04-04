"""Tests for embedding_probe domain module."""
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
from tests.conftest import load_card_fixture


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
        # Land (no mana cost) → no colored pips, so W/U/B/R/G=0 and C=1 (colorless)
        cards = [_card(LAND_TEXT)]
        for color in ("W", "U", "B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0)
        assert extract_card_color(cards, "C")[0] == pytest.approx(1.0)

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
        ordinal = [p for p in probes if p.probe_type == "ordinal"]
        # is-land:1, card-color:6, mana-produced:6 = 13 classification
        assert len(classification) == 13
        # pip-counts:6, mana-value:1 = 7 ordinal
        assert len(ordinal) == 7

    def test_is_land_threshold_floor(self):
        # is-land uses max(threshold_accuracy, 0.99)
        probes = build_default_probes(threshold_accuracy=0.80)
        is_land = next(p for p in probes if p.feature_name == "Is land")
        assert is_land.threshold == pytest.approx(0.99)

    def test_is_land_threshold_above_floor(self):
        probes = build_default_probes(threshold_accuracy=0.999)
        is_land = next(p for p in probes if p.feature_name == "Is land")
        assert is_land.threshold == pytest.approx(0.999)

    def test_mana_value_uses_accuracy_threshold(self):
        probes = build_default_probes(threshold_accuracy=0.80)
        mv = next(p for p in probes if p.feature_name == "Mana value")
        assert mv.threshold == pytest.approx(0.80)

    def test_mana_value_threshold_overrideable(self):
        probes = build_default_probes(threshold_accuracy=0.90)
        mv = next(p for p in probes if p.feature_name == "Mana value")
        assert mv.threshold == pytest.approx(0.90)

    def test_card_color_threshold_overrideable(self):
        probes = build_default_probes(threshold_accuracy=0.80)
        color_w = next(p for p in probes if p.feature_name == "Card color (W)")
        assert color_w.threshold == pytest.approx(0.80)

    def test_pip_counts_threshold_overrideable(self):
        probes = build_default_probes(threshold_accuracy=0.80)
        pip_w = next(p for p in probes if p.feature_name == "Pip counts (W)")
        assert pip_w.threshold == pytest.approx(0.80)

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
        fake_preds = np.zeros(20)
        with patch("sealed.domain.embedding_probe.cross_val_score", return_value=fake_scores), \
             patch("sealed.domain.embedding_probe.cross_val_predict", return_value=fake_preds):
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

    def test_embed_dim_slices_embedding(self):
        """embed_dim parameter must restrict probes to first N embedding dimensions."""
        rng = np.random.default_rng(42)
        # Cards with 6-dim embeddings: first 2 dims are meaningful, last 4 are appended extras
        cards = []
        for i in range(20):
            full_emb = rng.standard_normal(6).astype(np.float32)
            cards.append(CardData(
                name=f"card_{i}",
                embedding=full_emb,
                text=LAND_TEXT if i % 2 == 0 else SPELL_TEXT,
            ))
        probes = [build_default_probes()[0]]  # is-land

        seen_shapes = []

        def capture_shape(estimator, X, y, **kwargs):
            seen_shapes.append(X.shape)
            return np.array([0.9] * 5)

        with patch("sealed.domain.embedding_probe.cross_val_score", side_effect=capture_shape), \
             patch("sealed.domain.embedding_probe.cross_val_predict", return_value=np.zeros(20)):
            run_probes(cards, probes, embed_dim=2)

        assert seen_shapes[0] == (20, 2), f"Expected (20, 2), got {seen_shapes[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# T002 — Corrected card color labels (using real card fixture files)
# ─────────────────────────────────────────────────────────────────────────────

def _fixture_card(filename: str, embed_dim: int = 4) -> CardData:
    """Create a CardData from a real card fixture file."""
    text = load_card_fixture(filename)
    rng = np.random.default_rng(abs(hash(filename)) % (2**31))
    return CardData(
        name=filename,
        embedding=rng.standard_normal(embed_dim).astype(np.float32),
        text=text,
    )


# ─────────────────────────────────────────────────────────────────────────────
# T004 — compute_aux_labels() (20-element label vector)
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeAuxLabels:
    """T004: Verify compute_aux_labels() returns correct 20-element array.

    Label order (data-model.md):
      0: is_land
      1-6: card_color W/U/B/R/G/C
      7-12: pip_count W/U/B/R/G/C
      13: mana_value
      14-19: mana_produced W/U/B/R/G/C
    """

    def test_returns_ndarray_of_length_20(self):
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("lightning_bolt.txt")
        result = compute_aux_labels(text)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20,)

    def test_is_land_label_for_land(self):
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("forest.txt")
        result = compute_aux_labels(text)
        assert result[0] == pytest.approx(1.0), "forest is_land should be 1"

    def test_is_land_label_for_non_land(self):
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("lightning_bolt.txt")
        result = compute_aux_labels(text)
        assert result[0] == pytest.approx(0.0), "lightning bolt is_land should be 0"

    def test_card_color_uses_corrected_definition(self):
        # world_breaker is devoid → all color labels 0 except C=1
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("world_breaker.txt")
        result = compute_aux_labels(text)
        # indices 1-5: W/U/B/R/G = 0 (devoid)
        for i, color in enumerate(["W", "U", "B", "R", "G"], start=1):
            assert result[i] == pytest.approx(0.0), f"devoid: color_{color} should be 0"
        # index 6: C = 1 (colorless due to devoid)
        assert result[6] == pytest.approx(1.0), "devoid: color_C should be 1"

    def test_pip_counts_match_count_pips(self):
        from sealed.domain.embedding_probe import compute_aux_labels
        from sealed.domain.mana_scorer import count_pips
        text = load_card_fixture("norns_annex.txt")
        result = compute_aux_labels(text)
        pips = count_pips([text])
        colors = ["W", "U", "B", "R", "G", "C"]
        for i, color in enumerate(colors, start=7):
            expected = pips.counts.get(color, 0.0)
            assert result[i] == pytest.approx(expected), f"pip_count_{color} mismatch"

    def test_mana_value_matches_compute_mana_value(self):
        from sealed.domain.embedding_probe import compute_aux_labels
        from sealed.domain.mana_scorer import compute_mana_value
        text = load_card_fixture("norns_annex.txt")  # {3}{W/P}{W/P} → MV 5
        result = compute_aux_labels(text)
        # Extract mana cost line
        for line in text.splitlines():
            if line.strip().lower().startswith("mana cost:"):
                cost = line.strip()[len("mana cost:"):].strip()
                expected_mv = compute_mana_value(cost)
                break
        assert result[13] == pytest.approx(expected_mv)

    def test_mana_produced_for_land(self):
        # forest produces G
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("forest.txt")
        result = compute_aux_labels(text)
        # indices 14-19: W/U/B/R/G/C
        assert result[18] == pytest.approx(1.0), "forest should produce G"
        for i, color in zip([14, 15, 16, 17, 19], ["W", "U", "B", "R", "C"]):
            assert result[i] == pytest.approx(0.0), f"forest should not produce {color}"

    def test_mana_produced_for_mana_dork(self):
        # llanowar elves produces G
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("llanowar_elves.txt")
        result = compute_aux_labels(text)
        assert result[18] == pytest.approx(1.0), "llanowar elves should produce G"
        for i in [14, 15, 16, 17, 19]:
            assert result[i] == pytest.approx(0.0), f"llanowar should not produce index {i}"

    def test_mana_produced_for_artifact_mana_producer(self):
        # sol ring produces C
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("sol_ring.txt")
        result = compute_aux_labels(text)
        assert result[19] == pytest.approx(1.0), "sol ring should produce C"
        for i in [14, 15, 16, 17, 18]:
            assert result[i] == pytest.approx(0.0), f"sol ring should not produce index {i}"

    def test_non_mana_card_produces_nothing(self):
        # lightning bolt produces no mana
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("lightning_bolt.txt")
        result = compute_aux_labels(text)
        for i in range(14, 20):
            assert result[i] == pytest.approx(0.0), f"lightning bolt should not produce mana (index {i})"

    def test_devoid_does_not_affect_pip_counts(self):
        # world_breaker: {6}{G} + devoid → pip_G = 1.0 (pips unaffected by devoid)
        from sealed.domain.embedding_probe import compute_aux_labels
        text = load_card_fixture("world_breaker.txt")
        result = compute_aux_labels(text)
        # pip_count_G is index 11
        assert result[11] > 0.0, "devoid does not change pip counts"


# ─────────────────────────────────────────────────────────────────────────────
# T002 — Corrected card color labels (using real card fixture files)
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrectedCardColorLabels:
    """T002: Verify corrected card color label logic using real fixture files.

    Spec acceptance scenarios:
      1. {R} → R=1, all others=0
      2. {C} only → C=1, W/U/B/R/G=0
      3. {3} generic only → C=1
      4. land with no mana cost → C=1
      5. devoid + colored pips → C=1, W/U/B/R/G=0
      6. hybrid {R/G} → R=1, G=1, C=0
      7. Phyrexian {W/P} → W=1, C=0
    """

    def test_red_spell_color_correct(self):
        # Scenario 1: lightning bolt {R} → R=1, others=0
        cards = [_fixture_card("lightning_bolt.txt")]
        assert extract_card_color(cards, "R")[0] == pytest.approx(1.0)
        for color in ("W", "U", "B", "G", "C"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_colorless_pip_card_is_colorless(self):
        # Scenario 2: eldritch immunity {C} → C=1, colors=0
        cards = [_fixture_card("eldritch_immunity.txt")]
        assert extract_card_color(cards, "C")[0] == pytest.approx(1.0)
        for color in ("W", "U", "B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_generic_only_cost_is_colorless(self):
        # Scenario 3: abandoned sarcophagus {3} → C=1
        cards = [_fixture_card("abandoned_sarcophagus.txt")]
        assert extract_card_color(cards, "C")[0] == pytest.approx(1.0)
        for color in ("W", "U", "B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_land_no_mana_cost_is_colorless(self):
        # Scenario 4: forest (land, no mana cost line) → C=1
        cards = [_fixture_card("forest.txt")]
        assert extract_card_color(cards, "C")[0] == pytest.approx(1.0)
        for color in ("W", "U", "B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_devoid_card_is_colorless(self):
        # Scenario 5: world breaker {6}{G} + devoid → C=1, G=0
        cards = [_fixture_card("world_breaker.txt")]
        assert extract_card_color(cards, "C")[0] == pytest.approx(1.0)
        for color in ("W", "U", "B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_hybrid_card_color_correct(self):
        # Scenario 6: burning-tree emissary {R/G}{R/G} → R=1, G=1, C=0
        cards = [_fixture_card("burning_tree_emissary.txt")]
        assert extract_card_color(cards, "R")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "G")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "C")[0] == pytest.approx(0.0)
        for color in ("W", "U", "B"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_phyrexian_card_color_correct(self):
        # Scenario 7: norn's annex {3}{W/P}{W/P} → W=1, C=0
        cards = [_fixture_card("norns_annex.txt")]
        assert extract_card_color(cards, "W")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "C")[0] == pytest.approx(0.0)
        for color in ("U", "B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color

    def test_devoid_does_not_change_pip_counts(self):
        # World breaker has {G} in cost → pip_G > 0 even though color G = 0
        cards = [_fixture_card("world_breaker.txt")]
        # Pip counts use the raw mana cost, not affected by devoid
        pips_g = extract_pip_counts(cards, "G")[0]
        assert pips_g > 0.0, "devoid card should still have G pip count > 0"
        # But card color G must be 0 (devoid overrides)
        assert extract_card_color(cards, "G")[0] == pytest.approx(0.0)

    def test_multicolor_card_colors_correct(self):
        # absorb {W}{U}{U} → W=1, U=1, C=0
        cards = [_fixture_card("absorb.txt")]
        assert extract_card_color(cards, "W")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "U")[0] == pytest.approx(1.0)
        assert extract_card_color(cards, "C")[0] == pytest.approx(0.0)
        for color in ("B", "R", "G"):
            assert extract_card_color(cards, color)[0] == pytest.approx(0.0), color
