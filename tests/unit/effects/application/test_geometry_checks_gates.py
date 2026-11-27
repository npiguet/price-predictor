"""Decodability, the baseline comparison and gate three."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from effects.application.geometry_checks import (
    CachedVectors,
    CacheGeometry,
    check_baseline_geometry,
    check_decodability,
    gate_three_verdict,
)
from effects.domain.evaluation_report import CheckStatus
from tests.unit.effects.selection_builders import read_cache


class TestDecodability:
    def test_it_skips_on_an_empty_cache(self, tmp_path):
        result = check_decodability(CachedVectors(), tmp_path / "wr.txt")
        assert result.status is CheckStatus.SKIPPED

    def test_it_skips_without_a_win_rate_table(self, cache):
        cards, abilities = cache
        result = check_decodability(
            read_cache(abilities, cards, surface="script"), cards / "absent.txt",
        )
        assert result.status is CheckStatus.SKIPPED
        assert "win-rate table" in result.detail

    def test_it_skips_when_too_few_cards_overlap(self, cache, tmp_path):
        cards, abilities = cache
        win_rates = tmp_path / "cards-win-rates.txt"
        win_rates.write_text("card_name;a\n", encoding="utf-8")
        result = check_decodability(
            read_cache(abilities, cards, surface="script"), win_rates,
        )
        assert result.status is CheckStatus.SKIPPED
        assert "too few" in result.detail

    def test_the_skip_counts_the_cards_that_have_a_win_rate(self, tmp_path):
        """The threshold is on labelled cards, so that is the count to print."""
        win_rates = tmp_path / "cards-win-rates.txt"
        win_rates.write_text(
            "card_name;a\n" + "".join(f"Storm Crow {i};0.5\n" for i in range(3)),
            encoding="utf-8",
        )
        by_card = {f"storm crow {i}": np.ones(4) for i in range(60)}
        result = check_decodability(CachedVectors(by_card=by_card), win_rates)
        assert result.status is CheckStatus.SKIPPED
        assert "only 3 of 60 cached cards have a win rate" in result.detail

    def test_the_cache_s_lowercase_names_join_the_canonical_win_rate_names(self, tmp_path):
        """A sidecar names its card in lowercase; the win-rate table in Forge's
        canonical case. A case-sensitive join labels no card at all, and every
        probe reads nan."""
        header = (
            "card_name;wins_when_played;wins_when_in_deck;losses_when_played;"
            "losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;"
            "shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;"
            "shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;"
            "shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;"
            "shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G"
        )
        rng = np.random.default_rng(3)
        by_card, rows = {}, [header]
        for i in range(80):
            score = float(rng.uniform(0.2, 0.8))
            by_card[f"storm crow {i}"] = np.array([score, *rng.normal(size=3)])
            cells = [f"Storm Crow {i}", "40", "60", "40", "60"] + [f"{score:.5f}"] * 18
            rows.append(";".join(cells))
        win_rates = tmp_path / "cards-win-rates.txt"
        win_rates.write_text("\n".join(rows) + "\n", encoding="utf-8")

        result = check_decodability(
            CachedVectors(by_card=by_card), win_rates,
            held_out_cards=tuple(f"storm crow {i}" for i in range(20)),
        )
        assert result.status is CheckStatus.REPORTED
        assert "score_play nan" not in result.detail, result.detail

class TestBaselineComparison:
    def test_it_skips_without_the_variant_cache(self, cache):
        cards, abilities = cache
        result = check_baseline_geometry(
            read_cache(abilities, cards, surface="script"), CachedVectors(), "no-state",
        )
        assert result.status is CheckStatus.SKIPPED
        assert "--variant no-state" in result.detail

    def test_it_compares_the_two_geometries(self):
        rng = np.random.default_rng(2)
        full = CachedVectors(by_text={
            f"t{i}": rng.normal(size=8) for i in range(50)
        })
        collapsed = CachedVectors(by_text={
            f"t{i}": np.ones(8) + 0.001 * rng.normal(size=8) for i in range(50)
        })
        result = check_baseline_geometry(full, collapsed, "no-state")
        assert result.status is CheckStatus.REPORTED
        assert result.values["full_cosine"] < result.values["no-state_cosine"]

    def test_each_cache_s_geometry_is_measured_once(self, monkeypatch):
        """Gate 3 measures the shipping cache; both comparisons reuse it."""
        from effects.application import geometry_checks
        from effects.application.geometry_checks import run_gate_three

        calls = []
        measure = geometry_checks.mean_pairwise_cosine

        def counted(vectors, **kwargs):
            calls.append(vectors.shape)
            return measure(vectors, **kwargs)

        monkeypatch.setattr(geometry_checks, "mean_pairwise_cosine", counted)
        rng = np.random.default_rng(2)
        full = CachedVectors(by_text={f"t{i}": rng.normal(size=8) for i in range(50)})
        baselines = [
            CachedVectors(by_text={f"t{i}": rng.normal(size=8) for i in range(50)})
            for _ in range(2)
        ]
        run_gate_three(full, Path("abilities"))
        for baseline in baselines:
            check_baseline_geometry(full, baseline, "no-state")
        assert len(calls) == 3


class TestGateThree:
    def test_a_spread_space_passes(self):
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(400, 64))
        result = gate_three_verdict(CacheGeometry.of(vectors))
        assert result.status is CheckStatus.PASS

    def test_a_collapsed_space_fails(self):
        """Every ability on top of every other passes every average-case metric."""
        rng = np.random.default_rng(0)
        vectors = np.ones((400, 64)) + 0.001 * rng.normal(size=(400, 64))
        result = gate_three_verdict(CacheGeometry.of(vectors))
        assert result.status is CheckStatus.FAIL
        assert "collapsed" in result.detail

    def test_a_one_dimensional_space_fails_on_the_top_component(self):
        direction = np.zeros(64)
        direction[0] = 1.0
        vectors = np.outer(np.linspace(-1, 1, 400), direction)
        result = gate_three_verdict(CacheGeometry.of(vectors))
        assert result.status is CheckStatus.FAIL
        assert "principal component" in result.detail

    def test_it_blocks_shipping(self):
        vectors = np.ones((100, 8))
        assert gate_three_verdict(CacheGeometry.of(vectors)).blocks

    def test_an_all_zero_space_fails(self):
        """The effect head once trained on an all-zero e: no variance at all is
        the most collapsed a space can be, not an unmeasurable one."""
        result = gate_three_verdict(CacheGeometry.of(np.zeros((100, 64))))
        assert result.status is CheckStatus.FAIL
        assert "principal component" in result.detail

    def test_too_few_vectors_to_measure_fails(self):
        result = gate_three_verdict(CacheGeometry.of(np.ones((1, 8))))
        assert result.status is CheckStatus.FAIL
        assert "not measured" in result.detail
