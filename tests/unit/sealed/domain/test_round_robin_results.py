"""Unit tests for RoundRobinResults aggregation and reporting."""

from __future__ import annotations

import pytest

from sealed.domain.round_robin_results import RoundRobinResults, aggregate_results


class TestResultAggregation:
    def test_single_pool_a_wins(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("2;0\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=1)
        assert result.n_pools == 1
        assert result.n_matches == 1
        assert result.total_games == 2
        assert result.a_win_rates == pytest.approx([1.0])
        assert result.b_win_rates == pytest.approx([0.0])
        assert result.pool_deltas == pytest.approx([1.0])
        assert result.a_aggregate_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)
        assert result.a_match_win_rates == pytest.approx([1.0])
        assert result.b_match_win_rates == pytest.approx([0.0])
        assert result.pool_match_deltas == pytest.approx([1.0])
        assert result.a_aggregate_match_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_match_win_rate == pytest.approx(0.0)

    def test_single_pool_b_wins(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("0;2\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=1)
        assert result.a_win_rates == pytest.approx([0.0])
        assert result.b_win_rates == pytest.approx([1.0])
        assert result.pool_deltas == pytest.approx([-1.0])
        assert result.a_aggregate_win_rate == pytest.approx(0.0)
        assert result.b_aggregate_win_rate == pytest.approx(1.0)
        assert result.a_aggregate_match_win_rate == pytest.approx(0.0)
        assert result.b_aggregate_match_win_rate == pytest.approx(1.0)

    def test_aggregate_multiple_files(self, tmp_path):
        f1 = tmp_path / "out1.txt"
        f2 = tmp_path / "out2.txt"
        f1.write_text("2;0\n2;0\n", encoding="utf-8")
        f2.write_text("2;0\n2;0\n", encoding="utf-8")
        result = aggregate_results([f1, f2], n_pools=2)
        assert result.n_matches == 4
        assert result.a_aggregate_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)
        assert result.a_aggregate_match_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_match_win_rate == pytest.approx(0.0)

    def test_round_robin_per_pool_delta(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("2;0\n2;0\n0;2\n0;2\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=2)

        assert result.a_win_rates[0] == pytest.approx(1.0)
        assert result.a_win_rates[1] == pytest.approx(0.0)
        assert result.b_win_rates[0] == pytest.approx(0.5)
        assert result.b_win_rates[1] == pytest.approx(0.5)
        assert result.pool_deltas[0] == pytest.approx(0.5)
        assert result.pool_deltas[1] == pytest.approx(-0.5)

        assert result.a_match_win_rates[0] == pytest.approx(1.0)
        assert result.a_match_win_rates[1] == pytest.approx(0.0)
        assert result.b_match_win_rates[0] == pytest.approx(0.5)
        assert result.b_match_win_rates[1] == pytest.approx(0.5)
        assert result.pool_match_deltas[0] == pytest.approx(0.5)
        assert result.pool_match_deltas[1] == pytest.approx(-0.5)

    def test_round_robin_aggregate_win_rates(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        lines = "\n".join(["2;0"] * 9) + "\n"
        outcomes.write_text(lines, encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=3)
        assert result.n_matches == 9
        assert result.total_games == 18
        assert result.a_aggregate_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)
        assert result.a_aggregate_match_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_match_win_rate == pytest.approx(0.0)

    def test_match_rate_amplifies_game_rate(self, tmp_path):
        """A wins 4-3 in every match: 4/7 ≈ 57.1% per game, 100% per match."""
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("4;3\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=1)
        assert result.a_aggregate_win_rate == pytest.approx(4 / 7)
        assert result.b_aggregate_win_rate == pytest.approx(3 / 7)
        assert result.a_aggregate_match_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_match_win_rate == pytest.approx(0.0)

    def test_match_rate_vs_game_rate_differ_on_close_matches(self, tmp_path):
        """2x2 round-robin, A wins every match 4-3.
        Per-game: 16/28 = 57.1%, per-match: 4/4 = 100%."""
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("4;3\n4;3\n4;3\n4;3\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=2)
        assert result.a_aggregate_win_rate == pytest.approx(16 / 28)
        assert result.a_aggregate_match_win_rate == pytest.approx(1.0)

    def test_empty_files_return_zeros(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.txt"
        result = aggregate_results([nonexistent], n_pools=3)
        assert result.n_matches == 0
        assert result.total_games == 0
        assert result.a_aggregate_win_rate == pytest.approx(0.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)
        assert result.a_aggregate_match_win_rate == pytest.approx(0.0)
        assert result.b_aggregate_match_win_rate == pytest.approx(0.0)
        assert result.a_win_rates == []
        assert result.b_win_rates == []
        assert result.a_match_win_rates == []
        assert result.b_match_win_rates == []


class TestFormatReport:
    def test_includes_pool_count_and_aggregate(self):
        result = RoundRobinResults(
            n_pools=2,
            n_matches=4,
            total_games=8,
            a_win_rates=[1.0, 0.5],
            b_win_rates=[0.0, 0.5],
            pool_deltas=[1.0, 0.0],
            a_aggregate_win_rate=0.75,
            b_aggregate_win_rate=0.25,
            a_match_win_rates=[1.0, 0.5],
            b_match_win_rates=[0.0, 0.5],
            pool_match_deltas=[1.0, 0.0],
            a_aggregate_match_win_rate=0.75,
            b_aggregate_match_win_rate=0.25,
        )
        report = result.format_report()
        assert "Pools: 2" in report
        assert "Matches: 4" in report
        assert "75.0%" in report
        assert "25.0%" in report
        assert "Per-game" in report
        assert "Per-match" in report

    def test_distinct_game_and_match_rates_both_render(self):
        """A wins every game 4-3: per-game 57.1%, per-match 100%."""
        result = RoundRobinResults(
            n_pools=1,
            n_matches=4,
            total_games=28,
            a_win_rates=[4 / 7],
            b_win_rates=[3 / 7],
            pool_deltas=[1 / 7],
            a_aggregate_win_rate=4 / 7,
            b_aggregate_win_rate=3 / 7,
            a_match_win_rates=[1.0],
            b_match_win_rates=[0.0],
            pool_match_deltas=[1.0],
            a_aggregate_match_win_rate=1.0,
            b_aggregate_match_win_rate=0.0,
        )
        report = result.format_report()
        assert "57.1%" in report
        assert "100.0%" in report
        assert "Mean delta" in report
        assert "per-game" in report
        assert "per-match" in report

    def test_empty_result_omits_per_pool_section(self):
        result = RoundRobinResults.empty(n_pools=3, n_matches=0, total_games=0)
        report = result.format_report()
        assert "Per-pool comparison" not in report
