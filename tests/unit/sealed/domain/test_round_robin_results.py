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

    def test_single_pool_b_wins(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("0;2\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=1)
        assert result.a_win_rates == pytest.approx([0.0])
        assert result.b_win_rates == pytest.approx([1.0])
        assert result.pool_deltas == pytest.approx([-1.0])
        assert result.a_aggregate_win_rate == pytest.approx(0.0)
        assert result.b_aggregate_win_rate == pytest.approx(1.0)

    def test_aggregate_multiple_files(self, tmp_path):
        f1 = tmp_path / "out1.txt"
        f2 = tmp_path / "out2.txt"
        f1.write_text("2;0\n2;0\n", encoding="utf-8")
        f2.write_text("2;0\n2;0\n", encoding="utf-8")
        result = aggregate_results([f1, f2], n_pools=2)
        assert result.n_matches == 4
        assert result.a_aggregate_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)

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

    def test_round_robin_aggregate_win_rates(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        lines = "\n".join(["2;0"] * 9) + "\n"
        outcomes.write_text(lines, encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=3)
        assert result.n_matches == 9
        assert result.total_games == 18
        assert result.a_aggregate_win_rate == pytest.approx(1.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)

    def test_empty_files_return_zeros(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.txt"
        result = aggregate_results([nonexistent], n_pools=3)
        assert result.n_matches == 0
        assert result.total_games == 0
        assert result.a_aggregate_win_rate == pytest.approx(0.0)
        assert result.b_aggregate_win_rate == pytest.approx(0.0)
        assert result.a_win_rates == []
        assert result.b_win_rates == []


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
        )
        report = result.format_report()
        assert "Pools: 2" in report
        assert "Matches: 4" in report
        assert "75.0%" in report
        assert "25.0%" in report

    def test_empty_result_omits_per_pool_section(self):
        result = RoundRobinResults.empty(n_pools=3, n_matches=0, total_games=0)
        report = result.format_report()
        assert "Per-pool comparison" not in report
