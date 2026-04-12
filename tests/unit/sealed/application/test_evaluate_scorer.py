"""Unit tests for greedy deck search, result aggregation, and round-robin match writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sealed.application.evaluate_scorer import (
    greedy_deck_search,
    compute_basic_lands,
    aggregate_results,
    _write_round_robin_matches,
)
from sealed.domain.scorer_model import SetTransformerScorer


def _make_model():
    model = SetTransformerScorer(d_model=544, n_layers=1, n_heads=4, n_seeds=4, d_ff=544, mlp_hidden=64)
    model.eval()
    return model


def _make_pool_embeddings(n_cards=60):
    """Create synthetic card embeddings for a pool."""
    names = [f"card_{i}" for i in range(n_cards)]
    embeddings = {name: np.random.randn(544).astype(np.float32) for name in names}
    return names, embeddings


class TestGreedyDeckSearch:
    def test_returns_23_nonland_cards(self):
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60)
        deck = greedy_deck_search(model, pool_names, pool_embeddings)
        assert len(deck) == 23

    def test_deck_cards_are_from_pool(self):
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60)
        deck = greedy_deck_search(model, pool_names, pool_embeddings)
        for card in deck:
            assert card in pool_names

    def test_stops_when_no_improvement(self):
        """Greedy search should converge (finite iterations)."""
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(40)
        deck = greedy_deck_search(model, pool_names, pool_embeddings)
        assert len(deck) == 23


class TestBasicLandComputation:
    def test_fills_to_40_total(self):
        """Basic lands should fill remaining slots to 40 total cards."""
        nonland_texts = [
            f"name: card_{i}\nmana cost: {{R}}\ntypes: creature" for i in range(23)
        ]
        lands = compute_basic_lands(nonland_texts)
        assert sum(lands.values()) == 17

    def test_counts_duplicate_card_slots(self):
        """Duplicate card slots must each contribute to n_basics and pip counts."""
        # 23 nonland slots but only 1 unique text → must still produce 17 lands.
        text = "name: card\nmana cost: {R}\ntypes: creature"
        nonland_texts = [text] * 23
        lands = compute_basic_lands(nonland_texts)
        assert sum(lands.values()) == 17
        assert lands.get("Mountain", 0) == 17

    def test_proportional_to_color_pips(self):
        """Mono-color decks put all basics in that color."""
        nonland_texts = [
            f"name: card_{i}\nmana cost: {{R}}\ntypes: creature" for i in range(23)
        ]
        lands = compute_basic_lands(nonland_texts)
        assert lands.get("Mountain", 0) == 17

    def test_multicolor_distributes_proportionally(self):
        """Multi-color decks distribute lands proportionally."""
        texts = []
        for i in range(12):
            texts.append(f"name: red_{i}\nmana cost: {{R}}\ntypes: creature")
        for i in range(11):
            texts.append(f"name: green_{i}\nmana cost: {{G}}\ntypes: creature")
        lands = compute_basic_lands(texts)
        assert sum(lands.values()) == 17
        assert "Mountain" in lands
        assert "Forest" in lands

    def test_min_two_per_active_color(self):
        """A splash color with low pip count still gets at least 2 basics."""
        # 22 white-only cards + 1 single-black card → B has 1 pip vs W's 22.
        # Without the min-2 rule, B would round to 0 or 1.
        texts = [
            f"name: w_{i}\nmana cost: {{W}}\ntypes: creature" for i in range(22)
        ]
        texts.append("name: splash\nmana cost: {1}{B}\ntypes: creature")
        lands = compute_basic_lands(texts)
        assert sum(lands.values()) == 17
        assert lands.get("Swamp", 0) >= 2
        assert lands.get("Plains", 0) >= 2


class TestResultAggregation:
    def test_single_pool_a_wins(self, tmp_path):
        """Single pool (1 match): A wins all games → win rates reflect that."""
        outcomes = tmp_path / "outcomes.txt"
        # n_pools=1 → 1 match (A0 vs B0)
        outcomes.write_text("2;0\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=1)
        assert result["n_pools"] == 1
        assert result["n_matches"] == 1
        assert result["total_games"] == 2
        assert result["a_win_rates"] == pytest.approx([1.0])
        assert result["b_win_rates"] == pytest.approx([0.0])
        assert result["pool_deltas"] == pytest.approx([1.0])
        assert result["a_aggregate_win_rate"] == pytest.approx(1.0)
        assert result["b_aggregate_win_rate"] == pytest.approx(0.0)

    def test_single_pool_b_wins(self, tmp_path):
        """Single pool: B wins all → opposite win rates."""
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("0;2\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=1)
        assert result["a_win_rates"] == pytest.approx([0.0])
        assert result["b_win_rates"] == pytest.approx([1.0])
        assert result["pool_deltas"] == pytest.approx([-1.0])
        assert result["a_aggregate_win_rate"] == pytest.approx(0.0)
        assert result["b_aggregate_win_rate"] == pytest.approx(1.0)

    def test_aggregate_multiple_files(self, tmp_path):
        """Outcomes spread across multiple files are combined correctly."""
        # n_pools=1, 2 matches → but wait, 1 pool means 1×1=1 match
        # Use n_pools=2 with 4 matches split across 2 files
        f1 = tmp_path / "out1.txt"
        f2 = tmp_path / "out2.txt"
        # n_pools=2: k=0 (A0vB0), k=1 (A0vB1) in file1; k=2 (A1vB0), k=3 (A1vB1) in file2
        f1.write_text("2;0\n2;0\n", encoding="utf-8")
        f2.write_text("2;0\n2;0\n", encoding="utf-8")
        result = aggregate_results([f1, f2], n_pools=2)
        assert result["n_matches"] == 4
        assert result["a_aggregate_win_rate"] == pytest.approx(1.0)
        assert result["b_aggregate_win_rate"] == pytest.approx(0.0)

    def test_round_robin_per_pool_delta(self, tmp_path):
        """Per-pool delta is A_i win rate minus B_i win rate from the same pool."""
        outcomes = tmp_path / "outcomes.txt"
        # n_pools=2 → 4 matches (row-major: A0vB0, A0vB1, A1vB0, A1vB1)
        # A0 wins both its matches; A1 loses both
        outcomes.write_text("2;0\n2;0\n0;2\n0;2\n", encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=2)

        # A0: 4 wins / 4 games = 1.0; A1: 0 wins / 4 games = 0.0
        assert result["a_win_rates"][0] == pytest.approx(1.0)
        assert result["a_win_rates"][1] == pytest.approx(0.0)

        # B0: plays k=0 (A0) and k=2 (A1) → wb=0+2=2 over 4 games = 0.5
        # B1: plays k=1 (A0) and k=3 (A1) → wb=0+2=2 over 4 games = 0.5
        assert result["b_win_rates"][0] == pytest.approx(0.5)
        assert result["b_win_rates"][1] == pytest.approx(0.5)

        # pool_deltas: A_i - B_i
        assert result["pool_deltas"][0] == pytest.approx(0.5)   # 1.0 - 0.5
        assert result["pool_deltas"][1] == pytest.approx(-0.5)  # 0.0 - 0.5

    def test_round_robin_aggregate_win_rates(self, tmp_path):
        """Aggregate win rates are total wins / total games across all N² matches."""
        outcomes = tmp_path / "outcomes.txt"
        # n_pools=3 → 9 matches; all A wins (2;0)
        lines = "\n".join(["2;0"] * 9) + "\n"
        outcomes.write_text(lines, encoding="utf-8")
        result = aggregate_results([outcomes], n_pools=3)
        assert result["n_matches"] == 9
        assert result["total_games"] == 18
        assert result["a_aggregate_win_rate"] == pytest.approx(1.0)
        assert result["b_aggregate_win_rate"] == pytest.approx(0.0)

    def test_empty_files_return_zeros(self, tmp_path):
        """Missing or empty outcome files produce zero-filled result."""
        nonexistent = tmp_path / "does_not_exist.txt"
        result = aggregate_results([nonexistent], n_pools=3)
        assert result["n_matches"] == 0
        assert result["total_games"] == 0
        assert result["a_aggregate_win_rate"] == pytest.approx(0.0)
        assert result["b_aggregate_win_rate"] == pytest.approx(0.0)
        assert result["a_win_rates"] == []
        assert result["b_win_rates"] == []


class TestWriteRoundRobinMatches:
    def _make_deck(self, prefix: str, n: int = 5) -> list[str]:
        return [f"{prefix}_card_{i}" for i in range(n)]

    def test_generates_n_squared_lines(self, tmp_path):
        """3 A-decks × 3 B-decks = 9 total match lines."""
        a_decks = [self._make_deck(f"a{i}") for i in range(3)]
        b_decks = [self._make_deck(f"b{i}") for i in range(3)]
        worker_files = _write_round_robin_matches(a_decks, b_decks, n_workers=1, work_dir=tmp_path)
        total_lines = sum(
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
            for f in worker_files
        )
        assert total_lines == 9

    def test_row_major_order(self, tmp_path):
        """First line is A0 vs B0, second is A0 vs B1."""
        a_decks = [["a0"], ["a1"]]
        b_decks = [["b0"], ["b1"]]
        worker_files = _write_round_robin_matches(a_decks, b_decks, n_workers=1, work_dir=tmp_path)
        lines = [
            ln for ln in worker_files[0].read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        # Row-major: A0vB0, A0vB1, A1vB0, A1vB1
        assert lines[0] == "a0;b0"
        assert lines[1] == "a0;b1"
        assert lines[2] == "a1;b0"
        assert lines[3] == "a1;b1"

    def test_splits_into_per_worker_files(self, tmp_path):
        """9 matches across 2 workers are split ~evenly."""
        a_decks = [self._make_deck(f"a{i}") for i in range(3)]
        b_decks = [self._make_deck(f"b{i}") for i in range(3)]
        worker_files = _write_round_robin_matches(a_decks, b_decks, n_workers=2, work_dir=tmp_path)
        assert len(worker_files) == 2
        counts = [
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
            for f in worker_files
        ]
        assert sum(counts) == 9
        # With 9 lines and 2 workers: chunk_size=4, remainder=1 → first gets 5, second gets 4
        assert counts[0] == 5
        assert counts[1] == 4
