"""Unit tests for round-robin scheduling helpers."""

from __future__ import annotations

import pytest

from sealed.domain.round_robin_scheduling import row_major_pairings, split_evenly


class TestRowMajorPairings:
    def test_all_pairs_generated(self):
        a = [["a0"], ["a1"]]
        b = [["b0"], ["b1"], ["b2"]]
        pairs = row_major_pairings(a, b)
        assert len(pairs) == 6

    def test_row_major_order(self):
        a = [["a0"], ["a1"]]
        b = [["b0"], ["b1"]]
        pairs = row_major_pairings(a, b)
        assert pairs[0] == (["a0"], ["b0"])
        assert pairs[1] == (["a0"], ["b1"])
        assert pairs[2] == (["a1"], ["b0"])
        assert pairs[3] == (["a1"], ["b1"])

    def test_empty_inputs_yield_empty_pairs(self):
        assert row_major_pairings([], [["b"]]) == []
        assert row_major_pairings([["a"]], []) == []


class TestSplitEvenly:
    def test_empty_sequence_yields_empty_chunks(self):
        assert split_evenly([], 3) == [[], [], []]

    def test_even_divide(self):
        assert split_evenly([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_divide_front_loads_remainder(self):
        # 9 items across 2 workers → 5, 4
        chunks = split_evenly(list(range(9)), 2)
        assert chunks == [[0, 1, 2, 3, 4], [5, 6, 7, 8]]

    def test_n_greater_than_len(self):
        # 2 items across 5 workers → first 2 get one each, rest are empty
        chunks = split_evenly([1, 2], 5)
        assert chunks == [[1], [2], [], [], []]

    def test_preserves_order(self):
        chunks = split_evenly([10, 20, 30, 40, 50], 3)
        assert chunks == [[10, 20], [30, 40], [50]]

    def test_rejects_non_positive_n(self):
        with pytest.raises(ValueError):
            split_evenly([1, 2, 3], 0)
