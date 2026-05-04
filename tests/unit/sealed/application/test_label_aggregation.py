"""Unit tests for label aggregation (FR-010, FR-012)."""

from __future__ import annotations

from sealed.application.train_encoder import (
    _aggregate_counts,
    _build_winnability_map,
)
from sealed.infrastructure.cards_played_reader import CardsPlayedRow


def _row(
    *,
    cp_a: list[str],
    cp_b: list[str],
    cnp_a: list[str],
    cnp_b: list[str],
    winner: str = "A",
    starter: str = "A",
) -> CardsPlayedRow:
    return CardsPlayedRow(
        timestamp="2026-05-03T14:22:01Z",
        run_id="run",
        set_code="BLB",
        method_a="forge-best",
        method_b="forge-3sub",
        cards_played_a=cp_a,
        cards_played_b=cp_b,
        cards_not_played_a=cnp_a,
        cards_not_played_b=cnp_b,
        winner=winner,
        starter=starter,
    )


class TestAggregateCounts:
    def test_winning_side_only(self):
        rows = [
            _row(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[], winner="A"),
            _row(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[], winner="B"),
        ]
        counts = _aggregate_counts(rows)
        # LB seen twice as the winning side; GB seen twice as the winning side.
        assert counts["LB"] == (1, 1)  # only the row where A won counts
        assert counts["GB"] == (1, 1)

    def test_in_deck_includes_not_played(self):
        rows = [
            _row(cp_a=["LB"], cp_b=[], cnp_a=["GB"], cnp_b=[], winner="A"),
        ]
        counts = _aggregate_counts(rows)
        assert counts["LB"] == (1, 1)
        assert counts["GB"] == (0, 1)

    def test_card_counted_once_per_game(self):
        # Multiplicities don't increment counts within a single game.
        rows = [
            _row(cp_a=["LB", "LB", "LB"], cp_b=[], cnp_a=[], cnp_b=[], winner="A"),
        ]
        counts = _aggregate_counts(rows)
        assert counts["LB"] == (1, 1)

    def test_losing_side_excluded(self):
        rows = [
            _row(cp_a=["LB"], cp_b=["GB"], cnp_a=[], cnp_b=[], winner="A"),
        ]
        counts = _aggregate_counts(rows)
        assert "GB" not in counts
        assert counts["LB"] == (1, 1)


class TestBuildWinnabilityMap:
    def test_drops_zero_in_deck(self):
        # _aggregate_counts won't produce wins_when_in_deck == 0; but we
        # double-check the FR-012 filter at the map-building level.
        counts = {"LB": (1, 1), "GB": (0, 0)}
        labels = _build_winnability_map(counts, shrinkage_k=0.0)
        assert "LB" in labels
        assert "GB" not in labels

    def test_label_uses_shrinkage(self):
        counts = {"LB": (1, 1)}
        with_k0 = _build_winnability_map(counts, shrinkage_k=0.0)
        with_k20 = _build_winnability_map(counts, shrinkage_k=20.0)
        assert with_k0["LB"].shrunk_label == 1.0
        assert with_k20["LB"].shrunk_label < 1.0  # pulled toward 0.5
