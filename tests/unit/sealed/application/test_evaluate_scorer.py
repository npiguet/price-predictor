"""Unit tests for greedy deck builder and round-robin match writing."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from sealed.application.evaluate_scorer import (
    EvaluateScorerUseCase,
    write_round_robin_matches,
)
from sealed.domain.card_embedding_layout import total_dim
from sealed.domain.greedy_deck_builder import GreedyDeckBuilder
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer

D_MODEL = total_dim(256)


def _make_model():
    model = SetTransformerScorer(ScorerConfig(
        n_layers=1, n_heads=4, n_seeds=4, d_ff=544, mlp_hidden=64,
    ))
    model.eval()
    return model


def _make_pool_embeddings(n_cards=60):
    """Create synthetic card embeddings for a pool."""
    names = [f"card_{i}" for i in range(n_cards)]
    embeddings = {name: np.random.randn(D_MODEL).astype(np.float32) for name in names}
    return names, embeddings


class TestGreedyDeckBuilder:
    def test_returns_23_nonland_cards(self):
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        assert len(deck) == 23

    def test_deck_cards_are_from_pool(self):
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        for card in deck:
            assert card in pool_names

    def test_stops_when_no_improvement(self):
        """Greedy search should converge (finite iterations)."""
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(40)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        assert len(deck) == 23


class TestWriteRoundRobinMatches:
    def _make_deck(self, prefix: str, n: int = 5) -> list[str]:
        return [f"{prefix}_card_{i}" for i in range(n)]

    def test_generates_n_squared_lines(self, tmp_path):
        """3 A-decks × 3 B-decks = 9 total match lines."""
        a_decks = [self._make_deck(f"a{i}") for i in range(3)]
        b_decks = [self._make_deck(f"b{i}") for i in range(3)]
        worker_files = write_round_robin_matches(
            a_decks, b_decks, n_workers=1, work_dir=tmp_path,
        )
        total_lines = sum(
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
            for f in worker_files
        )
        assert total_lines == 9

    def test_row_major_order(self, tmp_path):
        """First line is A0 vs B0, second is A0 vs B1."""
        a_decks = [["a0"], ["a1"]]
        b_decks = [["b0"], ["b1"]]
        worker_files = write_round_robin_matches(
            a_decks, b_decks, n_workers=1, work_dir=tmp_path,
        )
        lines = [
            ln for ln in worker_files[0].read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        assert lines[0] == "a0;b0"
        assert lines[1] == "a0;b1"
        assert lines[2] == "a1;b0"
        assert lines[3] == "a1;b1"

    def test_splits_into_per_worker_files(self, tmp_path):
        """9 matches across 2 workers are split ~evenly."""
        a_decks = [self._make_deck(f"a{i}") for i in range(3)]
        b_decks = [self._make_deck(f"b{i}") for i in range(3)]
        worker_files = write_round_robin_matches(
            a_decks, b_decks, n_workers=2, work_dir=tmp_path,
        )
        assert len(worker_files) == 2
        counts = [
            sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
            for f in worker_files
        ]
        assert sum(counts) == 9
        assert counts[0] == 5
        assert counts[1] == 4


class TestResolveSetCode:
    def test_explicit_set_code_returned_as_is(self):
        use_case = EvaluateScorerUseCase()
        assert use_case._resolve_set_code("RVR") == "RVR"

    def test_explicit_set_code_does_not_call_eligible(self):
        use_case = EvaluateScorerUseCase()
        with patch(
            "sealed.application.evaluate_scorer.eligible_sealed_sets",
        ) as mock_eligible:
            use_case._resolve_set_code("DMU")
        mock_eligible.assert_not_called()

    def test_none_picks_random_from_eligible(self):
        use_case = EvaluateScorerUseCase()
        with patch(
            "sealed.application.evaluate_scorer.eligible_sealed_sets",
            return_value=["A", "B", "C"],
        ) as mock_eligible, patch(
            "sealed.application.evaluate_scorer.random.choice",
            return_value="B",
        ) as mock_choice:
            result = use_case._resolve_set_code(None)
        mock_eligible.assert_called_once()
        mock_choice.assert_called_once_with(["A", "B", "C"])
        assert result == "B"

    def test_none_with_empty_eligible_raises(self):
        use_case = EvaluateScorerUseCase()
        with patch(
            "sealed.application.evaluate_scorer.eligible_sealed_sets",
            return_value=[],
        ):
            try:
                use_case._resolve_set_code(None)
            except RuntimeError as e:
                assert "eligible" in str(e).lower()
            else:
                raise AssertionError("Expected RuntimeError")
