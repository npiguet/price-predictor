"""Unit tests for greedy deck search and result aggregation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from sealed.application.evaluate_scorer import (
    greedy_deck_search,
    compute_basic_lands,
    aggregate_results,
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
        # Deck should have exactly 23 non-land cards
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
        # Should complete without infinite loop
        deck = greedy_deck_search(model, pool_names, pool_embeddings)
        assert len(deck) == 23


class TestBasicLandComputation:
    def test_fills_to_40_total(self):
        """Basic lands should fill remaining slots to 40 total cards."""
        # 23 non-land cards → 17 basic lands needed
        nonland_texts = {f"card_{i}": f"name: card_{i}\nmana cost: {{R}}\ntypes: creature" for i in range(23)}
        lands = compute_basic_lands(nonland_texts)
        assert sum(lands.values()) == 17

    def test_proportional_to_color_pips(self):
        """Land distribution should be proportional to color pips."""
        # 23 cards all with {R} → 17 Mountains
        nonland_texts = {f"card_{i}": f"name: card_{i}\nmana cost: {{R}}\ntypes: creature" for i in range(23)}
        lands = compute_basic_lands(nonland_texts)
        assert lands.get("Mountain", 0) == 17

    def test_multicolor_distributes_proportionally(self):
        """Multi-color decks distribute lands proportionally."""
        # Mix of red and green
        texts = {}
        for i in range(12):
            texts[f"red_{i}"] = f"name: red_{i}\nmana cost: {{R}}\ntypes: creature"
        for i in range(11):
            texts[f"green_{i}"] = f"name: green_{i}\nmana cost: {{G}}\ntypes: creature"
        lands = compute_basic_lands(texts)
        total = sum(lands.values())
        assert total == 17
        assert "Mountain" in lands
        assert "Forest" in lands


class TestResultAggregation:
    def test_aggregate_single_file(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("2;1\n2;0\n1;2\n", encoding="utf-8")
        result = aggregate_results([outcomes])
        assert result["pools_evaluated"] == 3
        assert result["wins_scorer"] == 2 + 2 + 1
        assert result["wins_forge"] == 1 + 0 + 2
        assert result["total_games"] == (2+1) + (2+0) + (1+2)

    def test_aggregate_multiple_files(self, tmp_path):
        f1 = tmp_path / "out1.txt"
        f2 = tmp_path / "out2.txt"
        f1.write_text("2;1\n", encoding="utf-8")
        f2.write_text("0;2\n", encoding="utf-8")
        result = aggregate_results([f1, f2])
        assert result["pools_evaluated"] == 2
        assert result["wins_scorer"] == 2 + 0
        assert result["wins_forge"] == 1 + 2

    def test_win_rate_computed(self, tmp_path):
        outcomes = tmp_path / "outcomes.txt"
        outcomes.write_text("2;0\n2;0\n", encoding="utf-8")
        result = aggregate_results([outcomes])
        assert result["win_rate"] == pytest.approx(1.0)
