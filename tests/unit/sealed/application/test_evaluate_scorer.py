"""Unit tests for greedy deck builder and round-robin match writing."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch

from price_predictor.domain.entities import TransformerConfig
from price_predictor.infrastructure.transformer_model import CardPriceTransformerModel
from sealed.application.evaluate_scorer import (
    EvaluateScorerUseCase,
    write_round_robin_matches,
)
from sealed.domain.card_embedding_layout import total_dim
from sealed.domain.greedy_deck_builder import GreedyDeckBuilder
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer
from sealed.infrastructure.scorer_store import ScorerStore

D_MODEL = total_dim(256)


def _make_model():
    model = SetTransformerScorer(ScorerConfig(
        n_layers=1, n_heads=4, n_seeds=4, d_ff=544, mlp_hidden=64,
    ))
    model.eval()
    return model


def _make_pool_embeddings(n_cards=60, n_lands=0):
    """Create synthetic card embeddings for a pool.

    The IS_LAND deterministic-feature flag is set explicitly: spells get 0,
    lands get 1. ``n_lands`` of the ``n_cards`` total are marked as lands.
    """
    from sealed.domain.card_embedding_layout import FEATURE_COUNT, IS_LAND
    is_land_offset = D_MODEL - FEATURE_COUNT + IS_LAND
    names = [f"card_{i}" for i in range(n_cards)]
    embeddings: dict[str, np.ndarray] = {}
    for i, name in enumerate(names):
        emb = np.random.randn(D_MODEL).astype(np.float32)
        emb[is_land_offset] = 1.0 if i < n_lands else 0.0
        embeddings[name] = emb
    return names, embeddings


class TestGreedyDeckBuilder:
    def test_returns_23_spells_when_pool_is_all_spells(self):
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60, n_lands=0)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        assert len(deck) == 23

    def test_deck_cards_are_from_pool(self):
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60, n_lands=0)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        for card in deck:
            assert card in pool_names

    def test_stops_when_no_improvement(self):
        """Greedy search should converge (finite iterations)."""
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(40, n_lands=0)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        assert len(deck) == 23

    def test_pool_with_lands_can_yield_more_than_23_picks(self):
        """When the pool has lands, the greedy may pick some of them in
        addition to 23 spells — total picks then exceed 23 by however many
        lands the scorer included."""
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60, n_lands=5)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        # Strict invariant: at least 23 picks, no more than 23 + n_lands.
        assert 23 <= len(deck) <= 23 + 5

    def test_deck_holds_23_spell_invariant(self):
        """With a mixed pool, the picked deck always contains exactly 23
        non-land cards, regardless of how many lands were added."""
        from sealed.domain.card_embedding_layout import is_land_embedding
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60, n_lands=5)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        spell_count = sum(
            1 for name in deck if not is_land_embedding(pool_embeddings[name])
        )
        assert spell_count == 23

    def test_falls_back_when_too_few_spells(self):
        """If the pool can't supply 23 spells, the builder returns the whole
        pool unchanged rather than crashing."""
        model = _make_model()
        # 30 cards, 25 of them lands → only 5 spells, far short of 23.
        pool_names, pool_embeddings = _make_pool_embeddings(30, n_lands=25)
        deck = GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        assert sorted(deck) == sorted(pool_names)

    def test_restarts_runs_search_multiple_times_and_keeps_best(self):
        """With restarts > 1, the builder runs the search N times from
        independent random inits and returns the best-scoring deck across
        runs. We verify N runs by counting the random shuffle calls."""
        from unittest.mock import patch
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60, n_lands=0)

        with patch("sealed.domain.greedy_deck_builder.random.shuffle") as mock_shuffle:
            GreedyDeckBuilder(
                model, pool_embeddings, restarts=4,
            ).build(pool_names)
        assert mock_shuffle.call_count == 4

    def test_restarts_default_is_one(self):
        """Backward-compat: default behavior is unchanged (single run)."""
        from unittest.mock import patch
        model = _make_model()
        pool_names, pool_embeddings = _make_pool_embeddings(60, n_lands=0)

        with patch("sealed.domain.greedy_deck_builder.random.shuffle") as mock_shuffle:
            GreedyDeckBuilder(model, pool_embeddings).build(pool_names)
        assert mock_shuffle.call_count == 1


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


class TestEvaluateScorerToleratesPhaseBCheckpoints:
    """T040 (US3): the evaluate-scorer loader ignores Phase B-only keys
    (`encoder_state_dict`, `encoder_config`, `train_config`) so the same code
    path scores Phase A and Phase B checkpoints identically (SC-003)."""

    def test_load_phase_b_checkpoint(self, tmp_path):
        scorer_cfg = ScorerConfig(
            n_layers=1, n_heads=4, n_seeds=4, d_ff=544, mlp_hidden=64,
        )
        model = SetTransformerScorer(scorer_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        encoder_cfg = TransformerConfig(
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
            max_seq_len=8, vocab_size=32, dropout=0.0,
        )
        encoder = CardPriceTransformerModel(encoder_cfg)
        path = tmp_path / "phase_b.pt"
        ScorerStore().save_checkpoint(
            model, optimizer, epoch=3, best_val_accuracy=0.7,
            config=scorer_cfg, path=path,
            encoder_state_dict=encoder.state_dict(),
            encoder_config=asdict(encoder_cfg),
            train_config={"lr": 1e-5, "embedding_lr": 1e-7},
        )
        loaded_model = EvaluateScorerUseCase()._load_model(path)
        assert isinstance(loaded_model, SetTransformerScorer)
        assert loaded_model.config.n_layers == 1
        assert loaded_model.config.n_heads == 4
