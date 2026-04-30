"""Unit tests for GreedyDeckBuilder restart-state initialization."""

from __future__ import annotations

import random

import numpy as np
import pytest

from sealed.domain.card_embedding_layout import (
    COLOR_FLAGS,
    FEATURE_COUNT,
    IS_LAND,
    total_dim,
)
from sealed.domain.greedy_deck_builder import (
    COLOR_PAIRS,
    COLOR_PAIRS_STRATEGY,
    NONLAND_DECK_SIZE,
    GreedyDeckBuilder,
    card_colors,
)
from sealed.domain.scorer_model import ScorerConfig, SetTransformerScorer

D_MODEL = total_dim(256)
_IS_LAND_OFFSET = D_MODEL - FEATURE_COUNT + IS_LAND
_COLOR_FLAGS_OFFSET = D_MODEL - FEATURE_COUNT + COLOR_FLAGS.start


def _spell_embedding(*colors: str) -> np.ndarray:
    emb = np.zeros(D_MODEL, dtype=np.float32)
    for c in colors:
        emb[_COLOR_FLAGS_OFFSET + "WUBRG".index(c)] = 1.0
    return emb


def _land_embedding() -> np.ndarray:
    emb = np.zeros(D_MODEL, dtype=np.float32)
    emb[_IS_LAND_OFFSET] = 1.0
    return emb


def _make_builder(
    pool_embeddings: dict[str, np.ndarray], restarts: int | str,
) -> GreedyDeckBuilder:
    model = SetTransformerScorer(ScorerConfig(
        n_layers=1, n_heads=4, n_seeds=4, d_ff=544, mlp_hidden=64,
    ))
    model.eval()
    return GreedyDeckBuilder(model, pool_embeddings, restarts=restarts)


class TestRestartsValidation:
    def test_unknown_string_rejected(self):
        with pytest.raises(ValueError, match="restarts string"):
            _make_builder({}, restarts="random-init")

    def test_color_pairs_string_accepted(self):
        builder = _make_builder({}, restarts=COLOR_PAIRS_STRATEGY)
        assert builder._restarts == COLOR_PAIRS_STRATEGY

    def test_integer_clamped_to_one(self):
        builder = _make_builder({}, restarts=0)
        assert builder._restarts == 1

    def test_positive_integer_passed_through(self):
        builder = _make_builder({}, restarts=4)
        assert builder._restarts == 4


class TestColorPairStates:
    """`_color_pair_states` builds one state per pair with a viable on-color
    seed deck. The color filter applies only to the initial 23 spells."""

    def _build_pool(
        self,
        per_pair_count: int = 25,
        n_lands: int = 5,
    ) -> tuple[dict[str, np.ndarray], list[str], list[int], list[int]]:
        """Build a pool with ``per_pair_count`` spells in each of the 10 pairs.

        Each pair gets distinct mono-color spells so they don't double-count
        across pairs. Lands are colorless (qualify for every pair).
        """
        pool_embeddings: dict[str, np.ndarray] = {}
        pool_names: list[str] = []
        spell_pool: list[int] = []

        # Mono-color spells: each color contributes ``per_pair_count`` spells.
        # A WU-pair filter accepts both W and U mono-colors → 2 *
        # per_pair_count eligible. With per_pair_count=25 → 50 eligible per
        # pair, well above the 23 threshold.
        for color in "WUBRG":
            for j in range(per_pair_count):
                name = f"spell_{color}_{j}"
                pool_embeddings[name] = _spell_embedding(color)
                spell_pool.append(len(pool_names))
                pool_names.append(name)

        land_pool: list[int] = []
        for j in range(n_lands):
            name = f"land_{j}"
            pool_embeddings[name] = _land_embedding()
            land_pool.append(len(pool_names))
            pool_names.append(name)

        return pool_embeddings, pool_names, spell_pool, land_pool

    def test_produces_one_state_per_pair_when_all_viable(self):
        embs, names, spells, lands = self._build_pool(per_pair_count=25)
        builder = _make_builder(embs, restarts=COLOR_PAIRS_STRATEGY)
        random.seed(0)

        states = builder._init_restart_states(names, spells, lands)

        assert len(states) == 10

    def test_each_state_initial_deck_is_on_color(self):
        embs, names, spells, lands = self._build_pool(per_pair_count=25)
        builder = _make_builder(embs, restarts=COLOR_PAIRS_STRATEGY)
        random.seed(0)

        states = builder._init_restart_states(names, spells, lands)

        # The pair order matches COLOR_PAIRS, which the implementation iterates
        # in that order for any pair with enough eligible spells.
        for state, pair in zip(states, COLOR_PAIRS):
            assert len(state.deck_spells) == NONLAND_DECK_SIZE
            for idx in state.deck_spells:
                colors = card_colors(embs[names[idx]])
                assert colors.issubset(pair), (
                    f"deck spell {names[idx]} with colors {colors} "
                    f"not subset of pair {pair}"
                )

    def test_spells_remaining_includes_off_color(self):
        """The color filter applies only to the initial 23 — every other spell
        in the pool is in spells_remaining, even off-color cards."""
        embs, names, spells, lands = self._build_pool(per_pair_count=25)
        builder = _make_builder(embs, restarts=COLOR_PAIRS_STRATEGY)
        random.seed(0)

        states = builder._init_restart_states(names, spells, lands)

        for state, pair in zip(states, COLOR_PAIRS):
            chosen = set(state.deck_spells)
            expected_remaining = [i for i in spells if i not in chosen]
            assert sorted(state.spells_remaining) == sorted(expected_remaining)
            # At least one off-pair spell is in spells_remaining (the pool has
            # 5 colors; any 2-color pair leaves 3 colors off-pair).
            off_pair_count = sum(
                1 for i in state.spells_remaining
                if not card_colors(embs[names[i]]).issubset(pair)
            )
            assert off_pair_count > 0

    def test_lands_remaining_is_full_land_pool(self):
        embs, names, spells, lands = self._build_pool(n_lands=7)
        builder = _make_builder(embs, restarts=COLOR_PAIRS_STRATEGY)
        random.seed(0)

        states = builder._init_restart_states(names, spells, lands)

        for state in states:
            assert state.lands_remaining == lands
            assert state.deck_lands == []

    def test_pair_skipped_when_under_23_eligible_spells(self):
        """If only one mono-color has ≥ 23 cards, the only viable pairs are
        those including that color (4 of 10)."""
        embs: dict[str, np.ndarray] = {}
        names: list[str] = []
        spells: list[int] = []
        # 30 W spells, 5 each of U/B/R/G
        for j in range(30):
            name = f"spell_W_{j}"
            embs[name] = _spell_embedding("W")
            spells.append(len(names))
            names.append(name)
        for color in "UBRG":
            for j in range(5):
                name = f"spell_{color}_{j}"
                embs[name] = _spell_embedding(color)
                spells.append(len(names))
                names.append(name)

        builder = _make_builder(embs, restarts=COLOR_PAIRS_STRATEGY)
        random.seed(0)
        states = builder._init_restart_states(names, spells, [])

        # Only WU, WB, WR, WG can seat 23 W-or-X spells (30 W alone suffices).
        assert len(states) == 4

    def test_falls_back_to_one_random_restart_when_no_pair_viable(self):
        """If no pair has 23 eligible spells, fall back to a single random
        restart instead of returning an empty state list."""
        embs: dict[str, np.ndarray] = {}
        names: list[str] = []
        spells: list[int] = []
        # All 5 colors but only 6 spells per color → no pair has 23 eligible.
        # Total 30 spells (>= NONLAND_DECK_SIZE), so the fallback can seat one.
        for color in "WUBRG":
            for j in range(6):
                name = f"spell_{color}_{j}"
                embs[name] = _spell_embedding(color)
                spells.append(len(names))
                names.append(name)

        builder = _make_builder(embs, restarts=COLOR_PAIRS_STRATEGY)
        random.seed(0)
        states = builder._init_restart_states(names, spells, [])

        assert len(states) == 1
        assert len(states[0].deck_spells) == NONLAND_DECK_SIZE
