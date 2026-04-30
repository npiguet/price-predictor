"""Unit tests for the embedding-layout domain helpers."""

from __future__ import annotations

import numpy as np

from sealed.domain.card_embedding_layout import (
    COLOR_FLAGS,
    FEATURE_COUNT,
    card_colors,
    is_land_embedding,
    total_dim,
)

D_MODEL = total_dim(256)
_COLOR_FLAGS_OFFSET = D_MODEL - FEATURE_COUNT + COLOR_FLAGS.start


def _embedding_with_colors(*colors: str) -> np.ndarray:
    """Build a synthetic embedding whose COLOR_FLAGS slots match ``colors``."""
    emb = np.zeros(D_MODEL, dtype=np.float32)
    for c in colors:
        idx = "WUBRG".index(c)
        emb[_COLOR_FLAGS_OFFSET + idx] = 1.0
    return emb


class TestCardColors:
    def test_colorless_returns_empty_set(self):
        emb = np.zeros(D_MODEL, dtype=np.float32)
        assert card_colors(emb) == frozenset()

    def test_mono_color_returns_single_color(self):
        emb = _embedding_with_colors("W")
        assert card_colors(emb) == frozenset({"W"})

    def test_each_color_isolated(self):
        for c in "WUBRG":
            emb = _embedding_with_colors(c)
            assert card_colors(emb) == frozenset({c}), c

    def test_hybrid_two_colors(self):
        emb = _embedding_with_colors("R", "G")
        assert card_colors(emb) == frozenset({"R", "G"})

    def test_three_color_card(self):
        emb = _embedding_with_colors("W", "U", "B")
        assert card_colors(emb) == frozenset({"W", "U", "B"})

    def test_threshold_below_half_is_off(self):
        emb = _embedding_with_colors("W")
        emb[_COLOR_FLAGS_OFFSET] = 0.4  # below 0.5 threshold
        assert card_colors(emb) == frozenset()

    def test_threshold_above_half_is_on(self):
        emb = np.zeros(D_MODEL, dtype=np.float32)
        emb[_COLOR_FLAGS_OFFSET + 1] = 0.6  # U slot, above threshold
        assert card_colors(emb) == frozenset({"U"})

    def test_returns_frozenset(self):
        emb = _embedding_with_colors("R")
        assert isinstance(card_colors(emb), frozenset)


class TestIsLandEmbedding:
    """Pre-existing helper retested here for proximity."""

    def test_zero_means_not_land(self):
        emb = np.zeros(D_MODEL, dtype=np.float32)
        assert not is_land_embedding(emb)

    def test_one_means_land(self):
        emb = np.zeros(D_MODEL, dtype=np.float32)
        emb[D_MODEL - FEATURE_COUNT] = 1.0
        assert is_land_embedding(emb)
