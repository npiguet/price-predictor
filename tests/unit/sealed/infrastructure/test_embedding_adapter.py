"""T002 — Unit tests for EmbeddingAdapter."""
from __future__ import annotations

import numpy as np
import pytest

from sealed.infrastructure.embedding_adapter import EmbeddingAdapter
from sealed.infrastructure.embedding_store import EmbeddingStore
from sealed.infrastructure.pool_loader import card_npz_path


def _make_adapter(tmp_path):
    store = EmbeddingStore()
    return EmbeddingAdapter(store, tmp_path), store


def _write_npz(path, embed_dim=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embedding=np.ones(embed_dim, dtype=np.float32))


def _write_txt(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# ── get_embedding ─────────────────────────────────────────────────────────────

class TestGetEmbedding:
    def test_returns_array_from_store(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        npz_path = card_npz_path(tmp_path, "Grizzly Bears")
        _write_npz(npz_path, embed_dim=8)

        result = adapter.get_embedding("Grizzly Bears")
        assert result.shape == (8,)
        np.testing.assert_array_almost_equal(result, np.ones(8, dtype=np.float32))

    def test_accumulates_timing(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        _write_npz(card_npz_path(tmp_path, "A"), embed_dim=4)
        _write_npz(card_npz_path(tmp_path, "B"), embed_dim=4)

        adapter.get_embedding("A")
        adapter.get_embedding("B")
        assert adapter.total_load_s > 0.0

    def test_reset_timing_clears_counter(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        _write_npz(card_npz_path(tmp_path, "A"), embed_dim=4)
        adapter.get_embedding("A")
        adapter.reset_timing()
        assert adapter.total_load_s == 0.0


# ── is_land ───────────────────────────────────────────────────────────────────

class TestIsLand:
    def test_land_card_returns_true(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        txt_path = card_npz_path(tmp_path, "island").with_suffix(".txt")
        _write_txt(txt_path, "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n")
        assert adapter.is_land("island") is True

    def test_non_land_returns_false(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        txt_path = card_npz_path(tmp_path, "lightning bolt").with_suffix(".txt")
        _write_txt(txt_path, "name: lightning bolt\nmana cost: {R}\ntypes: instant\n")
        assert adapter.is_land("lightning bolt") is False

    def test_missing_txt_returns_false(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        # No .txt file written
        assert adapter.is_land("nonexistent card") is False

    def test_caches_result(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        txt_path = card_npz_path(tmp_path, "island").with_suffix(".txt")
        _write_txt(txt_path, "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n")

        result1 = adapter.is_land("island")
        # Delete the file — second call must use cache
        txt_path.unlink()
        result2 = adapter.is_land("island")

        assert result1 is True
        assert result2 is True


# ── get_card_text ─────────────────────────────────────────────────────────────

class TestGetCardText:
    def test_returns_full_text(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        txt_path = card_npz_path(tmp_path, "island").with_suffix(".txt")
        content = "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n"
        _write_txt(txt_path, content)

        result = adapter.get_card_text("island")
        assert result == content

    def test_missing_txt_returns_empty_string(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        result = adapter.get_card_text("nonexistent")
        assert result == ""

    def test_caches_result(self, tmp_path):
        adapter, _ = _make_adapter(tmp_path)
        txt_path = card_npz_path(tmp_path, "island").with_suffix(".txt")
        content = "name: island\ntypes: basic land island\nactivated[1]: {T}: add {U}\n"
        _write_txt(txt_path, content)

        result1 = adapter.get_card_text("island")
        # Delete the file — second call must use cache
        txt_path.unlink()
        result2 = adapter.get_card_text("island")

        assert result1 == content
        assert result2 == content
