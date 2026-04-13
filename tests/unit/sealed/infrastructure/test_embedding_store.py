"""Unit tests for EmbeddingStore infrastructure class."""

from __future__ import annotations

import os

import numpy as np

from sealed.infrastructure.embedding_store import EmbeddingStore


class TestEmbeddingStoreSave:
    def test_save_creates_npz_file(self, tmp_path):
        store = EmbeddingStore()
        path = tmp_path / "card.npz"
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        store.save(path, embedding)
        assert path.exists()

    def test_save_no_intermediate_tmp_file_after_completion(self, tmp_path):
        store = EmbeddingStore()
        path = tmp_path / "card.npz"
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        store.save(path, embedding)
        tmp = tmp_path / "card.tmp.npz"
        assert not tmp.exists()

    def test_save_creates_parent_directories(self, tmp_path):
        store = EmbeddingStore()
        path = tmp_path / "subdir" / "deeper" / "card.npz"
        embedding = np.array([1.0, 2.0], dtype=np.float32)
        store.save(path, embedding)
        assert path.exists()

    def test_save_overwrites_existing_file(self, tmp_path):
        store = EmbeddingStore()
        path = tmp_path / "card.npz"
        store.save(path, np.array([1.0], dtype=np.float32))
        store.save(path, np.array([99.0], dtype=np.float32))
        loaded = np.load(path)["embedding"]
        np.testing.assert_array_almost_equal(loaded, [99.0])


class TestEmbeddingStoreAtomicity:
    def test_atomic_write_uses_temp_then_rename(self, tmp_path, monkeypatch):
        """Verify the temp file pattern: save writes {stem}.tmp.npz then os.replace."""
        replaced_pairs = []
        original_replace = os.replace

        def mock_replace(src, dst):
            replaced_pairs.append((src, dst))
            original_replace(src, dst)

        monkeypatch.setattr(os, "replace", mock_replace)

        store = EmbeddingStore()
        path = tmp_path / "card.npz"
        store.save(path, np.array([1.0], dtype=np.float32))

        assert len(replaced_pairs) == 1
        src, dst = replaced_pairs[0]
        assert src.endswith("card.tmp.npz")
        assert dst == str(path)
