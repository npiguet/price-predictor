"""Unit tests for ConvertedCardLocator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    BASIC_LAND_TITLE_NAMES,
    ConvertedCardLocator,
    sanitize_card_name,
)


def _write_npz(letter_dir: Path, name: str, emb: np.ndarray) -> Path:
    letter_dir.mkdir(parents=True, exist_ok=True)
    path = letter_dir / f"{name}.npz"
    np.savez_compressed(path, embedding=emb)
    return path


class TestSanitizeCardName:
    def test_lowercases_and_underscores(self):
        assert sanitize_card_name("Lightning Bolt") == "lightning_bolt"

    def test_strips_apostrophes(self):
        assert sanitize_card_name("Urza's Saga") == "urzas_saga"

    def test_strips_accents(self):
        assert sanitize_card_name("Dand\u00e2n") == "dandan"

    def test_double_slash_becomes_underscore(self):
        assert sanitize_card_name("Glassworks // Shattered Yard") == "glassworks_shattered_yard"

    def test_strips_punctuation(self):
        assert sanitize_card_name("Borborygmos, Enraged!") == "borborygmos_enraged"


class TestEmbeddingPath:
    def test_exact_match(self, tmp_path):
        emb = np.random.randn(8).astype(np.float32)
        _write_npz(tmp_path / "l", "lightning_bolt", emb)
        loc = ConvertedCardLocator(tmp_path)
        assert loc.embedding_path("Lightning Bolt") is not None

    def test_returns_none_when_missing(self, tmp_path):
        loc = ConvertedCardLocator(tmp_path)
        assert loc.embedding_path("Nonexistent Card") is None

    def test_double_faced_resolved_by_prefix(self, tmp_path):
        emb = np.random.randn(8).astype(np.float32)
        _write_npz(tmp_path / "m", "mosswood_dreadknight_dread_whispers", emb)
        loc = ConvertedCardLocator(tmp_path)
        path = loc.embedding_path("Mosswood Dreadknight")
        assert path is not None
        assert path.stem == "mosswood_dreadknight_dread_whispers"

    def test_accented_name_resolved(self, tmp_path):
        emb = np.random.randn(8).astype(np.float32)
        _write_npz(tmp_path / "d", "dandan", emb)
        loc = ConvertedCardLocator(tmp_path)
        assert loc.embedding_path("Dand\u00e2n") is not None

    def test_exact_match_preferred_over_prefix(self, tmp_path):
        emb1 = np.random.randn(8).astype(np.float32)
        emb2 = np.random.randn(8).astype(np.float32)
        d_dir = tmp_path / "f"
        _write_npz(d_dir, "fire", emb1)
        _write_npz(d_dir, "fire_ice", emb2)
        loc = ConvertedCardLocator(tmp_path)
        assert loc.embedding_path("Fire").stem == "fire"


class TestTextPath:
    def test_text_file_lookup(self, tmp_path):
        l_dir = tmp_path / "l"
        l_dir.mkdir()
        (l_dir / "lightning_bolt.txt").write_text("name:Lightning Bolt\n")
        loc = ConvertedCardLocator(tmp_path)
        assert loc.load_text("Lightning Bolt").startswith("name:")

    def test_missing_text_returns_none(self, tmp_path):
        loc = ConvertedCardLocator(tmp_path)
        assert loc.load_text("Nonexistent") is None


class TestLoadEmbedding:
    def test_returns_numpy_array(self, tmp_path):
        emb = np.arange(8, dtype=np.float32)
        _write_npz(tmp_path / "l", "lightning_bolt", emb)
        loc = ConvertedCardLocator(tmp_path)
        loaded = loc.load_embedding("Lightning Bolt")
        np.testing.assert_array_equal(loaded, emb)

    def test_missing_returns_none(self, tmp_path):
        loc = ConvertedCardLocator(tmp_path)
        assert loc.load_embedding("Nonexistent") is None


class TestBasicLandConstants:
    def test_lowercase_set_contents(self):
        assert BASIC_LAND_NAMES == {"plains", "island", "swamp", "mountain", "forest"}

    def test_title_case_set_contents(self):
        assert BASIC_LAND_TITLE_NAMES == {"Plains", "Island", "Swamp", "Mountain", "Forest"}
