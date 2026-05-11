"""Unit tests for ConvertedCardLocator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from price_predictor.infrastructure.card_filenames import sanitize_card_name
from sealed.infrastructure.converted_card_locator import (
    BASIC_LAND_NAMES,
    ConvertedCardLocator,
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

    def test_collapses_underscore_runs(self):
        # "&" is stripped but leaves the surrounding spaces -> "__" -> "_".
        assert sanitize_card_name("Anchovy & Banana Pizza") == "anchovy_banana_pizza"
        assert sanitize_card_name("Don & Leo, Problem Solvers") == "don_leo_problem_solvers"


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
        assert loc.load_text("Lightning Bolt").text.startswith("name:")

    def test_missing_text_returns_none(self, tmp_path):
        loc = ConvertedCardLocator(tmp_path)
        assert loc.load_text("Nonexistent") is None


class TestRebalancedAndFaceFallback:
    def test_alchemy_rebalanced_resolved_in_rebalanced_dir(self, tmp_path):
        # "A-Akki Ronin" -> cardsfolder/rebalanced/a-akki_ronin.txt
        reb = tmp_path / "rebalanced"
        reb.mkdir()
        (reb / "a-akki_ronin.txt").write_text("name:Akki Ronin\n")
        loc = ConvertedCardLocator(tmp_path)
        path = loc.text_path("A-Akki Ronin")
        assert path is not None
        assert path.name == "a-akki_ronin.txt"

    def test_meld_combined_name_falls_back_to_front_face(self, tmp_path):
        # Forge stores meld cards under the front-face name only.
        b_dir = tmp_path / "b"
        b_dir.mkdir()
        (b_dir / "bruna_the_fading_light.txt").write_text("name:Bruna\n")
        loc = ConvertedCardLocator(tmp_path)
        path = loc.text_path(
            "Bruna, the Fading Light // Brisela, Voice of Nightmares",
        )
        assert path is not None
        assert path.name == "bruna_the_fading_light.txt"

    def test_typo_in_dfc_filename_resolved_via_front_face_prefix(self, tmp_path):
        # Forge filename has a typo ("...minsdstinger"); the front-face
        # prefix search still finds it.
        a_dir = tmp_path / "a"
        a_dir.mkdir()
        (a_dir / "aetherblade_agent_gitaxian_minsdstinger.txt").write_text("x")
        loc = ConvertedCardLocator(tmp_path)
        path = loc.text_path("Aetherblade Agent // Gitaxian Mindstinger")
        assert path is not None
        assert path.name == "aetherblade_agent_gitaxian_minsdstinger.txt"

    def test_amp_card_collapsed_underscores(self, tmp_path):
        a_dir = tmp_path / "a"
        a_dir.mkdir()
        (a_dir / "anchovy_banana_pizza.txt").write_text("x")
        loc = ConvertedCardLocator(tmp_path)
        assert loc.text_path("Anchovy & Banana Pizza") is not None


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
