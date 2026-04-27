"""Resolve card names to converted-card files on disk.

Centralises the sanitization, prefix-fallback lookup, and basic-land sets
that previously lived (duplicated) in ``match_data_loader`` and
``evaluate_scorer``. Both ``.txt`` (converted card scripts) and ``.npz``
(card embeddings) live in the same letter-keyed directory layout, so a
single locator handles both.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.card_filenames import (
    sanitize_card_name,
)
from sealed.infrastructure.card_name_corrections import FILENAME_CORRECTIONS

BASIC_LAND_NAMES: frozenset[str] = frozenset(
    {"plains", "island", "swamp", "mountain", "forest"}
)


class ConvertedCardLocator:
    """Looks up converted card files (`.txt` and `.npz`) by card name.

    Card names from match outcomes / pool files use the canonical Forge
    spelling (e.g. ``"Lim-Dûl's Vault"``); on-disk filenames are ASCII,
    lowercase, punctuation-stripped (``"lim_duls_vault.npz"``). Double-faced,
    split, and adventure cards have filenames like ``"frontface_backface.npz"``
    but the source files reference only the front face — those are resolved
    by prefix search.
    """

    def __init__(self, cards_path: Path) -> None:
        self._cards_path = cards_path
        self._letter_index: dict[str, dict[str, Path]] = {}

    def text_path(self, card_name: str) -> Path | None:
        return self._find_file(card_name, ".txt")

    def embedding_path(self, card_name: str) -> Path | None:
        return self._find_file(card_name, ".npz")

    def load_text(self, card_name: str) -> ConvertedCardText | None:
        path = self.text_path(card_name)
        return ConvertedCardText.from_file(path) if path else None

    def load_embedding(self, card_name: str) -> np.ndarray | None:
        path = self.embedding_path(card_name)
        if path is None:
            return None
        with np.load(path) as f:
            return f["embedding"].copy()

    def expected_path(self, card_name: str, ext: str) -> Path:
        """Return the expected exact-match path (used for error messages)."""
        filename, first_letter = self._split_filename(card_name)
        return self._cards_path / first_letter / f"{filename}{ext}"

    def _find_file(self, card_name: str, ext: str) -> Path | None:
        filename, first_letter = self._split_filename(card_name)
        index = self._index_for(first_letter)

        exact = index.get(f"{filename}{ext}")
        if exact is not None:
            return exact

        prefix = filename + "_"
        for fname, path in index.items():
            if fname.endswith(ext) and fname.startswith(prefix):
                return path
        return None

    def _index_for(self, letter: str) -> dict[str, Path]:
        cached = self._letter_index.get(letter)
        if cached is not None:
            return cached
        letter_dir = self._cards_path / letter
        index: dict[str, Path] = {}
        if letter_dir.is_dir():
            for entry in letter_dir.iterdir():
                index[entry.name] = entry
        self._letter_index[letter] = index
        return index

    def _split_filename(self, card_name: str) -> tuple[str, str]:
        resolved = sanitize_card_name(card_name, FILENAME_CORRECTIONS)
        if "/" in resolved:
            first_letter, filename = resolved.split("/", 1)
        else:
            filename = resolved
            first_letter = filename[0] if filename else "_"
        return filename, first_letter
