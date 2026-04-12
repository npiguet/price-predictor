"""Resolve card names to converted-card files on disk.

Centralises the sanitization, prefix-fallback lookup, and basic-land sets
that previously lived (duplicated) in ``match_data_loader`` and
``evaluate_scorer``. Both ``.txt`` (converted card scripts) and ``.npz``
(card embeddings) live in the same letter-keyed directory layout, so a
single locator handles both.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np

from sealed.infrastructure.card_name_corrections import FILENAME_CORRECTIONS

BASIC_LAND_NAMES: frozenset[str] = frozenset(
    {"plains", "island", "swamp", "mountain", "forest"}
)
BASIC_LAND_TITLE_NAMES: frozenset[str] = frozenset(
    {"Plains", "Island", "Swamp", "Mountain", "Forest"}
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

    def text_path(self, card_name: str) -> Path | None:
        return self._find_file(card_name, ".txt")

    def embedding_path(self, card_name: str) -> Path | None:
        return self._find_file(card_name, ".npz")

    def load_text(self, card_name: str) -> str | None:
        path = self.text_path(card_name)
        return path.read_text(encoding="utf-8") if path else None

    def load_embedding(self, card_name: str) -> np.ndarray | None:
        path = self.embedding_path(card_name)
        if path is None:
            return None
        return np.load(path)["embedding"]

    def expected_path(self, card_name: str, ext: str) -> Path:
        """Return the expected exact-match path (used for error messages)."""
        filename, first_letter = self._split_filename(card_name)
        return self._cards_path / first_letter / f"{filename}{ext}"

    def _find_file(self, card_name: str, ext: str) -> Path | None:
        filename, first_letter = self._split_filename(card_name)
        letter_dir = self._cards_path / first_letter

        exact = letter_dir / f"{filename}{ext}"
        if exact.exists():
            return exact

        if letter_dir.is_dir():
            prefix = filename + "_"
            for candidate in letter_dir.iterdir():
                if candidate.suffix == ext and candidate.stem.startswith(prefix):
                    return candidate
        return None

    def _split_filename(self, card_name: str) -> tuple[str, str]:
        resolved = sanitize_card_name(card_name)
        if "/" in resolved:
            first_letter, filename = resolved.split("/", 1)
        else:
            filename = resolved
            first_letter = filename[0] if filename else "_"
        return filename, first_letter


def sanitize_card_name(name: str) -> str:
    """Convert a Forge card name to its on-disk filename.

    NFKD-decomposes accents (â → a), lowercases, strips punctuation, and
    applies known filename corrections from ``FILENAME_CORRECTIONS``.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    sanitized = (
        ascii_name.lower()
        .replace(" // ", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace(",", "")
        .replace(":", "")
        .replace("!", "")
        .replace('"', "")
        .replace("&", "")
        .replace("+", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("/", "")
    )
    return FILENAME_CORRECTIONS.get(sanitized, sanitized)
