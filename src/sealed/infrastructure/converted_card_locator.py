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
        self._embedding_cache: dict[str, np.ndarray | None] = {}
        self._text_cache: dict[str, ConvertedCardText | None] = {}

    def text_path(self, card_name: str) -> Path | None:
        return self._find_file(card_name, ".txt")

    def embedding_path(self, card_name: str) -> Path | None:
        return self._find_file(card_name, ".npz")

    def load_text(self, card_name: str) -> ConvertedCardText | None:
        """Load and parse a card's ``.txt``, memoized by name.

        Like ``load_embedding``, the parse result is cached so repeated lookups
        — e.g. ``compute_basic_lands`` re-reading the chosen cards' mana costs
        on every deck in ``pick-decks`` / ``build-decks`` — become a dict hit
        instead of re-opening and re-parsing the ``.txt``. ``ConvertedCardText``
        is treated read-only by all callers, so the cached instance is shared
        directly (no copy needed).
        """
        if card_name not in self._text_cache:
            path = self.text_path(card_name)
            self._text_cache[card_name] = (
                ConvertedCardText.from_file(path) if path else None
            )
        return self._text_cache[card_name]

    def load_embedding(self, card_name: str) -> np.ndarray | None:
        """Load a card's ``.npz`` embedding, memoized by name.

        Cards recur heavily across sealed pools, so the per-name cache turns
        repeated lookups into a dict hit instead of re-opening and decompressing
        the ``.npz`` every time (the dominant cost in ``pick-decks`` /
        ``build-decks`` / ``evaluate-scorer``, which reload per pool). A
        defensive copy is returned on each call so callers keep the original
        "fresh array per call" contract and cannot mutate the cached row. Same
        memoize-by-name technique as ``MatchDataLoader._intern`` and
        ``train_picker._prepare_pools``.
        """
        if card_name not in self._embedding_cache:
            path = self.embedding_path(card_name)
            if path is None:
                self._embedding_cache[card_name] = None
            else:
                with np.load(path) as f:
                    self._embedding_cache[card_name] = f["embedding"].copy()
        cached = self._embedding_cache[card_name]
        return None if cached is None else cached.copy()

    def expected_path(self, card_name: str, ext: str) -> Path:
        """Return the expected exact-match path (used for error messages)."""
        filename, first_letter = self._split_filename(card_name)
        return self._cards_path / first_letter / f"{filename}{ext}"

    def _find_file(self, card_name: str, ext: str) -> Path | None:
        hit = self._find_exact_or_prefix(card_name, ext)
        if hit is not None:
            return hit
        # ``Front // Back`` fallback: Forge writes meld cards under the
        # front-face name only ("Bruna, the Fading Light // Brisela, Voice
        # of Nightmares" -> ``bruna_the_fading_light.txt``), and some
        # double-faced filenames carry typos ("...minsdstinger") that the
        # front-face prefix search still resolves. Retry with the front face.
        if " // " in card_name:
            front_face = card_name.split(" // ", 1)[0].strip()
            if front_face:
                return self._find_exact_or_prefix(front_face, ext)
        return None

    def _find_exact_or_prefix(self, card_name: str, ext: str) -> Path | None:
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

    _REBALANCED_DIR = "rebalanced"

    def _split_filename(self, card_name: str) -> tuple[str, str]:
        # Alchemy/Arena rebalanced cards are named "A-Akki Ronin" and live
        # under cardsfolder/rebalanced/ as "a-akki_ronin.txt" — the literal
        # "a-" prefix is kept (the general sanitizer would turn it into
        # "a_"), and the directory is "rebalanced" rather than first-letter.
        if card_name[:2].upper() == "A-" and len(card_name) > 2:
            stem = sanitize_card_name(card_name[2:], FILENAME_CORRECTIONS)
            return f"a-{stem}", self._REBALANCED_DIR

        resolved = sanitize_card_name(card_name, FILENAME_CORRECTIONS)
        if "/" in resolved:
            first_letter, filename = resolved.split("/", 1)
        else:
            filename = resolved
            first_letter = filename[0] if filename else "_"
        return filename, first_letter
