"""Resolve card names to converted-card files on disk.

Centralises the sanitization, prefix-fallback lookup, and basic-land sets
that previously lived (duplicated) in ``match_data_loader`` and
``evaluate_scorer``. Both ``.txt`` (converted card scripts) and ``.npz``
(card embeddings) live in the same letter-keyed directory layout, so a
single locator handles both.

There are three converted source trees and they do not share a layout:
``cardsfolder`` is letter-keyed, while ``tokenscripts`` and ``variant-scripts``
are flat. One filename therefore names different files in different trees —
Ajani's Pridemate is ``cardsfolder/a/ajanis_pridemate.txt`` on one side and
``tokenscripts/ajanis_pridemate.txt`` on the other — so a locator is bound to a
tree at construction. ``cardsfolder`` is the default, which is what every
existing sealed caller means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.card_filenames import (
    sanitize_card_name,
)
from sealed.domain.deck import BASIC_LAND_NAMES
from sealed.infrastructure.card_name_corrections import FILENAME_CORRECTIONS

# Re-exported for callers that historically imported it from here.
__all__ = [
    "BASIC_LAND_NAMES",
    "SOURCE_TREE_LAYOUTS",
    "ConvertedCardLocator",
]

_LETTER_KEYED = "letter-keyed"
_FLAT = "flat"

#: How each converted source tree lays its files out.
SOURCE_TREE_LAYOUTS: dict[str, str] = {
    "cardsfolder": _LETTER_KEYED,
    "tokenscripts": _FLAT,
    "variant-scripts": _FLAT,
}


class ConvertedCardLocator:
    """Looks up converted card files (`.txt` and `.npz`) by card name.

    Card names from match outcomes / pool files use the canonical Forge
    spelling (e.g. ``"Lim-Dûl's Vault"``); on-disk filenames are ASCII,
    lowercase, punctuation-stripped (``"lim_duls_vault.npz"``). Double-faced,
    split, and adventure cards have filenames like ``"frontface_backface.npz"``
    but the source files reference only the front face — those are resolved
    by prefix search.

    ``tree`` names which converted source tree ``cards_path`` holds, which fixes
    the directory layout to search and prefixes the paths
    :meth:`script_file` returns.
    """

    def __init__(self, cards_path: Path, *, tree: str = "cardsfolder") -> None:
        if tree not in SOURCE_TREE_LAYOUTS:
            raise ValueError(
                f"unknown converted source tree {tree!r}; "
                f"known trees: {sorted(SOURCE_TREE_LAYOUTS)}"
            )
        self._cards_path = cards_path
        self._tree = tree
        self._layout = SOURCE_TREE_LAYOUTS[tree]
        self._letter_index: dict[str, dict[str, Path]] = {}
        self._embedding_cache: dict[str, np.ndarray | None] = {}
        self._text_cache: dict[str, ConvertedCardText | None] = {}

    @property
    def tree(self) -> str:
        return self._tree

    def script_file(self, card_name: str) -> str | None:
        """The tree-prefixed path a provenance key names, or None if unresolved.

        ``cardsfolder/a/ajanis_pridemate.txt`` or
        ``tokenscripts/ajanis_pridemate.txt`` — written with forward slashes on
        every platform, because the Java writer and this reader must produce the
        same string or the join fails loudly.
        """
        path = self.text_path(card_name)
        if path is None:
            return None
        relative = path.relative_to(self._cards_path)
        return f"{self._tree}/{relative.as_posix()}"

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
        filename, subdirectory = self._split_filename(card_name)
        base = self._cards_path / subdirectory if subdirectory else self._cards_path
        return base / f"{filename}{ext}"

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
        filename, subdirectory = self._split_filename(card_name)
        index = self._index_for(subdirectory)

        exact = index.get(f"{filename}{ext}")
        if exact is not None:
            return exact

        prefix = filename + "_"
        for fname, path in index.items():
            if fname.endswith(ext) and fname.startswith(prefix):
                if self._prefix_hit_agrees(path, filename):
                    return path
        return None

    def _prefix_hit_agrees(self, path: Path, filename: str) -> bool:
        """Trust a prefix hit only when the file's own ``name:`` line does not
        contradict the queried name. Multi-face files carry the front-face
        name ("name: fire" in ``fire_ice.txt``), so front-face fallbacks pass;
        a single-face file whose name merely extends the query ("Undercity" →
        ``undercity_dire_rat.txt``) is rejected. Files without a readable
        ``name:`` line are trusted as before.
        """
        txt = path if path.suffix == ".txt" else path.with_suffix(".txt")
        try:
            with open(txt, encoding="utf-8") as f:
                for _ in range(4):
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith("name:"):
                        return sanitize_card_name(line[5:].strip()) == filename
        except OSError:
            pass
        return True

    def _index_for(self, subdirectory: str) -> dict[str, Path]:
        """Index one directory's entries by filename, memoized.

        ``subdirectory`` is the letter (or ``rebalanced``) in a letter-keyed
        tree and empty in a flat one, where the whole tree is one directory.
        """
        cached = self._letter_index.get(subdirectory)
        if cached is not None:
            return cached
        directory = (
            self._cards_path / subdirectory if subdirectory else self._cards_path
        )
        index: dict[str, Path] = {}
        if directory.is_dir():
            for entry in directory.iterdir():
                index[entry.name] = entry
        self._letter_index[subdirectory] = index
        return index

    _REBALANCED_DIR = "rebalanced"

    def _split_filename(self, card_name: str) -> tuple[str, str]:
        # Alchemy/Arena rebalanced cards are named "A-Akki Ronin" and live
        # under cardsfolder/rebalanced/ as "a-akki_ronin.txt" — the literal
        # "a-" prefix is kept (the general sanitizer would turn it into
        # "a_"), and the directory is "rebalanced" rather than first-letter.
        rebalanced = card_name[:2].upper() == "A-" and len(card_name) > 2
        if rebalanced:
            stem = sanitize_card_name(card_name[2:], FILENAME_CORRECTIONS)
            filename, first_letter = f"a-{stem}", self._REBALANCED_DIR
        else:
            resolved = sanitize_card_name(card_name, FILENAME_CORRECTIONS)
            if "/" in resolved:
                first_letter, filename = resolved.split("/", 1)
            else:
                filename = resolved
                first_letter = filename[0] if filename else "_"
        # A flat tree has no per-letter directories, so everything sits at the
        # root; the sanitized filename is unchanged either way.
        if self._layout == _FLAT:
            return filename, ""
        return filename, first_letter
