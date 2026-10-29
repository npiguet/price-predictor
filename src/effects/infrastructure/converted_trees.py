"""The converted source trees one run reads its ability texts from.

``--cards-folder`` names the converted card and token trees, and
``--variant-scripts`` the perturbed one. Every command that joins a key to its
text, or computes the holdout, needs the same few things out of them: which
folder holds which tree, the converted cards, a sidecar cache over every root
and the texts the holdout is chosen from. They are read here once, so the
holdout ``build-corpus`` splits by and the one ``holdout-cards`` depletes pools
by come from one reading of the trees rather than one each.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from effects.domain.provenance import SourceTree
from effects.infrastructure.card_sources import (
    ConvertedCard,
    load_card_texts,
    load_converted_cards,
)
from effects.infrastructure.sidecar_io import SidecarCache, load_tree_texts, sidecar_roots

if TYPE_CHECKING:
    from effects.domain.split import HeldOutCards


@dataclass(frozen=True)
class ConvertedTrees:
    """``tree -> folder`` over the converted trees, and what is read from them.

    The reads are lazy and kept, because the card tree is some 33,000 files
    and a command that needs both the cards and the holdout reads it once.
    """

    #: ``tree -> folder``, the variant tree included when one was named.
    roots: Mapping[str, Path]

    @classmethod
    def from_folders(
        cls, folders: Iterable[Path | str], variant_scripts: Path | str | None = None,
    ) -> ConvertedTrees:
        """From ``--cards-folder`` values, each folder's tree read off its name."""
        return cls(sidecar_roots(folders, variant_scripts))

    @property
    def root_paths(self) -> dict[str, str]:
        """``roots`` as plain strings, which is how a worker process receives them."""
        return {str(tree): str(folder) for tree, folder in self.roots.items()}

    @property
    def cards_folder(self) -> Path | None:
        """The converted card tree, or None when no folder holds one."""
        return self.roots.get(SourceTree.CARDSFOLDER)

    @cached_property
    def converted_cards(self) -> dict[str, ConvertedCard]:
        """``card name -> its converted file``; empty without a card tree on disk."""
        folder = self.cards_folder
        if folder is None or not folder.is_dir():
            return {}
        return load_converted_cards(folder)

    @cached_property
    def card_files(self) -> dict[str, str]:
        """``card name -> tree-prefixed script file`` over the card tree."""
        return {name: card.script_file for name, card in self.converted_cards.items()}

    @cached_property
    def sidecars(self) -> SidecarCache:
        return SidecarCache(self.roots)

    def card_texts(self) -> dict[str, list[str]]:
        """``card name -> the script texts of its sidecar's lines``."""
        return load_card_texts(self.card_files, self.sidecars)

    def token_texts(self) -> dict[str, list[str]]:
        """``token script file -> its sidecar's script texts``; empty without tokens."""
        return load_tree_texts(self.roots.get(SourceTree.TOKENSCRIPTS))

    def holdout(self, *, permille: int, max_carriers: int, unit: str) -> HeldOutCards:
        """The trainer's holdout rule over the card tree, token scripts as carriers.

        Token scripts count because a token printing a held-out text puts its
        script in the holdout, so the games it appears in are held out too.
        """
        from effects.domain.split import text_keyed_holdout

        return text_keyed_holdout(
            self.card_files, self.card_texts(),
            permille=permille, max_carriers=max_carriers, unit=unit,
            token_texts=self.token_texts(),
        )
