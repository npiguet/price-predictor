"""The converted source trees one run reads its ability texts from.

:class:`SourceTrees` is the value every command and worker names them by:
``--cards-folder`` plus the optional ``--variant-scripts`` tree.
:class:`ConvertedTrees` is what is read from them. ``build-corpus``'s split
and ``holdout-cards``' depletion list both come from
:meth:`ConvertedTrees.holdout`, one reading of the trees.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
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
    from effects.domain.split import HeldOutCards, HoldoutRule


@dataclass(frozen=True, slots=True)
class SourceTrees:
    """``--cards-folder`` values plus the optional ``--variant-scripts`` tree.

    A plain value, so it pickles into a worker; every sidecar cache, folder
    list and root map a command reads its texts through is derived from it.
    """

    cards_folders: tuple[Path, ...]
    variant_scripts: Path | None = None

    @classmethod
    def of(
        cls, cards_folders: Iterable[Path | str],
        variant_scripts: Path | str | None = None,
    ) -> SourceTrees:
        return cls(
            tuple(Path(folder) for folder in cards_folders),
            Path(variant_scripts) if variant_scripts else None,
        )

    def folders(self) -> list[Path]:
        """Every tree's folder, the variant tree last where one is named."""
        folders = list(self.cards_folders)
        if self.variant_scripts is not None:
            folders.append(self.variant_scripts)
        return folders

    def sidecar_roots(self) -> dict[str, Path]:
        """``tree -> folder``, each card folder keyed by the tree its name ends
        in (FR-063b), so ``output/gen1-cardsfolder`` serves ``cardsfolder/`` keys."""
        return sidecar_roots(self.cards_folders, self.variant_scripts)

    def sidecar_cache(self) -> SidecarCache:
        """A fresh cache over these trees, for a caller that owns its tallies."""
        return SidecarCache(self.sidecar_roots())

    def converted(self, *, sidecars: SidecarCache | None = None) -> ConvertedTrees:
        """What is read from these trees; ``sidecars`` shares a cache already
        built over the same roots."""
        return ConvertedTrees(self.sidecar_roots(), sidecars)


def process_sidecars(roots: Mapping[str, Path]) -> SidecarCache:
    """This process's sidecar cache over ``roots``, built once and kept.

    For worker processes, which receive the roots (:meth:`SourceTrees.sidecar_roots`)
    rather than a cache, which does not pickle, and read largely the same
    cards shard after shard. One cache per process: a different ``roots``
    replaces it.
    """
    return _process_sidecars(tuple(sorted(
        (str(tree), Path(folder)) for tree, folder in roots.items()
    )))


@lru_cache(maxsize=1)
def _process_sidecars(roots: tuple[tuple[str, Path], ...]) -> SidecarCache:
    return SidecarCache(dict(roots))


@dataclass(frozen=True)
class ConvertedTrees:
    """``tree -> folder`` over the converted trees (the variant tree included
    when named), and what is read from them, lazily and once; built by
    :meth:`SourceTrees.converted`."""

    roots: Mapping[str, Path]
    #: Built over ``roots`` when not given.
    sidecars: SidecarCache = field(default=None, compare=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.sidecars is None:
            object.__setattr__(self, "sidecars", SidecarCache(self.roots))

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

    def card_texts(self) -> dict[str, list[str]]:
        """``card name -> the script texts of its sidecar's lines``."""
        return load_card_texts(self.card_files, self.sidecars)

    def token_texts(self) -> dict[str, list[str]]:
        """``token script file -> its sidecar's script texts``; empty without tokens."""
        return load_tree_texts(self.roots.get(SourceTree.TOKENSCRIPTS))

    def holdout(self, rule: HoldoutRule) -> HeldOutCards:
        """``rule`` applied over the card tree, token scripts as carriers."""
        from effects.domain.split import text_keyed_holdout

        return text_keyed_holdout(
            self.card_files, self.card_texts(), rule, token_texts=self.token_texts(),
        )
