"""Reading and writing the ability cache under ``output/effects/abilities/``.

The tree mirrors the source tree, so a provenance key's ``script_file`` locates
its vectors without an index.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from effects.domain.ability_cache_layout import (
    ARRAY_KEY,
    CACHE_SUFFIX,
    DEFAULT_CACHE_ROOT,
    cache_path_for,
    validate_alignment,
    variant_of_cache_file,
)
from effects.domain.baselines import Baseline
from effects.domain.provenance import ProvenanceSidecar

if TYPE_CHECKING:
    from effects.domain.ability_tokenizer import Surface

logger = logging.getLogger(__name__)


class AbilityCacheStore:
    """The ``.npz`` cache for one variant."""

    def __init__(
        self, root: Path = DEFAULT_CACHE_ROOT, *, variant: str = Baseline.FULL,
    ) -> None:
        self.root = Path(root)
        self.variant = variant

    def path_for(self, script_file: str) -> Path:
        return cache_path_for(script_file, root=self.root, variant=self.variant)

    def read(self, script_file: str) -> np.ndarray | None:
        """One source's rows, or None when this variant holds no cache for it."""
        path = self.path_for(script_file)
        return _load(path) if path.exists() else None

    def iter_matrices(self) -> Iterator[tuple[Path, np.ndarray]]:
        """Every cache file this variant wrote, with its rows, in path order."""
        for path in self.written_files():
            yield path, _load(path)

    def write(
        self, script_file: str, matrix: np.ndarray,
        sidecar: ProvenanceSidecar | None = None,
    ) -> Path:
        """Write one source's rows. Validates alignment when given the sidecar."""
        if sidecar is not None:
            validate_alignment(matrix, sidecar)
        path = self.path_for(script_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{ARRAY_KEY: matrix.astype(np.float32)})
        return path

    def written_files(self) -> list[Path]:
        """Every cache file this variant wrote under ``root``, by the suffix
        ``cache_path_for`` gave it, so ``--clean`` leaves other variants'."""
        if not self.root.is_dir():
            return []
        return sorted(
            path for path in self.root.rglob(f"*{CACHE_SUFFIX}")
            if variant_of_cache_file(path) == self.variant
        )

    def clean(self) -> int:
        """Delete this variant's cache files; return how many went."""
        paths = self.written_files()
        for path in paths:
            path.unlink()
        logger.info("Removed %d %s cache files under %s",
                    len(paths), self.variant, self.root)
        return len(paths)


def _load(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data[ARRAY_KEY]


@dataclass(frozen=True, slots=True)
class CachedLine:
    """One sidecar line as its cache row is keyed: the text it encodes on the
    checkpoint's surface, and its converted prose where it has any."""

    text: str
    prose: str | None


@dataclass(frozen=True, slots=True)
class IndexedSource:
    """One source's sidecar, reduced to what keys its cache rows; a line with
    no text on the surface is None."""

    script_file: str
    card: str
    lines: tuple[CachedLine | None, ...]


class SidecarIndex:
    """Every sidecar under ``sidecar_roots``, read once and keyed on ``surface``.

    Every variant's cache is row-aligned with the same sidecars, so the
    evaluator reads the trees once and joins each variant's matrices to them,
    rather than re-walking them per variant.
    """

    def __init__(self, sidecar_roots: Path | Iterable[Path], *, surface: Surface) -> None:
        from effects.domain.ability_encoder import encoding_text
        from effects.infrastructure.sidecar_io import (
            converted_text_path,
            iter_sidecars,
            prose_for,
            prose_lines,
        )

        roots = (
            (Path(sidecar_roots),) if isinstance(sidecar_roots, (str, Path))
            else tuple(Path(root) for root in sidecar_roots)
        )
        sources: list[IndexedSource] = []
        for root in roots:
            for path, sidecar in iter_sidecars(root):
                rendered = prose_lines(converted_text_path(path))
                lines: list[CachedLine | None] = []
                for line in sidecar.lines:
                    prose = prose_for(line, rendered)
                    text = encoding_text(line, prose, surface)
                    lines.append(CachedLine(text, prose) if text else None)
                sources.append(IndexedSource(sidecar.script_file, sidecar.card, tuple(lines)))
        self.sources: tuple[IndexedSource, ...] = tuple(sources)

    def matrices(
        self, abilities_root: Path, *, variant: str = Baseline.FULL,
    ) -> Iterator[tuple[IndexedSource, np.ndarray]]:
        """Each source this variant's cache holds rows for, beside them, in
        tree and path order; a cache whose rows do not match its sidecar is
        skipped with a warning."""
        store = AbilityCacheStore(abilities_root, variant=variant)
        if not store.root.is_dir():
            return
        for source in self.sources:
            matrix = store.read(source.script_file)
            if matrix is None:
                continue
            if matrix.shape[0] != len(source.lines):
                logger.warning(
                    "skipping %s: %d cache rows for %d sidecar lines",
                    store.path_for(source.script_file), matrix.shape[0],
                    len(source.lines),
                )
                continue
            yield source, matrix
