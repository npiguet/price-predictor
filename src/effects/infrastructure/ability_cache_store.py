"""Reading and writing the ability cache under the cache root a command names.

The tree mirrors the source tree, so a provenance key's ``script_file`` locates
its vectors without an index. Each variant's cache is read through its stamp
(:func:`require_cache_stamp`), never by a bare :class:`AbilityCacheStore`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from effects.domain.ability_cache_layout import (
    ARRAY_KEY,
    CACHE_SUFFIX,
    CacheStamp,
    CacheStampError,
    cache_path_for,
    stamp_path,
    validate_alignment,
    variant_of_cache_file,
)
from effects.domain.baselines import Baseline
from effects.domain.provenance import ProvenanceSidecar
from price_predictor.infrastructure.atomic_write import write_atomically

if TYPE_CHECKING:
    from collections.abc import Sequence

    from effects.domain.ability_tokenizer import Surface
    from effects.infrastructure.effect_model_store import EffectCheckpoint

logger = logging.getLogger(__name__)


class AbilityCacheStore:
    """The ``.npz`` cache for one variant."""

    def __init__(
        self, root: Path, *, variant: Baseline = Baseline.FULL,
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


def cache_stamp(
    checkpoint_path: Path, checkpoint: EffectCheckpoint, *,
    cards_folders: Sequence[Path | str], surface: Surface | str,
) -> CacheStamp:
    """The stamp a cache encoded from ``checkpoint`` over ``cards_folders`` on
    ``surface`` carries: what ``encode-abilities`` writes and a reader wants."""
    from effects.infrastructure.converted_trees import SourceTrees
    from effects.infrastructure.effect_model_store import content_hash

    return CacheStamp(
        checkpoint_hash=content_hash(checkpoint_path), surface=str(surface),
        vocab_hash=checkpoint.provenance.vocab_hash,
        sidecar_digest=SourceTrees.of(cards_folders).converted_sidecar_digest(),
    )


def read_stamp(root: Path, variant: Baseline = Baseline.FULL) -> CacheStamp | None:
    """The stamp of ``variant``'s cache under ``root``, or None when it has none."""
    path = stamp_path(root, variant)
    if not path.is_file():
        return None
    return CacheStamp.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_stamp(root: Path, stamp: CacheStamp, *, variant: Baseline = Baseline.FULL) -> None:
    """Write ``variant``'s stamp atomically, after its last cache file."""
    text = json.dumps(stamp.as_dict(), indent=2, sort_keys=True) + "\n"
    write_atomically(
        stamp_path(root, variant), lambda path: path.write_text(text, encoding="utf-8"),
    )


def remove_stamp(root: Path, variant: Baseline = Baseline.FULL) -> None:
    """Remove ``variant``'s stamp before its first cache file is written, so a
    run cut short leaves a cache no reader takes for a whole one."""
    stamp_path(root, variant).unlink(missing_ok=True)


def require_cache_stamp(
    root: Path, wanted: CacheStamp | Callable[[], CacheStamp | None] | None, *,
    variant: Baseline = Baseline.FULL,
) -> AbilityCacheStore:
    """``variant``'s cache under ``root``, once its stamp is ``wanted``'s. A root holding none of
    the variant's files needs no stamp; files with no stamp, a stamp that differs, or a
    reader holding no checkpoint of the variant (``wanted`` None) are refused. ``wanted`` may be
    a function, called only when there are files, since a stamp hashes the checkpoint file."""
    store = AbilityCacheStore(root, variant=variant)
    if store.written_files():
        _require_stamp(root, wanted() if callable(wanted) else wanted, variant)
    return store


def _require_stamp(root: Path, wanted: CacheStamp | None, variant: Baseline) -> None:
    """Refuse ``variant``'s cache files under ``root`` unless their stamp is ``wanted``."""
    if wanted is None:
        raise CacheStampError(
            f"the {variant} cache under {root} was encoded from a {variant} checkpoint, "
            f"which this run holds none of: pass --variant-checkpoint {variant}=PATH"
        )
    recorded = read_stamp(root, variant)
    if recorded is None:
        raise CacheStampError(
            f"the {variant} cache under {root} has no stamp ({stamp_path(root, variant).name}), "
            "so nothing names the checkpoint it was encoded from: re-encode it "
            f"with encode-abilities --clean --variant {variant}"
        )
    differences = recorded.differences(wanted)
    if differences:
        raise CacheStampError(
            f"{'; '.join(differences)}: the {variant} cache under {root} was not "
            "encoded from this checkpoint over these trees; re-encode it, or "
            "point --abilities-root at the cache this checkpoint wrote"
        )


def _load(path: Path) -> np.ndarray:
    with np.load(path) as data:
        return data[ARRAY_KEY]


@dataclass(frozen=True, slots=True)
class CachedLine:
    """One sidecar line as its cache row is keyed: the text it encodes on the
    checkpoint's surface, its converted prose where it has any, and the script
    API type its sidecar line records."""

    text: str
    prose: str | None
    api_type: str | None = None


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
        from effects.infrastructure.sidecar_io import encoded_lines, iter_sidecars

        roots = (
            (Path(sidecar_roots),) if isinstance(sidecar_roots, (str, Path))
            else tuple(Path(root) for root in sidecar_roots)
        )
        sources: list[IndexedSource] = []
        for root in roots:
            for path, sidecar in iter_sidecars(root):
                lines = tuple(
                    CachedLine(encoded.text, encoded.prose, encoded.line.script_api_type)
                    if encoded.text else None
                    for encoded in encoded_lines(path, sidecar, surface)
                )
                sources.append(IndexedSource(sidecar.script_file, sidecar.card, lines))
        self.sources: tuple[IndexedSource, ...] = tuple(sources)

    def matrices(
        self, store: AbilityCacheStore,
    ) -> Iterator[tuple[IndexedSource, np.ndarray]]:
        """Each source ``store`` holds rows for, beside them, in tree and path
        order; a cache whose rows do not match its sidecar is skipped with a
        warning. ``store`` comes from :func:`require_cache_stamp`."""
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
