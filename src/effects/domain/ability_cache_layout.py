"""The ability cache's on-disk layout.

The cache is one ``float32`` array of shape ``(n_lines, e_dim)`` per source
file, **row-aligned with that source's sidecar**: row *i* is the vector for
``sidecar.lines[i]``. Rendered lines for a converted tree, script lines for the
variant tree. That alignment is the whole interface — a consumer holds the
cache file and the sidecar, and needs nothing else to know which vector is which
ability.

This is a different artifact from ``sealed/domain/card_embedding_layout.py``,
which describes the sealed ``.npz``: one pooled card vector followed by
deterministic features. Shape, arity and consumers all differ, so the two are
parallel concepts rather than one.

The cache lives under ``output/effects/abilities/`` rather than beside the
converted text, because the sealed pipeline's ``encode-cards --clean`` deletes
every ``.npz`` under ``output/cardsfolder/`` and would take this one with it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from effects.domain.provenance import ProvenanceSidecar

#: Root of the cache tree. One subtree per source tree, mirroring its layout.
DEFAULT_CACHE_ROOT = Path("output/effects/abilities")
#: The key the `.npz` stores its matrix under.
ARRAY_KEY = "e"
CACHE_SUFFIX = ".npz"


def cache_path_for(
    script_file: str, *, root: Path = DEFAULT_CACHE_ROOT, variant: str = "full",
) -> Path:
    """Where a source file's cache lives.

    ``full`` writes ``<name>.npz``; any other variant writes
    ``<name>.{variant}.npz`` **beside** it rather than over it, so encoding a
    baseline can never replace the cache the shipping model's checks read.
    """
    relative = Path(script_file)
    stem = relative.stem
    suffix = CACHE_SUFFIX if variant == "full" else f".{variant}{CACHE_SUFFIX}"
    return Path(root) / relative.parent / f"{stem}{suffix}"


def variant_of_cache_file(path: Path) -> str | None:
    """The variant a cache file belongs to, as :func:`cache_path_for` named it.

    ``<name>.{variant}.npz`` for a variant the store knows, ``<name>.npz`` for
    ``full`` — whatever dots the source's own name holds — and None for a file
    that is not a cache file at all.
    """
    from effects.domain.variants import VARIANTS

    name = Path(path).name
    if not name.endswith(CACHE_SUFFIX):
        return None
    stem = name[: -len(CACHE_SUFFIX)]
    for variant in VARIANTS:
        if variant != "full" and stem.endswith(f".{variant}"):
            return variant
    return "full"


def validate_alignment(matrix: np.ndarray, sidecar: ProvenanceSidecar) -> None:
    """Raise unless ``matrix`` has one row per sidecar line.

    A misaligned cache is the worst failure mode available here: every vector
    still loads, every shape still checks out, and each ability is read as its
    neighbour's.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"ability cache for {sidecar.script_file} must be 2-D "
            f"(n_lines, e_dim), got shape {matrix.shape}"
        )
    if matrix.shape[0] != len(sidecar.lines):
        raise ValueError(
            f"ability cache for {sidecar.script_file} has {matrix.shape[0]} rows "
            f"but its sidecar has {len(sidecar.lines)} lines; row i must be "
            "lines[i] or every ability reads as its neighbour's"
        )


def pooled_card_vector(matrix: np.ndarray) -> np.ndarray:
    """Mean and max over a card's ability rows, concatenated (FR-114).

    The pooled form the decodability battery and the scorer smoke test read. A
    card with no ability rows pools to zeros of the right width rather than
    raising: a vanilla creature has no ability lines and is still a card.
    """
    if matrix.size == 0:
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        return np.zeros(2 * width, dtype=np.float32)
    return np.concatenate(
        [matrix.mean(axis=0), matrix.max(axis=0)],
    ).astype(np.float32)
