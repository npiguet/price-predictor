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

The cache lives under a cache run (``output/effects/<gen>/cache/<run>/``) rather
than beside the converted text, because the sealed pipeline's ``encode-cards --clean`` deletes
every ``.npz`` under the converted card tree
(``output/effects/gen3/converted/cardsfolder/``) and would take this one with it.

Each variant's cache has a stamp beside it at the cache root
(:func:`stamp_path`): a :class:`CacheStamp` naming the checkpoint, surface,
vocabulary and converted trees it was encoded from, which every reader
compares with the checkpoint it evaluates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from effects.domain.baselines import BASELINES, Baseline
from effects.domain.provenance import ProvenanceSidecar

#: The key the `.npz` stores its matrix under.
ARRAY_KEY = "e"
CACHE_SUFFIX = ".npz"


def cache_suffix(variant: Baseline) -> str:
    """The file suffix a variant's cache files end in.

    ``full`` writes ``<name>.npz``; any other variant writes
    ``<name>.{variant}.npz`` **beside** it rather than over it, so encoding a
    baseline can never replace the cache the shipping model's checks read.
    """
    return CACHE_SUFFIX if variant == Baseline.FULL else f".{variant}{CACHE_SUFFIX}"


def cache_path_for(
    script_file: str, *, root: Path, variant: Baseline = Baseline.FULL,
) -> Path:
    """Where a source file's cache lives, under the variant's :func:`cache_suffix`."""
    relative = Path(script_file)
    return Path(root) / relative.parent / f"{relative.stem}{cache_suffix(variant)}"


def variant_of_cache_file(path: Path) -> Baseline | None:
    """The variant a cache file belongs to, as :func:`cache_path_for` named it.

    ``<name>.{variant}.npz`` for a variant the store knows, ``<name>.npz`` for
    ``full`` — whatever dots the source's own name holds — and None for a file
    that is not a cache file at all.
    """
    name = Path(path).name
    if not name.endswith(CACHE_SUFFIX):
        return None
    stem = name[: -len(CACHE_SUFFIX)]
    for variant in BASELINES:
        if variant != Baseline.FULL and stem.endswith(f".{variant}"):
            return variant
    return Baseline.FULL


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

    The pooled form the decodability battery reads. A
    card with no ability rows pools to zeros of the right width rather than
    raising: a vanilla creature has no ability lines and is still a card.
    """
    if matrix.size == 0:
        width = matrix.shape[1] if matrix.ndim == 2 else 0
        return np.zeros(2 * width, dtype=np.float32)
    return np.concatenate(
        [matrix.mean(axis=0), matrix.max(axis=0)],
    ).astype(np.float32)


class CacheStampError(ValueError):
    """A cache with no stamp, a stamp missing a member, or one naming another
    checkpoint, surface, vocabulary or conversion than the reader's."""


def stamp_path(root: Path, variant: Baseline = Baseline.FULL) -> Path:
    """The stamp of ``variant``'s cache under ``root``: ``ability-cache.stamp.json``
    for ``full``, ``ability-cache.{variant}.stamp.json`` for a baseline, since
    the variants share a root and each was encoded from its own checkpoint."""
    infix = "" if variant == Baseline.FULL else f".{variant}"
    return Path(root) / f"ability-cache{infix}.stamp.json"


@dataclass(frozen=True, slots=True)
class CacheStamp:
    """What a cache was encoded from. A member None in a *wanted* stamp is not
    compared: the reader holds no checkpoint of that variant."""

    #: sha256 of the checkpoint file's bytes.
    checkpoint_hash: str | None
    #: The encoding surface of the checkpoint's vocabulary (``prose``, ``script``).
    surface: str
    #: The checkpoint's recorded vocabulary hash.
    vocab_hash: str
    #: The converted trees' sidecar digest (``SourceTrees.converted_sidecar_digest``).
    sidecar_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheStamp:
        """The stamp ``as_dict`` wrote; one missing a member is refused by its name."""
        missing = [spec.name for spec in fields(cls) if spec.name not in data]
        if missing:
            raise CacheStampError(f"the cache stamp records no {', '.join(missing)}")
        return cls(**{spec.name: data[spec.name] for spec in fields(cls)})

    def without_checkpoint(self) -> CacheStamp:
        """This stamp wanted with the checkpoint left uncompared."""
        return CacheStamp(None, self.surface, self.vocab_hash, self.sidecar_digest)

    def differences(self, wanted: CacheStamp) -> list[str]:
        """Each member ``wanted`` states that this stamp records otherwise,
        with both values."""
        return [
            f"{spec.name}: cache {getattr(self, spec.name)}, wanted {getattr(wanted, spec.name)}"
            for spec in fields(self)
            if getattr(wanted, spec.name) is not None
            and getattr(self, spec.name) != getattr(wanted, spec.name)
        ]
