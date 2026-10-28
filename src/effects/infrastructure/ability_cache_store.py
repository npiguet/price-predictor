"""Writing the ability cache under ``output/effects/abilities/``.

One ``.npz`` per source file, holding a ``(n_lines, e_dim)`` float32 array under
the key ``e``, row-aligned with that source's sidecar. The tree layout mirrors
the source tree, so a consumer that knows a provenance key's ``script_file``
knows where its vectors are without an index; each reads the files its own way
(``geometry_checks.load_cache``, the probes).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from effects.domain.ability_cache_layout import (
    ARRAY_KEY,
    CACHE_SUFFIX,
    DEFAULT_CACHE_ROOT,
    cache_path_for,
    validate_alignment,
    variant_of_cache_file,
)
from effects.domain.provenance import ProvenanceSidecar

logger = logging.getLogger(__name__)


class AbilityCacheStore:
    """The ``.npz`` cache for one variant."""

    def __init__(
        self, root: Path = DEFAULT_CACHE_ROOT, *, variant: str = "full",
    ) -> None:
        self.root = Path(root)
        self.variant = variant

    def path_for(self, script_file: str) -> Path:
        return cache_path_for(script_file, root=self.root, variant=self.variant)

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
        """Every cache file this variant wrote under ``root``.

        A variant's files are picked out by the suffix ``cache_path_for`` gave
        them, which is what lets ``--clean`` remove only what this command
        wrote and leave the shipping cache — or another variant's — in place.
        """
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
