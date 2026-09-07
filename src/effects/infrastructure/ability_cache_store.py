"""Reading and writing the ability cache under ``output/effects/abilities/``.

One ``.npz`` per source file, holding a ``(n_lines, e_dim)`` float32 array under
the key ``e``, row-aligned with that source's sidecar. The tree layout mirrors
the source tree, so a consumer that knows a provenance key's ``script_file``
knows where its vectors are without an index.
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
)
from effects.domain.provenance import ProvenanceKey, ProvenanceSidecar

logger = logging.getLogger(__name__)


class AbilityCacheStore:
    """The ``.npz`` cache for one variant."""

    def __init__(
        self, root: Path = DEFAULT_CACHE_ROOT, *, variant: str = "full",
    ) -> None:
        self.root = Path(root)
        self.variant = variant
        self._loaded: dict[str, np.ndarray] = {}

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

    def read(self, script_file: str) -> np.ndarray | None:
        """One source's rows, memoized. None when it has not been encoded."""
        cached = self._loaded.get(script_file)
        if cached is not None:
            return cached
        path = self.path_for(script_file)
        if not path.exists():
            return None
        with np.load(path) as data:
            matrix = data[ARRAY_KEY]
        self._loaded[script_file] = matrix
        return matrix

    def vector_for(
        self, key: ProvenanceKey, sidecar: ProvenanceSidecar,
    ) -> np.ndarray | None:
        """The vector a provenance key names.

        None where the key resolves to no line (a deduplicated trait) or the
        source has not been encoded — both are absences the caller handles by
        contributing zeros, not errors.
        """
        row = sidecar.row_for(key)
        if row is None:
            return None
        matrix = self.read(key.script_file)
        if matrix is None or row >= matrix.shape[0]:
            return None
        return matrix[row]

    def has(self, script_file: str) -> bool:
        return self.path_for(script_file).exists()

    def written_files(self) -> list[Path]:
        """Every cache file this variant wrote under ``root``.

        A variant's files are picked out by suffix, which is what lets
        ``--clean`` remove only what this command wrote and leave the shipping
        cache — or another variant's — in place.
        """
        if not self.root.is_dir():
            return []
        if self.variant == "full":
            return sorted(
                path for path in self.root.rglob(f"*{CACHE_SUFFIX}")
                if path.name.count(".") == 1
            )
        return sorted(self.root.rglob(f"*.{self.variant}{CACHE_SUFFIX}"))

    def clean(self) -> int:
        """Delete this variant's cache files; return how many went."""
        paths = self.written_files()
        for path in paths:
            path.unlink()
        logger.info("Removed %d %s cache files under %s",
                    len(paths), self.variant, self.root)
        return len(paths)
