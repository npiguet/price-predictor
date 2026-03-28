"""EmbeddingStore: atomic save and load of .npz embedding files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class EmbeddingStore:
    """Persists card embeddings as compressed .npz files.

    Writes are atomic: data is written to a temp file ({stem}.tmp.npz)
    then renamed to the final path so a partial write is never visible.
    """

    def save(self, path: Path, embedding: np.ndarray) -> None:
        """Save embedding to path atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (path.stem + ".tmp.npz")
        np.savez_compressed(tmp, embedding=embedding)
        os.replace(str(tmp), str(path))

    def load(self, path: Path) -> np.ndarray:
        """Load embedding from a .npz file."""
        return np.load(path)["embedding"]
