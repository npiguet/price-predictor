"""The cached ``.npz`` text vectors: per-card lookup, the corpus cloud, the manifold gate."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from probe_encoder import TEXT_DIM
from probe_join import build_join
from probe_paths import CARDS_PATH, SCRATCH


def load_embedding_matrix(
    names: Sequence[str], join: pd.DataFrame | None = None,
) -> np.ndarray:
    """``(len(names), 512)`` float32 text vectors from the ``.npz`` cache."""
    join = build_join() if join is None else join
    by_name = dict(zip(join["name"], join["npz_path"]))
    out = np.empty((len(names), TEXT_DIM), dtype=np.float32)
    for i, name in enumerate(names):
        path = by_name.get(name)
        if path is None:
            raise KeyError(f"no cached embedding for card: {name!r}")
        with np.load(path) as f:
            out[i] = f["embedding"][:TEXT_DIM]
    return out


CORPUS_MATRIX = SCRATCH / "corpus_embeddings.npz"


def corpus_embedding_matrix(
    *, force: bool = False, cards_path: Path = CARDS_PATH,
) -> tuple[list[str], np.ndarray]:
    """Every cached ``.npz`` in the corpus → ``(keys, (N, 512))``.

    A key is the extension-less path relative to the cards folder, e.g.
    ``s/serra_angel`` — join it against a row's ``txt_path`` with
    :func:`corpus_key`. This is the reference cloud for
    :func:`manifold_distance`: all 32,606 converted cards, not just the
    ~28k carrying winnability labels.
    """
    if CORPUS_MATRIX.exists() and not force:
        with np.load(CORPUS_MATRIX, allow_pickle=False) as f:
            return list(f["names"]), f["matrix"]
    paths = sorted(cards_path.rglob("*.npz"))
    matrix = np.empty((len(paths), TEXT_DIM), dtype=np.float32)
    names: list[str] = []
    for i, path in enumerate(paths):
        with np.load(path) as f:
            matrix[i] = f["embedding"][:TEXT_DIM]
        names.append(corpus_key(path, cards_path))
    np.savez_compressed(CORPUS_MATRIX, names=np.array(names), matrix=matrix)
    return names, matrix


def corpus_key(path: Path | str, cards_path: Path = CARDS_PATH) -> str:
    """Extension-less corpus-relative key of a card file (``s/serra_angel``)."""
    rel = Path(path).resolve().relative_to(cards_path.resolve())
    return rel.with_suffix("").as_posix()



def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def manifold_distance(
    embeddings: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    self_rows: np.ndarray | None = None,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance from each row to its nearest real-card embedding.

    ``reference`` defaults to the full 512-dim corpus cache. ``self_rows``
    gives, per input row, the reference index to exclude (use it when the
    inputs *are* real cards, so the nearest neighbour is not the card
    itself). Returns ``(distance, nearest_index)``.

    An edited card whose distance clears the real-card 95th percentile
    has left the manifold: whatever the probe reads off it is an
    extrapolation, not a counterfactual.
    """
    if reference is None:
        _, reference = corpus_embedding_matrix()
    ref = _normalize(np.asarray(reference, dtype=np.float32))
    query = _normalize(np.asarray(embeddings, dtype=np.float32))
    dist = np.empty(len(query), dtype=np.float32)
    idx = np.empty(len(query), dtype=np.int64)
    for lo in range(0, len(query), chunk):
        sims = query[lo:lo + chunk] @ ref.T
        if self_rows is not None:
            rows = np.arange(sims.shape[0])
            sims[rows, self_rows[lo:lo + chunk]] = -np.inf
        best = sims.argmax(axis=1)
        idx[lo:lo + chunk] = best
        dist[lo:lo + chunk] = 1.0 - sims[np.arange(sims.shape[0]), best]
    return dist, idx

