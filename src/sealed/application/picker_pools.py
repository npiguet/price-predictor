"""The ``train-picker`` pools: each sealed pool's cards as rows of one shared
embedding table, the fixed validation slice, the per-epoch shuffle and the
zero-padded batch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from sealed.domain.card_embedding_layout import is_land_embedding
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.infrastructure.converted_card_locator import ConvertedCardLocator
from sealed.infrastructure.pool_file_reader import SealedPool


@dataclass(frozen=True)
class SharedRows:
    """A pool's cards as rows of the shared embedding table: the table and
    the pool's ``int`` row indices into it, in pool order."""

    table: np.ndarray
    indices: np.ndarray


class PreparedPool:
    """A sealed pool plus its card embeddings.

    Cards repeat heavily across pools of the same set, so a 100k-pool file of
    per-pool ``(N, dim)`` float32 copies costs ~20 GB. To bound that, pools may
    instead store ``int32`` indices into a shared embedding table built once by
    ``_prepare_pools``; ``.embeddings`` materializes the per-pool ``(N, dim)``
    view on demand (transient, GC'd after each batch). The dense form — passing
    a real ``(N, dim)`` array — is kept for tests and small callers.
    """

    def __init__(
        self,
        set_code: str,
        names: list[str],
        embeddings: np.ndarray,
        is_land: np.ndarray,
    ) -> None:
        self.set_code = set_code
        self.names = names
        self.is_land = is_land
        self._dense: np.ndarray | None = embeddings
        self._table: np.ndarray | None = None
        self._indices: np.ndarray | None = None

    @classmethod
    def from_shared(
        cls,
        set_code: str,
        names: list[str],
        rows: SharedRows,
        is_land: np.ndarray,
    ) -> "PreparedPool":
        obj = cls.__new__(cls)
        obj.set_code = set_code
        obj.names = names
        obj.is_land = is_land
        obj._dense = None
        obj._table = rows.table
        obj._indices = rows.indices
        return obj

    @property
    def embeddings(self) -> np.ndarray:
        if self._dense is not None:
            return self._dense
        assert self._table is not None and self._indices is not None
        return self._table[self._indices]

    @property
    def num_cards(self) -> int:
        return int(self.is_land.shape[0])

    @property
    def dim(self) -> int:
        if self._dense is not None:
            return int(self._dense.shape[1])
        assert self._table is not None
        return int(self._table.shape[1])


class _EmbeddingTable:
    """The shared embedding table ``_prepare_pools`` builds: each distinct
    card's embedding loaded once, a missing card remembered as ``None``."""

    def __init__(self, locator: ConvertedCardLocator) -> None:
        self.locator = locator
        self.rows: list[np.ndarray] = []
        self.is_land: list[bool] = []
        self.name_to_idx: dict[str, int | None] = {}

    def intern(self, name: str) -> int | None:
        idx = self.name_to_idx.get(name, -1)
        if idx != -1:
            return idx
        emb = self.locator.load_embedding(name)
        if emb is None:
            self.name_to_idx[name] = None
            return None
        new_idx = len(self.rows)
        self.rows.append(np.asarray(emb, dtype=np.float32))
        self.is_land.append(bool(is_land_embedding(emb)))
        self.name_to_idx[name] = new_idx
        return new_idx

    def embeddable(self, names: list[str]) -> tuple[list[str], list[int]]:
        """The names with an embedding, in pool order, and their table rows."""
        valid_names: list[str] = []
        idxs: list[int] = []
        for name in names:
            idx = self.intern(name)
            if idx is not None:
                valid_names.append(name)
                idxs.append(idx)
        return valid_names, idxs

    def prepared(
        self, pending: list[tuple[str, list[str], list[int]]],
    ) -> list[PreparedPool]:
        """Each kept pool as a ``PreparedPool`` over the stacked table."""
        table = np.stack(self.rows).astype(np.float32)
        land_table = np.array(self.is_land, dtype=bool)
        prepared: list[PreparedPool] = []
        for set_code, valid_names, idxs in pending:
            index_arr = np.asarray(idxs, dtype=np.int64)
            prepared.append(
                PreparedPool.from_shared(
                    set_code, valid_names, SharedRows(table, index_arr),
                    land_table[index_arr],
                )
            )
        return prepared


def _prepare_pools(
    pools: list[SealedPool], locator: ConvertedCardLocator,
) -> list[PreparedPool]:
    """Load embeddings for every pool; skip pools with < 23 embeddable cards.

    Each distinct card is loaded once into a shared table; pools keep only int
    indices into it (see ``PreparedPool``) so a large pool file stays at the
    size of the unique-card set rather than the sum of all pool copies.
    """
    table = _EmbeddingTable(locator)
    pending: list[tuple[str, list[str], list[int]]] = []
    for pool in pools:
        valid_names, idxs = table.embeddable(pool.cards)
        if len(valid_names) < NONLAND_DECK_SIZE:
            continue
        pending.append((pool.set_code, valid_names, idxs))

    if not pending:
        return []
    return table.prepared(pending)


def _split_pools(
    pools: list[PreparedPool], val_fraction: float,
) -> tuple[list[PreparedPool], list[PreparedPool]]:
    """Front ``val_fraction`` is the fixed validation slice; rest is training."""
    n_val = int(len(pools) * val_fraction)
    return pools[:n_val], pools[n_val:]


def _shuffle_train(
    pools: list[PreparedPool], rng: random.Random,
) -> list[PreparedPool]:
    out = list(pools)
    rng.shuffle(out)
    return out


def _collate(pool_slice: list[PreparedPool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-pad a slice of pools to ``(B, max_N, dim)`` + masks."""
    b = len(pool_slice)
    max_n = max(p.num_cards for p in pool_slice)
    dim = pool_slice[0].dim
    cards = np.zeros((b, max_n, dim), dtype=np.float32)
    mask = np.zeros((b, max_n), dtype=bool)
    land = np.zeros((b, max_n), dtype=bool)
    for i, p in enumerate(pool_slice):
        n = p.num_cards
        cards[i, :n] = p.embeddings
        mask[i, :n] = True
        land[i, :n] = p.is_land
    return cards, mask, land
