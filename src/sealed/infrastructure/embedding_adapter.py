"""EmbeddingAdapter: wraps EmbeddingStore to satisfy CardEmbeddingPort."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from sealed.infrastructure.embedding_store import EmbeddingStore
from sealed.infrastructure.pool_loader import card_npz_path


class EmbeddingAdapter:
    """Wraps EmbeddingStore to satisfy CardEmbeddingPort (structural protocol)."""

    def __init__(self, store: EmbeddingStore, cards_path: Path) -> None:
        self._store = store
        self._cards_path = cards_path
        self.total_load_s: float = 0.0
        self._land_cache: dict[str, bool] = {}
        self._text_cache: dict[str, str] = {}

    def get_embedding(self, card_name: str) -> np.ndarray:
        t0 = time.perf_counter()
        result = self._store.load(card_npz_path(self._cards_path, card_name))
        self.total_load_s += time.perf_counter() - t0
        return result

    def is_land(self, card_name: str) -> bool:
        if card_name in self._land_cache:
            return self._land_cache[card_name]
        txt_path = card_npz_path(self._cards_path, card_name).with_suffix(".txt")
        result = False
        if txt_path.exists():
            for line in txt_path.read_text(encoding="utf-8").splitlines():
                low = line.lower()
                if low.startswith("type") and "land" in low:
                    result = True
                    break
        self._land_cache[card_name] = result
        return result

    def get_card_text(self, card_name: str) -> str:
        if card_name in self._text_cache:
            return self._text_cache[card_name]
        txt_path = card_npz_path(self._cards_path, card_name).with_suffix(".txt")
        text = txt_path.read_text(encoding="utf-8") if txt_path.exists() else ""
        self._text_cache[card_name] = text
        return text

    def reset_timing(self) -> None:
        self.total_load_s = 0.0
