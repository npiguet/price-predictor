"""PoolLoader: reads pools.txt and assembles pool tensors from .npz embeddings."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BASIC_LAND_NAMES = ["Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"]
_N_BASIC = len(BASIC_LAND_NAMES)  # 6
_DEFAULT_N_SLOTS = 96


def card_npz_path(cards_path: Path, card_name: str) -> Path:
    """Return the .npz path for a card name, matching encode-cards output layout.

    encode-cards saves embeddings alongside card scripts, which are organised as
    ``cards_path/{first_letter}/{snake_case_name}.npz``.  Card names from pools.txt
    use title-case with spaces, so this function normalises them to match.
    """
    stem = card_name.lower().replace(" ", "_")
    return cards_path / stem[0] / f"{stem}.npz"


class PoolLoader:
    def load_pools(self, pools_path: Path) -> list[str]:
        """Return list of semicolon-separated card name strings (one per pool).

        Raises ValueError if path is missing or file is empty.
        """
        if not pools_path.exists():
            raise ValueError(f"pools.txt not found or empty: {pools_path}")
        lines = [
            ln.strip()
            for ln in pools_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not lines:
            raise ValueError(f"pools.txt not found or empty: {pools_path}")
        return lines

    def assemble_pool_tensor(self, pool_names: str, cards_path: Path) -> torch.Tensor:
        """Assemble float32 tensor for a pool; all flags zero.

        Shape: [max(n_booster + 6, 96), card_embed_dim + 4].

        Raises FileNotFoundError(card_name) if any .npz is missing.
        """
        booster_names = pool_names.split(";")
        all_names = booster_names + BASIC_LAND_NAMES
        embeds: list[np.ndarray] = []
        for name in all_names:
            p = card_npz_path(cards_path, name)
            if not p.exists():
                raise FileNotFoundError(f"missing embedding for card: {name}")
            embeds.append(np.load(p)["embedding"])

        embed_dim = embeds[0].shape[0]
        d_model = embed_dim + 4
        n_total = len(embeds)
        n_rows = max(n_total, _DEFAULT_N_SLOTS)

        tensor = np.zeros((n_rows, d_model), dtype=np.float32)
        for i, emb in enumerate(embeds):
            tensor[i, :embed_dim] = emb
            # All flags remain 0 (initial values per spec)

        return torch.from_numpy(tensor)
