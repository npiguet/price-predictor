"""Episode dataclass for PPO training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    pool_names: str            # semicolon-separated card names (booster cards only)
    shuffle_seeds: np.ndarray  # int32, shape (MAX_PICKS,)
    actions: np.ndarray        # int32, shape (n_picks,) — pool indices (spells + phase-2 lands)
    log_probs: np.ndarray      # float32, shape (n_picks,)
    reward: float
    effective_run: int         # n_total - max(n_spell - 23, 0); used to update best_run
