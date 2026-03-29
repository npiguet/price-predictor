"""Episode dataclass for PPO training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    pool_names: str            # semicolon-separated card names (booster cards only)
    shuffle_seeds: np.ndarray  # int32, shape (MAX_PICKS,)
    actions: np.ndarray        # int32, shape (n_picks,) — pool indices
    log_probs: np.ndarray      # float32, shape (n_picks,)
    reward: float
