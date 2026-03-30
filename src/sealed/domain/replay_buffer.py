"""Episode dataclass for PPO training."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    pool_names: str            # semicolon-separated card names (booster cards only)
    shuffle_seeds: np.ndarray  # int64, shape (best_run + 1,) — seed per step including terminal
    actions: np.ndarray        # int32, shape (n_picks,) — pool indices (spells + phase-2 lands)
    log_probs: np.ndarray      # float32, shape (n_picks,)
    step_rewards: np.ndarray   # float32, shape (n_picks,) — per-step reward (+1 good, -1 bad)
    reward: float              # scalar summary used for logging
    effective_run: int         # n_total - max(n_spell - 23, 0)
    termination: str           # "success" or "duplicate"
    term_action: int           # pool index of the terminating duplicate pick; -1 for successful episodes
    term_log_prob: float       # log-probability of the terminating pick; 0.0 for successful episodes
