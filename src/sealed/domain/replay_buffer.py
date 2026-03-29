"""Episode dataclass and ReplayBuffer for PPO training."""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    pool_names: str            # semicolon-separated card names (booster cards only)
    shuffle_seeds: np.ndarray  # int32, shape (MAX_PICKS,)
    actions: np.ndarray        # int32, shape (n_picks,) — pool indices
    log_probs: np.ndarray      # float32, shape (n_picks,)
    reward: float


class ReplayBuffer:
    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._buf: deque[Episode] = deque()

    def append(self, episode: Episode) -> None:
        if len(self._buf) == self.max_size:
            self._buf.popleft()
        self._buf.append(episode)

    def sample(self, n: int) -> list[Episode]:
        data = list(self._buf)
        if n >= len(data):
            return list(data)
        return random.sample(data, n)

    def __len__(self) -> int:
        return len(self._buf)

    def to_list(self) -> list[Episode]:
        return list(self._buf)

    @classmethod
    def from_list(cls, episodes: list[Episode], max_size: int = 1000) -> ReplayBuffer:
        buf = cls(max_size=max_size)
        for ep in episodes[-max_size:]:
            buf.append(ep)
        return buf
