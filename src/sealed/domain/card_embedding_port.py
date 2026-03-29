"""CardEmbeddingPort: protocol satisfied structurally by any embedding provider."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class CardEmbeddingPort(Protocol):
    def get_embedding(self, card_name: str) -> np.ndarray: ...
    def is_land(self, card_name: str) -> bool: ...
