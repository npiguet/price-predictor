"""T007 — Unit tests for Episode dataclass."""
from __future__ import annotations

import numpy as np

from sealed.domain.replay_buffer import Episode

MAX_PICKS = 40


def test_episode_fields_accessible():
    ep = Episode(
        pool_names="CardA;CardB;CardC",
        shuffle_seeds=np.zeros(MAX_PICKS, dtype=np.int64),
        actions=np.arange(3, dtype=np.int32),
        log_probs=np.zeros(3, dtype=np.float32),
        reward=1.0,
    )
    assert ep.pool_names == "CardA;CardB;CardC"
    assert ep.reward == 1.0
    assert len(ep.actions) == 3
    assert len(ep.log_probs) == 3
    assert len(ep.shuffle_seeds) == MAX_PICKS
