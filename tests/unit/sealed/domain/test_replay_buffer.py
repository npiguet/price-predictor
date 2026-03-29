"""T007 — Unit tests for ReplayBuffer and Episode."""
from __future__ import annotations

import numpy as np
import pytest

from sealed.domain.replay_buffer import Episode, ReplayBuffer

MAX_PICKS = 40


def _make_episode(reward: float = 0.0, n_picks: int = 3) -> Episode:
    return Episode(
        pool_names="CardA;CardB;CardC",
        shuffle_seeds=np.zeros(MAX_PICKS, dtype=np.int64),
        actions=np.arange(n_picks, dtype=np.int32),
        log_probs=np.zeros(n_picks, dtype=np.float32),
        reward=reward,
    )


# ── FIFO eviction ─────────────────────────────────────────────────────────────

def test_fifo_eviction():
    buf = ReplayBuffer(max_size=3)
    ep0 = _make_episode(0.0)
    ep1 = _make_episode(1.0)
    ep2 = _make_episode(2.0)
    ep3 = _make_episode(3.0)
    buf.append(ep0)
    buf.append(ep1)
    buf.append(ep2)
    assert len(buf) == 3
    buf.append(ep3)  # should evict ep0
    assert len(buf) == 3
    items = buf.to_list()
    # Check by identity, not equality (Episode contains numpy arrays)
    item_ids = {id(x) for x in items}
    assert id(ep0) not in item_ids, "ep0 should have been evicted"
    assert id(ep3) in item_ids, "ep3 should be present"


def test_max_size_respected():
    buf = ReplayBuffer(max_size=5)
    for i in range(10):
        buf.append(_make_episode(float(i)))
    assert len(buf) == 5


# ── sample ─────────────────────────────────────────────────────────────────────

def test_sample_n_greater_than_len_returns_all():
    buf = ReplayBuffer(max_size=10)
    for i in range(4):
        buf.append(_make_episode(float(i)))
    result = buf.sample(100)
    assert len(result) == 4


def test_sample_n_less_than_len():
    buf = ReplayBuffer(max_size=10)
    for i in range(8):
        buf.append(_make_episode(float(i)))
    result = buf.sample(3)
    assert len(result) == 3


def test_sample_without_replacement():
    buf = ReplayBuffer(max_size=10)
    episodes = [_make_episode(float(i)) for i in range(5)]
    for ep in episodes:
        buf.append(ep)
    result = buf.sample(5)
    # All unique objects
    assert len(set(id(e) for e in result)) == 5


# ── to_list / from_list ────────────────────────────────────────────────────────

def test_to_list_from_list_round_trip():
    buf = ReplayBuffer(max_size=10)
    episodes = [_make_episode(float(i)) for i in range(5)]
    for ep in episodes:
        buf.append(ep)
    lst = buf.to_list()
    restored = ReplayBuffer.from_list(lst, max_size=10)
    assert len(restored) == 5
    # Compare by identity (Episode contains numpy arrays, equality is ambiguous)
    assert [id(x) for x in restored.to_list()] == [id(x) for x in lst]


def test_from_list_respects_max_size():
    episodes = [_make_episode(float(i)) for i in range(20)]
    buf = ReplayBuffer.from_list(episodes, max_size=5)
    assert len(buf) == 5
    # Should have the last 5
    assert buf.to_list() == episodes[-5:]


def test_len_empty_buffer():
    buf = ReplayBuffer(max_size=10)
    assert len(buf) == 0
