"""Per-epoch GAE advantage precompute writes finite advantages (spec 020 D3)."""

from __future__ import annotations

import numpy as np
import torch

from draft.application.imitation_batches import BatchSource
from draft.application.rl_advantage_refresh import (
    AdvantageRefresh,
    CriticPass,
    _precompute_advantages,
)
from draft.application.rl_examples import RLExample
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_state import TYPE_PACK, TYPE_POOL

DIM = 8  # concat_width 24 divisible by the default 8 heads


def _ex(reward: float) -> RLExample:
    types = [TYPE_PACK, TYPE_PACK, TYPE_POOL]
    n = len(types)
    return RLExample(
        draft_index=0,
        card_idx=np.array([0, 1, 2], dtype=np.int32),
        type_idx=np.array(types, dtype=np.int8),
        packs_ago=np.zeros(n, dtype=np.int8),
        pick_ago=np.zeros(n, dtype=np.int8),
        pack_number=1,
        pick_number=1,
        action_token=0,
        learner_active=True,
        critic_active=True,
        reward=reward,
    )


def _refresh(table: np.ndarray) -> AdvantageRefresh:
    source = BatchSource(table, mean=0.0, std=1.0, device=torch.device("cpu"))
    return AdvantageRefresh(CriticPass(source, batch_size=4), gae_lambda=0.95)


def test_precompute_writes_finite_values_and_advantages() -> None:
    torch.manual_seed(0)
    model = DraftAgentModel(DraftAgentConfig(embedding_dim=DIM, packs=1, P=4))
    table = np.random.default_rng(0).standard_normal((6, DIM)).astype(np.float32)
    # Two trajectories with different terminal rewards → some advantage spread.
    trajectories = [
        [_ex(2.0), _ex(2.0), _ex(2.0)],
        [_ex(-1.0), _ex(-1.0)],
    ]
    _precompute_advantages(model, trajectories, _refresh(table), epoch=0)
    for traj in trajectories:
        for ex in traj:
            assert np.isfinite(ex.value)
            assert np.isfinite(ex.advantage)
    # λ<1, terminal reward: the last pick's advantage is reward − value.
    last = trajectories[0][-1]
    assert abs(last.advantage - (2.0 - last.value)) < 1e-6


def test_precompute_no_trajectories_is_noop() -> None:
    torch.manual_seed(0)
    model = DraftAgentModel(DraftAgentConfig(embedding_dim=DIM, packs=1, P=4))
    table = np.zeros((1, DIM), dtype=np.float32)
    _precompute_advantages(model, [], _refresh(table))  # must not raise
