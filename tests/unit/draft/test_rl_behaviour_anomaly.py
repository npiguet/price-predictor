"""Behaviour-anomaly summary over learner picks (spec 020 FR-009/SC-004; D6)."""

from __future__ import annotations

import math

import numpy as np
import torch

from draft.application.train_draft_agent_rl import (
    RLExample,
    TrainDraftAgentRLConfig,
    _behaviour_anomaly_summary,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_state import TYPE_PACK, TYPE_POOL

DIM = 8  # concat_width 8+4+4+8 = 24 is divisible by the default 8 heads


def _ex(learner: bool, action_token: int = 0) -> RLExample:
    types = [TYPE_PACK, TYPE_PACK, TYPE_POOL]
    n = len(types)
    return RLExample(
        draft_index=0,
        card_idx=np.arange(n, dtype=np.int32),
        type_idx=np.array(types, dtype=np.int8),
        packs_ago=np.zeros(n, dtype=np.int8),
        pick_ago=np.zeros(n, dtype=np.int8),
        pack_number=1,
        pick_number=1,
        action_token=action_token,
        learner_active=learner,
        critic_active=True,
        reward=1.0,
    )


def _config() -> TrainDraftAgentRLConfig:
    return TrainDraftAgentRLConfig(
        checkpoint=None, learner_agents=("g",), rollout_temperature=1.0, batch_size=8,
    )


def test_summary_counts_only_learner_picks() -> None:
    torch.manual_seed(0)
    ref = DraftAgentModel(DraftAgentConfig(embedding_dim=DIM, packs=1, P=4))
    table = np.random.default_rng(0).standard_normal((6, DIM)).astype(np.float32)
    examples = [_ex(True), _ex(True), _ex(False)]  # 2 learner, 1 not
    summary = _behaviour_anomaly_summary(
        ref, examples, table, 0.0, 1.0, _config(), torch.device("cpu"),
    )
    assert summary.n_learner == 2
    assert 0.0 <= summary.frac_below_floor <= 1.0
    # Behaviour log-prob is a log of a probability → ≤ 0 and finite.
    assert math.isfinite(summary.mean_behaviour_logprob)
    assert summary.mean_behaviour_logprob <= 0.0


def test_summary_handles_no_learner_picks() -> None:
    torch.manual_seed(0)
    ref = DraftAgentModel(DraftAgentConfig(embedding_dim=DIM, packs=1, P=4))
    table = np.random.default_rng(0).standard_normal((6, DIM)).astype(np.float32)
    summary = _behaviour_anomaly_summary(
        ref, [_ex(False)], table, 0.0, 1.0, _config(), torch.device("cpu"),
    )
    assert summary.n_learner == 0
    assert summary.n_total == 0
    assert summary.frac_below_floor == 0.0


def test_summary_subsamples_large_corpora() -> None:
    # Above the floor, the check scans only a capped subset (1% floored at 2000),
    # not the whole corpus — so it stays fast on million-pick corpora.
    from draft.application.train_draft_agent_rl import _ANOMALY_SAMPLE_FLOOR

    torch.manual_seed(0)
    ref = DraftAgentModel(DraftAgentConfig(embedding_dim=DIM, packs=1, P=4))
    table = np.random.default_rng(0).standard_normal((6, DIM)).astype(np.float32)
    n_total = _ANOMALY_SAMPLE_FLOOR + 500   # just above the floor
    summary = _behaviour_anomaly_summary(
        ref, [_ex(True) for _ in range(n_total)], table, 0.0, 1.0,
        _config(), torch.device("cpu"),
    )
    assert summary.n_total == n_total
    assert summary.n_learner == _ANOMALY_SAMPLE_FLOOR   # capped at the floor
