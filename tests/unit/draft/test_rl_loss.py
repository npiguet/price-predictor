"""RL loss decomposition: policy/value/entropy/KL, masking, temperature (spec 020)."""

from __future__ import annotations

import numpy as np
import torch

from draft.application.train_draft_agent_rl import (
    RLExample,
    TrainDraftAgentRLConfig,
    _collate,
    _compute_loss,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_state import TYPE_PACK, TYPE_POOL, TYPE_TAKEN

DIM = 8  # concat_width 8+4+4+8 = 24 is divisible by the default 8 heads


def _model() -> DraftAgentModel:
    torch.manual_seed(0)
    return DraftAgentModel(DraftAgentConfig(embedding_dim=DIM, packs=1, P=4))


def _table() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((6, DIM)).astype(np.float32)


def _ex(types: list[int], action_token: int, learner: bool, reward: float, adv: float) -> RLExample:
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
        reward=reward,
        advantage=adv,
    )


def _config(**kw) -> TrainDraftAgentRLConfig:
    return TrainDraftAgentRLConfig(
        checkpoint=None, learner_agents=("gen-k",), rollout_temperature=1.0, **kw,
    )


def test_total_is_weighted_sum_of_components() -> None:
    model = _model()
    ref = _model()  # different init → nonzero KL
    table = _table()
    batch = _collate(
        [_ex([TYPE_PACK, TYPE_PACK, TYPE_POOL], 0, True, 1.0, 0.5),
         _ex([TYPE_PACK, TYPE_POOL, TYPE_TAKEN], 0, False, -1.0, 0.0)],
        table, 0.0, 1.0, torch.device("cpu"),
    )
    cfg = _config(value_weight=2.0)
    out = _compute_loss(model, ref, batch, cfg, entropy_coef=0.1, kl_coef=0.3)
    expected = (
        out.policy + 2.0 * out.value - 0.1 * out.entropy + 0.3 * out.kl
    )
    assert torch.allclose(out.total, expected, atol=1e-6)
    for t in (out.total, out.policy, out.value, out.entropy, out.kl):
        assert torch.isfinite(t).all()


def test_kl_is_zero_against_identical_reference() -> None:
    model = _model()
    ref = _model()
    ref.load_state_dict(model.state_dict())  # identical → KL == 0
    table = _table()
    batch = _collate(
        [_ex([TYPE_PACK, TYPE_PACK, TYPE_POOL], 0, True, 1.0, 0.5)],
        table, 0.0, 1.0, torch.device("cpu"),
    )
    out = _compute_loss(model, ref, batch, _config(), entropy_coef=0.0, kl_coef=1.0)
    assert torch.allclose(out.kl, torch.zeros(()), atol=1e-5)
    assert out.entropy >= 0.0


def test_non_learner_batch_has_zero_policy_terms() -> None:
    model = _model()
    ref = _model()
    table = _table()
    # Both rows non-learner → only the critic (value) term is active.
    batch = _collate(
        [_ex([TYPE_PACK, TYPE_POOL], 0, False, 1.0, 0.0),
         _ex([TYPE_PACK, TYPE_TAKEN], 0, False, -1.0, 0.0)],
        table, 0.0, 1.0, torch.device("cpu"),
    )
    out = _compute_loss(model, ref, batch, _config(), entropy_coef=0.1, kl_coef=0.3)
    assert float(out.policy) == 0.0
    assert float(out.entropy) == 0.0
    assert float(out.kl) == 0.0
    assert float(out.value.detach()) > 0.0


def test_backward_is_finite_with_masked_rows() -> None:
    model = _model()
    ref = _model()
    table = _table()
    batch = _collate(
        [_ex([TYPE_PACK, TYPE_PACK, TYPE_POOL], 1, True, 1.0, -0.7),
         _ex([TYPE_PACK, TYPE_POOL, TYPE_TAKEN], 0, False, 0.5, 0.0)],
        table, 0.0, 1.0, torch.device("cpu"),
    )
    out = _compute_loss(model, ref, batch, _config(), entropy_coef=0.05, kl_coef=0.2)
    out.total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_temperature_changes_entropy() -> None:
    model = _model()
    ref = _model()
    table = _table()
    rows = [_ex([TYPE_PACK, TYPE_PACK, TYPE_PACK, TYPE_POOL], 0, True, 1.0, 0.1)]
    cold = _compute_loss(
        model, ref, _collate(rows, table, 0.0, 1.0, torch.device("cpu")),
        _config(), entropy_coef=1.0, kl_coef=0.0,
    )
    hot = _compute_loss(
        model, ref, _collate(rows, table, 0.0, 1.0, torch.device("cpu")),
        TrainDraftAgentRLConfig(
            checkpoint=None, learner_agents=("g",), rollout_temperature=5.0,
        ),
        entropy_coef=1.0, kl_coef=0.0,
    )
    # Higher temperature flattens the PACK softmax → higher entropy.
    assert float(hot.entropy.detach()) > float(cold.entropy.detach())
