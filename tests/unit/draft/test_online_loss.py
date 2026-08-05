"""Gen-3 single-term GRPO loss: -mean(A · logpi_T(a|s)) over learner picks.

Covers data-model §5 and spec FR-010.
"""

from __future__ import annotations

import numpy as np
import torch

from draft.application.train_draft_agent_online import (
    OnlineExample,
    _collate,
    _compute_loss,
)
from draft.domain.draft_agent_model import DraftAgentConfig, DraftAgentModel
from draft.domain.draft_state import TYPE_PACK, TYPE_POOL
from price_predictor.infrastructure.torch_training import masked_log_softmax

_DIM = 8
_PACKS = 3
_P = 4


def _config() -> DraftAgentConfig:
    return DraftAgentConfig(
        embedding_dim=_DIM, packs=_PACKS, P=_P, n_layers=1, n_heads=2,
    )


def _example(advantage: float, *, n_pool: int = 2, n_pack: int = 3,
             action_offset: int = 0) -> OnlineExample:
    n = n_pool + n_pack
    type_idx = np.array([TYPE_POOL] * n_pool + [TYPE_PACK] * n_pack, dtype=np.int8)
    return OnlineExample(
        card_idx=np.arange(n, dtype=np.int32),
        type_idx=type_idx,
        packs_ago=np.zeros(n, dtype=np.int8),
        pick_ago=np.zeros(n, dtype=np.int8),
        pack_number=1,
        pick_number=1,
        action_token=n_pool + action_offset,
        advantage=advantage,
    )


def _table(rows: int = 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((rows, _DIM)).astype(np.float32)


def _forward_logits(model: DraftAgentModel, batch) -> torch.Tensor:
    logits, _ = model(
        batch.card_emb, batch.type_idx, batch.packs_ago, batch.pick_ago,
        batch.card_mask, batch.pack_number, batch.pick_number,
    )
    return logits


def test_loss_equals_negative_mean_advantage_weighted_logp() -> None:
    torch.manual_seed(0)
    model = DraftAgentModel(_config())
    model.eval()
    device = torch.device("cpu")
    examples = [_example(1.5), _example(-0.5, action_offset=2), _example(0.25)]
    batch = _collate(examples, _table(), device)

    loss, _ = _compute_loss(model, batch, temperature=1.0)

    with torch.no_grad():
        logits = _forward_logits(model, batch)
        logp = masked_log_softmax(logits, batch.pack_mask)
        taken = logp.gather(1, batch.action_token.unsqueeze(1)).squeeze(1)
        expected = -(batch.advantage * taken).mean()
    assert torch.allclose(loss, expected, atol=1e-6)


def test_log_probs_are_normalised_over_pack_positions_only() -> None:
    torch.manual_seed(1)
    model = DraftAgentModel(_config())
    model.eval()
    batch = _collate([_example(1.0)], _table(), torch.device("cpu"))

    with torch.no_grad():
        logits = _forward_logits(model, batch)
        logp = masked_log_softmax(logits, batch.pack_mask)
    # The PACK block's probabilities sum to 1; non-PACK positions are -inf.
    probs = logp.exp()
    assert torch.allclose(probs[batch.pack_mask].sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.isinf(logp[~batch.pack_mask]).all()


def test_temperature_scales_logits_before_the_softmax() -> None:
    torch.manual_seed(2)
    model = DraftAgentModel(_config())
    model.eval()
    device = torch.device("cpu")
    examples = [_example(1.0), _example(-1.0, action_offset=1)]
    batch = _collate(examples, _table(), device)

    temp = 2.0
    loss, _ = _compute_loss(model, batch, temperature=temp)

    with torch.no_grad():
        logits = _forward_logits(model, batch)
        logp = masked_log_softmax(logits / temp, batch.pack_mask)
        taken = logp.gather(1, batch.action_token.unsqueeze(1)).squeeze(1)
        expected = -(batch.advantage * taken).mean()
    assert torch.allclose(loss, expected, atol=1e-6)

    loss_t1, _ = _compute_loss(model, batch, temperature=1.0)
    assert not torch.allclose(loss, loss_t1, atol=1e-6)


def test_zero_advantage_gives_zero_loss_and_no_gradient() -> None:
    torch.manual_seed(3)
    model = DraftAgentModel(_config())
    model.eval()
    batch = _collate([_example(0.0), _example(0.0)], _table(), torch.device("cpu"))

    loss, _ = _compute_loss(model, batch, temperature=1.0)
    assert abs(float(loss.detach())) < 1e-7

    loss.backward()
    for p in model.policy_head.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))


def test_critic_head_receives_no_gradient() -> None:
    """The critic is carried through untouched — it is not in the loss (FR-027)."""
    torch.manual_seed(4)
    model = DraftAgentModel(_config())
    model.eval()
    batch = _collate([_example(2.0), _example(-1.0)], _table(), torch.device("cpu"))

    loss, _ = _compute_loss(model, batch, temperature=1.0)
    model.zero_grad(set_to_none=True)
    loss.backward()

    for p in model.critic_head.parameters():
        assert p.grad is None or torch.allclose(p.grad, torch.zeros_like(p.grad))
    # …while the policy head genuinely moves.
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.policy_head.parameters()
    )


def test_loss_is_finite_with_ragged_padded_batches() -> None:
    torch.manual_seed(5)
    model = DraftAgentModel(_config())
    model.eval()
    examples = [
        _example(1.0, n_pool=1, n_pack=2),
        _example(-1.0, n_pool=5, n_pack=4, action_offset=3),
        _example(0.5, n_pool=3, n_pack=1),
    ]
    batch = _collate(examples, _table(), torch.device("cpu"))

    loss, _ = _compute_loss(model, batch, temperature=1.5)
    loss.backward()

    assert torch.isfinite(loss)
    for p in model.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
