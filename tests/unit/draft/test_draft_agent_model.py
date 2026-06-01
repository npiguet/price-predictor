"""DraftAgentModel architecture + forward shapes (FR-023 … FR-028, SC-006)."""

from __future__ import annotations

import pytest
import torch

from draft.domain.draft_agent_model import (
    DraftAgentArchitectureError,
    DraftAgentConfig,
    DraftAgentModel,
)
from draft.domain.draft_state import NUM_TYPES


def test_default_d_model_is_concat_width_no_projection() -> None:
    cfg = DraftAgentConfig(embedding_dim=40, packs=3, P=15)
    assert cfg.d_model == 40 + NUM_TYPES + cfg.d_packs_ago + cfg.d_pick_ago
    model = DraftAgentModel(cfg)
    assert isinstance(model.input_projection, torch.nn.Identity)


def test_non_default_d_model_inserts_linear() -> None:
    cfg = DraftAgentConfig(embedding_dim=40, packs=3, P=15, d_model=64)
    model = DraftAgentModel(cfg)
    assert isinstance(model.input_projection, torch.nn.Linear)
    assert model.input_projection.in_features == cfg.concat_width
    assert model.input_projection.out_features == 64


def test_indivisible_heads_raises_fast() -> None:
    with pytest.raises(DraftAgentArchitectureError):
        DraftAgentConfig(embedding_dim=40, packs=3, P=15, d_model=64, n_heads=7)


def _forward(cfg: DraftAgentConfig, b: int, n: int):
    model = DraftAgentModel(cfg)
    model.eval()
    torch.manual_seed(0)
    card_emb = torch.randn(b, n, cfg.embedding_dim)
    type_idx = torch.randint(0, NUM_TYPES, (b, n))
    packs_ago = torch.randint(0, 3, (b, n))
    pick_ago = torch.randint(0, cfg.P, (b, n))
    card_mask = torch.ones(b, n, dtype=torch.bool)
    pack_number = torch.randint(1, cfg.packs + 1, (b,))
    pick_number = torch.randint(1, cfg.P + 1, (b,))
    with torch.no_grad():
        return model(card_emb, type_idx, packs_ago, pick_ago, card_mask,
                     pack_number, pick_number)


def test_forward_head_shapes() -> None:
    cfg = DraftAgentConfig(embedding_dim=40, packs=3, P=15)
    logits, critic = _forward(cfg, b=4, n=10)
    assert logits.shape == (4, 10)  # one policy logit per card token
    assert critic.shape == (4,)     # one critic scalar per example (CONTEXT token)


def test_padding_does_not_change_real_token_outputs() -> None:
    """Masked padding must not leak into the real tokens' representations."""
    cfg = DraftAgentConfig(embedding_dim=16, packs=3, P=15)
    model = DraftAgentModel(cfg)
    model.eval()
    torch.manual_seed(1)
    n = 6
    card_emb = torch.randn(1, n, cfg.embedding_dim)
    type_idx = torch.randint(0, NUM_TYPES, (1, n))
    packs_ago = torch.randint(0, 3, (1, n))
    pick_ago = torch.randint(0, cfg.P, (1, n))
    pack_number = torch.tensor([2])
    pick_number = torch.tensor([3])

    full_mask = torch.ones(1, n, dtype=torch.bool)
    with torch.no_grad():
        logits_a, critic_a = model(
            card_emb, type_idx, packs_ago, pick_ago, full_mask, pack_number, pick_number,
        )

    # Append 3 padding tokens with garbage features but mask them out.
    pad = 3
    card_emb2 = torch.cat([card_emb, torch.randn(1, pad, cfg.embedding_dim)], dim=1)
    type_idx2 = torch.cat([type_idx, torch.randint(0, NUM_TYPES, (1, pad))], dim=1)
    packs2 = torch.cat([packs_ago, torch.randint(0, 3, (1, pad))], dim=1)
    pick2 = torch.cat([pick_ago, torch.randint(0, cfg.P, (1, pad))], dim=1)
    mask2 = torch.cat([torch.ones(1, n, dtype=torch.bool),
                       torch.zeros(1, pad, dtype=torch.bool)], dim=1)
    with torch.no_grad():
        logits_b, critic_b = model(
            card_emb2, type_idx2, packs2, pick2, mask2, pack_number, pick_number,
        )

    assert torch.allclose(logits_a, logits_b[:, :n], atol=1e-5)
    assert torch.allclose(critic_a, critic_b, atol=1e-5)
