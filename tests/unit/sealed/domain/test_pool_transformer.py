"""T005 — Unit tests for PoolTransformerModel."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from sealed.domain.pool_transformer import PoolTransformerConfig, PoolTransformerModel

# Miniaturized config: d_model=16 = card_embed_dim(8) + 8 padding; 16 % 2 == 0 ✓
MINI = PoolTransformerConfig(
    n_slots=4,
    d_model=16,
    n_layers=1,
    n_heads=2,
    card_embed_dim=8,
    ff_dim=16,
    dropout=0.0,
)


def test_forward_output_shape():
    model = PoolTransformerModel(MINI)
    model.eval()
    batch = 3
    x = torch.zeros(batch, MINI.n_slots, MINI.d_model)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch, MINI.n_slots), f"Expected ({batch}, {MINI.n_slots}), got {logits.shape}"


def test_forward_single_batch():
    model = PoolTransformerModel(MINI)
    model.eval()
    x = torch.randn(1, MINI.n_slots, MINI.d_model)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, MINI.n_slots)


def test_softmax_is_valid_distribution():
    """Softmax over logits should produce a valid probability distribution."""
    model = PoolTransformerModel(MINI)
    model.eval()
    x = torch.randn(1, MINI.n_slots, MINI.d_model)
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits[0], dim=-1)
    assert abs(probs.sum().item() - 1.0) < 1e-5
    assert (probs >= 0).all()


def test_config_stored_on_model():
    model = PoolTransformerModel(MINI)
    assert model.config is MINI


def test_production_config_valid_n_heads():
    """Production default n_heads=8 must divide d_model=520 evenly."""
    from sealed.domain.pool_transformer import PoolTransformerConfig
    cfg = PoolTransformerConfig()
    assert cfg.d_model % cfg.n_heads == 0, (
        f"d_model={cfg.d_model} is not divisible by n_heads={cfg.n_heads}"
    )
