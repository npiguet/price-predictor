"""PoolTransformerConfig and PoolTransformerModel for Stage 1 training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PoolTransformerConfig:
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 536   # must be divisible by n_heads; set from card_embed_dim at startup
    ff_dim: int = 2048
    n_slots: int = 96
    card_embed_dim: int = 527  # 2*d_model(256) + 15 mana features; actual value set at startup
    dropout: float = 0.1

    @classmethod
    def from_embed_dim(cls, card_embed_dim: int, **overrides) -> "PoolTransformerConfig":
        """Construct config from detected embedding dimension.

        d_model = ceil((card_embed_dim + 8) / n_heads) * n_heads so that
        the transformer receives head-divisible input. The 8 extra slots are
        flag dimensions added by _build_base_tensor (pick_count, available, 6 reserved).
        """
        n_heads = overrides.pop("n_heads", 8)
        raw = card_embed_dim + 8
        d_model = ((raw + n_heads - 1) // n_heads) * n_heads
        return cls(card_embed_dim=card_embed_dim, d_model=d_model, n_heads=n_heads, **overrides)


class PoolTransformerModel(nn.Module):
    def __init__(self, config: PoolTransformerConfig) -> None:
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, slot_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_features: [batch, n_slots, d_model]
        Returns:
            logits: [batch, n_slots]
        """
        enc_out = self.encoder(slot_features)      # [batch, n_slots, d_model]
        return self.head(enc_out).squeeze(-1)       # [batch, n_slots]
