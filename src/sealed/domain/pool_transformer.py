"""PoolTransformerConfig and PoolTransformerModel for Stage 1 training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PoolTransformerConfig:
    n_layers: int = 8
    n_heads: int = 4  # 516 / 4 = 129; spec says 8 but 516 % 8 != 0
    d_model: int = 516  # = card_embed_dim(512) + 3 flags + 1 reserved (pick_count, available, is_land, pad)
    ff_dim: int = 2048
    n_slots: int = 96
    card_embed_dim: int = 512
    dropout: float = 0.1


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
        self.head = nn.Linear(config.d_model, config.n_slots)

    def forward(self, slot_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slot_features: [batch, n_slots, d_model]
        Returns:
            logits: [batch, n_slots]
        """
        enc_out = self.encoder(slot_features)  # [batch, n_slots, d_model]
        pooled = enc_out.mean(dim=1)           # [batch, d_model]
        return self.head(pooled)               # [batch, n_slots]
