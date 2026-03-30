"""PoolTransformerConfig and PoolTransformerModel for Stage 1 training."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PoolTransformerConfig:
    n_layers: int = 8
    n_heads: int = 8  # 520 / 8 = 65
    d_model: int = 520  # = card_embed_dim(512) + 3 flags + 5 reserved padding
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
