"""Encoder-only transformer model for card price regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from price_predictor.domain.entities import TransformerConfig


class CardPriceTransformerModel(nn.Module):
    """Transformer encoder that predicts shifted-log card prices from token IDs.

    Architecture: cat([max_pooled(d_model), mean_pooled(d_model), meta(meta_dim)])
    → Linear(2*d_model + meta_dim, regression_hidden_dim) → ReLU → Linear(..., 1)
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.embed_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        self.output_dropout = nn.Dropout(config.dropout)
        head_input_dim = 2 * config.d_model + config.meta_dim
        self.output_head = nn.Sequential(
            nn.Linear(head_input_dim, config.regression_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.regression_hidden_dim, 1),
        )

    def _encode_and_pool(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Embed, run encoder, and return cat([max_pooled, mean_pooled]).

        Padding (attention_mask == 0) is filled with -inf for max-pooling and
        zeroed for mean-pooling so that PAD tokens never contribute.

        Returns:
            (batch_size, 2 * d_model)
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        padding_mask_3d = padding_mask.unsqueeze(-1)
        x_max = x.masked_fill(padding_mask_3d, float("-inf"))
        max_pooled = x_max.max(dim=1).values

        x_mean = x.masked_fill(padding_mask_3d, 0.0)
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = x_mean.sum(dim=1) / lengths

        return torch.cat([max_pooled, mean_pooled], dim=-1)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return pooled card embedding without meta or output head.

        Returns:
            (batch_size, 2 * d_model) — cat([max_pooled, mean_pooled])
        """
        return self._encode_and_pool(input_ids, attention_mask)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs.
            attention_mask: (batch_size, seq_len) 1 for real tokens, 0 for padding.
            meta: (batch_size, meta_dim) side-channel metadata vector.

        Returns:
            (batch_size,) predictions in shifted-log-price space.
        """
        pooled_text = self._encode_and_pool(input_ids, attention_mask)
        pooled = torch.cat([pooled_text, meta], dim=-1)
        pooled = self.output_dropout(pooled)
        return self.output_head(pooled).squeeze(-1)
