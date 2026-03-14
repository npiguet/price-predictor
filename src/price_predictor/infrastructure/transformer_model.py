"""Encoder-only transformer model for card price regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from price_predictor.domain.entities import TransformerConfig


class CardPriceTransformerModel(nn.Module):
    """Transformer encoder that predicts shifted-log card prices from token IDs."""

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
        self.output_head = nn.Linear(config.d_model, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            input_ids: (batch_size, seq_len) token IDs from BERT tokenizer.
            attention_mask: (batch_size, seq_len) 1 for real tokens, 0 for padding.

        Returns:
            (batch_size,) predictions in shifted-log-price space.
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # TransformerEncoder expects src_key_padding_mask where True = ignore
        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        # CLS extraction (position 0)
        cls_output = x[:, 0, :]
        cls_output = self.output_dropout(cls_output)
        logits = self.output_head(cls_output).squeeze(-1)
        return logits
