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

    def _embed(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Shared embedding computation: token+position embed → encoder → pool.

        Returns:
            (batch_size, 2 * d_model) — cat([max_pooled, mean_pooled])

        Gradients flow through this method; use encode() for no-grad inference.
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)
        padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        padding_mask_3d = (attention_mask == 0).unsqueeze(-1)
        x_max = x.masked_fill(padding_mask_3d, float("-inf"))
        max_pooled = x_max.max(dim=1).values
        x_mean = x.masked_fill(padding_mask_3d, 0.0)
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = x_mean.sum(dim=1) / lengths
        return torch.cat([max_pooled, mean_pooled], dim=-1)  # (batch, 2*d_model)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return pooled card embedding without meta or output head.

        Returns:
            (batch_size, 2 * d_model) — cat([max_pooled, mean_pooled])
        """
        return self._embed(input_ids, attention_mask)

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
        pooled_embed = self._embed(input_ids, attention_mask)  # (batch, 2*d_model)
        pooled = torch.cat([pooled_embed, meta], dim=-1)  # (batch, 2*d_model + meta_dim)
        pooled = self.output_dropout(pooled)
        logits = self.output_head(pooled).squeeze(-1)
        return logits


class AuxiliaryTrainingModel(nn.Module):
    """Wrapper that adds 20 auxiliary linear heads to CardPriceTransformerModel.

    Used only during training. Save checkpoint with save_model(wrapper.base, ...).
    The 20 aux heads are discarded after training.

    Classification heads (indices 0-6, 14-19) output (batch,).
    Ordinal heads (indices 7-13) output (batch, K) where K = len(class_list).
    """

    def __init__(
        self,
        base: CardPriceTransformerModel,
        ordinal_class_maps: dict[int, list[float]],
    ) -> None:
        """Create auxiliary model with variable-output heads.

        Args:
            base: The base price predictor model.
            ordinal_class_maps: Maps head index → sorted list of class values.
                Heads present in this dict are ordinal and get nn.Linear(d_emb, K)
                where K = len(classes). All other heads are binary classifiers with
                nn.Linear(d_emb, 1).
        """
        super().__init__()
        self.base = base
        d_emb = 2 * base.config.d_model
        heads: list[nn.Module] = []
        for i in range(20):
            if i in ordinal_class_maps:
                K = len(ordinal_class_maps[i])
                heads.append(nn.Linear(d_emb, K))
            else:
                heads.append(nn.Linear(d_emb, 1))
        self.aux_heads = nn.ModuleList(heads)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        meta: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run forward pass returning price prediction and auxiliary predictions.

        Returns:
            price_pred: (batch_size,)
            aux_preds: list of 20 tensors; classification heads → (batch_size,),
                ordinal heads → (batch_size, K)
        """
        pooled_embed = self.base._embed(input_ids, attention_mask)  # (batch, 2*d_model)
        # Price prediction through base model's output head
        pooled_with_meta = torch.cat([pooled_embed, meta], dim=-1)
        pooled_with_meta = self.base.output_dropout(pooled_with_meta)
        price_pred = self.base.output_head(pooled_with_meta).squeeze(-1)
        # Auxiliary predictions (raw pooled embedding, no dropout)
        aux_preds = []
        for head in self.aux_heads:
            out = head(pooled_embed)
            if out.shape[-1] == 1:
                aux_preds.append(out.squeeze(-1))  # (batch,) for classification
            else:
                aux_preds.append(out)              # (batch, K) for ordinal
        return price_pred, aux_preds
