"""Sealed encoder: token + card encoder + regression head for winnability.

Trained from scratch on per-card winnability targets aggregated from
``cards-played.txt``. The regression head is used only during training and
filtered out at save time so the persisted ``.pt`` carries only encoder
weights (FR-020).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SealedEncoderConfig:
    """Architecture knobs for the sealed encoder.

    ``d_model`` and ``ff_dim`` are hardcoded constants (FR-022); the rest
    are exposed via CLI flags. ``vocab_size`` and ``max_seq_len`` come from
    the loaded vocab and the corpus measurement, respectively.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int
    max_seq_len: int
    dropout: float
    n_pool_queries: int

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "d_model", "n_layers", "n_heads",
            "ff_dim", "max_seq_len", "n_pool_queries",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )
        if self.d_model % self.n_pool_queries != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_pool_queries "
                f"({self.n_pool_queries})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


class _TokenEncoder(nn.Module):
    """Token + positional embedding + dropout."""

    def __init__(self, config: SealedEncoderConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.dropout(x)


class _MultiQueryAttentionPool(nn.Module):
    """Pool a token sequence into a fixed-size vector via K learned queries.

    Owns ``K = n_pool_queries`` learned query vectors of length
    ``d_model / K`` each. For each query, runs single-head attention against
    the contextualized token sequence (key/value = encoder output) and
    concatenates the K outputs into a ``(B, d_model)`` vector. Padding is
    masked using ``key_padding_mask``.
    """

    def __init__(self, d_model: int, n_pool_queries: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_pool_queries != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_pool_queries "
                f"({n_pool_queries})"
            )
        self.n_pool_queries = n_pool_queries
        self.head_dim = d_model // n_pool_queries
        self.queries = nn.Parameter(
            torch.randn(1, n_pool_queries, self.head_dim) * 0.02,
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=self.head_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.kv_proj = nn.Linear(d_model, n_pool_queries * self.head_dim)

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        kv = self.kv_proj(x)
        kv = kv.view(batch_size, x.size(1), self.n_pool_queries, self.head_dim)
        # nn.MultiheadAttention's key_padding_mask: True = ignore (pad).
        key_padding_mask = attention_mask == 0
        outputs: list[torch.Tensor] = []
        for k in range(self.n_pool_queries):
            q = self.queries[:, k:k + 1, :].expand(batch_size, -1, -1)
            kv_k = kv[:, :, k, :]
            attn_out, _ = self.attn(q, kv_k, kv_k, key_padding_mask=key_padding_mask)
            outputs.append(attn_out.squeeze(1))
        return torch.cat(outputs, dim=-1)


class _CardEncoderBlock(nn.Module):
    """Transformer encoder stack + dual pool (multi-query attention ‖ max).

    Output shape is ``(B, 2 * d_model)``: half from the multi-query
    attention pool and half from element-wise max pooling, mirroring the
    price-side ``cat([max, mean])`` layout but with a learned attention
    pool on the second half (per data-model.md §"Multi-query attention pool").
    """

    def __init__(self, config: SealedEncoderConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.attn_pool = _MultiQueryAttentionPool(
            config.d_model, config.n_pool_queries, config.dropout,
        )

    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        padding_mask = attention_mask == 0
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        attn_pooled = self.attn_pool(encoded, attention_mask)

        padding_mask_3d = padding_mask.unsqueeze(-1)
        max_input = encoded.masked_fill(padding_mask_3d, float("-inf"))
        max_pooled = max_input.max(dim=1).values

        return torch.cat([attn_pooled, max_pooled], dim=-1)


class SealedEncoderModel(nn.Module):
    """Encoder + regression head for the winnability target.

    Three child modules:
      * ``token_encoder`` — token + positional embedding + dropout.
      * ``card_encoder`` — transformer stack + dual pool (attention ‖ max).
      * ``regression_head`` — Linear(2*d_model, 1) + Sigmoid; training-only.

    The state-dict prefix layout (``token_encoder.*`` / ``card_encoder.*`` /
    ``regression_head.*``) lets the save path filter the head out by key
    prefix at save time without monkey-patching the model.
    """

    def __init__(self, config: SealedEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = _TokenEncoder(config)
        self.card_encoder = _CardEncoderBlock(config)
        self.regression_head = nn.Sequential(
            nn.Linear(2 * config.d_model, 1),
            nn.Sigmoid(),
        )

    def _encode_and_pool(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.token_encoder(input_ids)
        return self.card_encoder(x, attention_mask)

    @torch.no_grad()
    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the pooled card embedding without the regression head.

        Returns:
            (batch_size, 2 * d_model)
        """
        return self._encode_and_pool(input_ids, attention_mask)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full training path: encoder → pool → head → sigmoid.

        Returns:
            (batch_size,) winnability prediction in ``[0, 1]``.
        """
        pooled = self._encode_and_pool(input_ids, attention_mask)
        return self.regression_head(pooled).squeeze(-1)
