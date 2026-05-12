"""Set Transformer scorer: SAB + PMA + scoring MLP."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from sealed.domain.card_embedding_layout import DET_FEATURE_DIM, total_dim


@dataclass
class ScorerConfig:
    # Default = dual-pool 2*256 text + 32 features; train-scorer derives the
    # actual width from the .npz embedding cache instead.
    d_model: int = total_dim(256)
    n_layers: int = 6
    n_heads: int = 4
    n_seeds: int = 4
    d_ff: int = 1088
    mlp_hidden: int = 256
    dropout: float = 0.2


class SAB(nn.Module):
    """Self-Attention Block: multihead self-attention + feedforward + LayerNorm.

    The ``attn_padding_mask`` argument follows ``nn.MultiheadAttention``'s
    convention (``True`` marks positions to ignore). The scorer's ``forward``
    method inverts the user-facing "True = real card" mask once and passes the
    same inverted tensor to every SAB and PMA layer.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ff_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_padding_mask)
        x = self.norm1(x + self.attn_dropout(attn_out))
        x = self.norm2(x + self.ff_dropout(self.ff(x)))
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention: learned seed vectors attend over set elements.

    See ``SAB`` for the ``attn_padding_mask`` convention.
    """

    def __init__(self, d_model: int, n_heads: int, n_seeds: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.01)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, attn_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seeds = self.seeds.expand(x.size(0), -1, -1)
        attn_out, _ = self.attn(seeds, x, x, key_padding_mask=attn_padding_mask)
        return self.norm(seeds + self.attn_dropout(attn_out))


class SetTransformerScorer(nn.Module):
    """Scores a deck (unordered set of card vectors) to produce a scalar quality score.

    Architecture:
        1. Normalize deterministic features (the trailing ``DET_FEATURE_DIM`` dims)
        2. Stack of SAB layers (self-attention over cards)
        3. PMA pooling to fixed-size representation
        4. Scoring MLP → scalar
    """

    def __init__(self, config: ScorerConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.det_feature_offset = config.d_model - DET_FEATURE_DIM

        self.register_buffer("feat_mean", torch.zeros(DET_FEATURE_DIM))
        self.register_buffer("feat_std", torch.ones(DET_FEATURE_DIM))

        self.sab_layers = nn.ModuleList([
            SAB(config.d_model, config.n_heads, config.d_ff, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])
        self.pma = PMA(
            config.d_model, config.n_heads, config.n_seeds, dropout=config.dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.n_seeds * config.d_model, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, config.mlp_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, 1),
        )

    def normalize_features(self, cards: torch.Tensor) -> torch.Tensor:
        """Normalize the trailing deterministic-feature slice using stored stats.

        Returns a fresh tensor; never mutates the input. Works on any tensor
        whose last dimension is ``d_model`` — typically ``(B, L, d_model)``
        for a deck batch or ``(P, d_model)`` for a flat pool array.
        """
        out = cards.clone()
        offset = self.det_feature_offset
        out[..., offset:] = (out[..., offset:] - self.feat_mean) / self.feat_std
        return out

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize_features(x)

    def forward_prenormalized(
        self, cards: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Like ``forward``, but assumes ``cards`` are already normalized.

        Used by callers that batch many forward passes over the same card pool
        — they pre-normalize the pool once and skip the per-forward
        ``.clone()`` in ``normalize_features``.
        """
        # nn.MultiheadAttention's key_padding_mask: True = ignore.
        attn_padding_mask = ~mask
        x = cards
        for sab in self.sab_layers:
            x = sab(x, attn_padding_mask=attn_padding_mask)

        pooled = self.pma(x, attn_padding_mask=attn_padding_mask)  # (batch, n_seeds, d_model)
        flat = pooled.flatten(start_dim=1)  # (batch, n_seeds * d_model)
        return self.mlp(flat)

    def forward(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score a batch of decks.

        Args:
            cards: ``(batch, max_cards, d_model)`` card feature vectors.
            mask: ``(batch, max_cards)`` boolean mask, True = real card, False = padding.

        Returns:
            ``(batch, 1)`` scalar score per deck.
        """
        return self.forward_prenormalized(self.normalize_features(cards), mask)
