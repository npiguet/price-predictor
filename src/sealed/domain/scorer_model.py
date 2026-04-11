"""Set Transformer scorer: SAB + PMA + scoring MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class SAB(nn.Module):
    """Self-Attention Block: multihead self-attention + feedforward + LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # nn.MultiheadAttention key_padding_mask: True = ignore
        attn_mask = ~key_padding_mask if key_padding_mask is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=attn_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention: learned seed vectors attend over set elements."""

    def __init__(self, d_model: int, n_heads: int, n_seeds: int) -> None:
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.01)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seeds = self.seeds.expand(x.size(0), -1, -1)
        attn_mask = ~key_padding_mask if key_padding_mask is not None else None
        attn_out, _ = self.attn(seeds, x, x, key_padding_mask=attn_mask)
        return self.norm(seeds + attn_out)


class SetTransformerScorer(nn.Module):
    """Scores a deck (unordered set of card vectors) to produce a scalar quality score.

    Architecture:
        1. Normalize deterministic features (indices 512-543)
        2. Stack of SAB layers (self-attention over cards)
        3. PMA pooling to fixed-size representation
        4. Scoring MLP → scalar
    """

    def __init__(
        self,
        d_model: int = 544,
        n_layers: int = 2,
        n_heads: int = 4,
        n_seeds: int = 4,
        d_ff: int = 1088,
        mlp_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Normalization buffers for indices 512-543 (deterministic features)
        self.register_buffer("feat_mean", torch.zeros(32))
        self.register_buffer("feat_std", torch.ones(32))

        # Self-attention layers
        self.sab_layers = nn.ModuleList([SAB(d_model, n_heads, d_ff) for _ in range(n_layers)])

        # Pooling
        self.pma = PMA(d_model, n_heads, n_seeds)

        # Scoring MLP
        self.mlp = nn.Sequential(
            nn.Linear(n_seeds * d_model, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize deterministic features (last 32 dims) using stored statistics."""
        x = x.clone()
        x[:, :, 512:] = (x[:, :, 512:] - self.feat_mean) / self.feat_std
        return x

    def forward(self, cards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Score a batch of decks.

        Args:
            cards: (batch, max_cards, 544) card feature vectors.
            mask: (batch, max_cards) boolean mask, True = real card, False = padding.

        Returns:
            (batch, 1) scalar score per deck.
        """
        x = self._normalize(cards)

        for sab in self.sab_layers:
            x = sab(x, key_padding_mask=mask)

        pooled = self.pma(x, key_padding_mask=mask)  # (batch, n_seeds, d_model)
        flat = pooled.flatten(start_dim=1)  # (batch, n_seeds * d_model)
        return self.mlp(flat)
