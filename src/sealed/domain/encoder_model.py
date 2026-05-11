"""Sealed encoder: token + card encoder + per-card winnability heads + MLM head.

Trained from scratch under spec 016's multi-head + MLM objective: nine
regression cells per card (``score_play``, ``score_draw``, ``played_rate``,
``cast_lift``, and one ``color_lift`` per WUBRG color) plus a masked-language
modeling head reading the contextualized token sequence. The regression
heads and the MLM head are training-only and filtered out at save time so
the persisted ``.pt`` carries only encoder weights (FR-020).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

# WUBRG color order — fixed to match the per-color counter ordering used by
# ``train_encoder._build_label_map`` and the columns in ``cards-win-rates.txt``.
COLOR_ORDER: tuple[str, ...] = ("W", "U", "B", "R", "G")


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
    """Pool a token sequence into a fixed-size vector via K learned queries."""

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

    Returns ``(contextualized, pooled)`` where ``contextualized`` is the
    pre-pool transformer output ``(B, T, d_model)`` consumed by the MLM
    head per FR-015a / Decision D-9, and ``pooled`` is ``(B, 2 * d_model)``
    consumed by the regression heads.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = attention_mask == 0
        encoded = self.encoder(x, src_key_padding_mask=padding_mask)

        attn_pooled = self.attn_pool(encoded, attention_mask)

        padding_mask_3d = padding_mask.unsqueeze(-1)
        max_input = encoded.masked_fill(padding_mask_3d, float("-inf"))
        max_pooled = max_input.max(dim=1).values

        pooled = torch.cat([attn_pooled, max_pooled], dim=-1)
        return encoded, pooled


def _build_regression_heads(d_model: int) -> nn.ModuleDict:
    """Five regression-head projections (Decision D-8).

    The four signed heads (``score_play``, ``score_draw``, ``cast_lift``,
    ``color_lift``) project to ``[-1, +1]`` via ``Tanh``; ``played_rate``
    projects to ``[0, 1]`` via ``Sigmoid``. ``color_lift`` is one fused
    five-output projection with one column per WUBRG letter.
    """
    in_features = 2 * d_model
    return nn.ModuleDict({
        "score_play": nn.Sequential(nn.Linear(in_features, 1), nn.Tanh()),
        "score_draw": nn.Sequential(nn.Linear(in_features, 1), nn.Tanh()),
        "played_rate": nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid()),
        "cast_lift": nn.Sequential(nn.Linear(in_features, 1), nn.Tanh()),
        "color_lift": nn.Sequential(
            nn.Linear(in_features, len(COLOR_ORDER)), nn.Tanh(),
        ),
    })


class SealedEncoderModel(nn.Module):
    """Encoder + nine regression heads + MLM head.

    Children (state-dict prefixes shown in parentheses):
      * ``token_encoder.*`` — token + positional embedding + dropout. Saved.
      * ``card_encoder.*`` — transformer stack + dual pool. Saved.
      * ``regression_heads.*`` — five linear projections (Decision D-8).
        Training-only; **filtered out at save time** per FR-020.
      * ``mlm_head.*`` — ``Linear(d_model, vocab_size)`` reading the
        pre-pool contextualized token sequence (FR-015a). Training-only;
        **filtered out at save time** per FR-020.
    """

    def __init__(self, config: SealedEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_encoder = _TokenEncoder(config)
        self.card_encoder = _CardEncoderBlock(config)
        self.regression_heads = _build_regression_heads(config.d_model)
        self.mlm_head = nn.Linear(config.d_model, config.vocab_size)

    def _encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_encoder(input_ids)
        return self.card_encoder(x, attention_mask)

    @torch.no_grad()
    def encode(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the pooled card embedding without the heads.

        Returns:
            ``(batch_size, 2 * d_model)``
        """
        _contextualized, pooled = self._encode(input_ids, attention_mask)
        return pooled

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full training path.

        Returns a dict with keys:
          * ``score_play``, ``score_draw``, ``played_rate``, ``cast_lift``
            — each ``(B,)``, range-matched.
          * ``color_lift`` — ``(B, 5)`` (one column per WUBRG letter, in
            ``COLOR_ORDER``).
          * ``contextualized`` — ``(B, T, d_model)``, the pre-pool
            transformer output. The caller feeds the masked positions
            of this through ``mlm_head`` to get the MLM logits — the
            head is position-wise, so applying it only to the ~15 % of
            positions that were masked avoids materialising the full
            ``(B, T, vocab_size)`` tensor and an ~84 % wasted matmul.
        """
        contextualized, pooled = self._encode(input_ids, attention_mask)
        out: dict[str, torch.Tensor] = {}
        for name in ("score_play", "score_draw", "played_rate", "cast_lift"):
            out[name] = self.regression_heads[name](pooled).squeeze(-1)
        out["color_lift"] = self.regression_heads["color_lift"](pooled)
        out["contextualized"] = contextualized
        return out
