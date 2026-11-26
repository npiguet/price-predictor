"""The ability encoder: one ability line to its bottleneck vector ``e``.

A transformer pooled through ``[CLS]`` to a small fixed width, so whatever the
effect head needs about an ability has to be in the vector that ships in the
cache. Each token adds a role embedding (FR-069), because ``{R}`` in a cost and
in an effect mean opposite things; only prose carries roles, since the spans
index the prose. A number adds a monotone embedding, so its order is in the
geometry rather than learned per digit. A model architecture is domain in this
project (constitution Principle IV).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from effects.domain.provenance import Role

#: Fixed order, so an embedding row means the same role across checkpoints.
#: Row 0 is "no role", a token outside every span.
ROLE_ORDER: tuple[Role, ...] = tuple(Role)
NO_ROLE_INDEX = 0

# Defaults of `--encoder-d-model` and `--encoder-layers` (FR-061), and the size
# a checkpoint that records neither loads at.
ENCODER_D_MODEL = 256
ENCODER_N_LAYERS = 4
ENCODER_N_HEADS = 4
FF_MULTIPLIER = 4
DROPOUT = 0.1

MAX_ABILITY_TOKENS = 512


@dataclass(frozen=True, slots=True)
class AbilityEncoderConfig:
    """Architecture of the ability encoder, carried so a checkpoint is
    self-describing. No noise here: the batcher applies it (FR-057)."""

    vocab_size: int
    e_dim: int = 256
    d_model: int = ENCODER_D_MODEL
    n_layers: int = ENCODER_N_LAYERS
    n_heads: int = ENCODER_N_HEADS
    #: 0 means "follow ``d_model``"; resolved in ``__post_init__``.
    ff_dim: int = 0
    dropout: float = DROPOUT
    max_seq_len: int = MAX_ABILITY_TOKENS

    def __post_init__(self) -> None:
        if self.ff_dim == 0 and self.d_model > 0:
            object.__setattr__(self, "ff_dim", self.d_model * FF_MULTIPLIER)
        for name in ("vocab_size", "e_dim", "d_model", "n_layers", "n_heads",
                     "ff_dim", "max_seq_len"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )


class MonotoneNumberEmbedding(nn.Module):
    """``base + log1p(n) * direction``: ``log1p`` because 1 against 2 damage
    decides a combat and 17 against 18 rarely does."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.zeros(d_model))
        self.direction = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.base, std=0.02)
        nn.init.normal_(self.direction, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``values``: ``(...,)`` numbers. Returns ``(..., d_model)``."""
        scaled = torch.log1p(values.clamp(min=0.0)).unsqueeze(-1)
        return self.base + scaled * self.direction


class AbilityEncoder(nn.Module):
    """Tokens of one ability line to its ``e`` vector."""

    def __init__(self, config: AbilityEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        # +1 for "no role": a token outside every span.
        self.role_embedding = nn.Embedding(len(ROLE_ORDER) + 1, config.d_model)
        self.number_embedding = MonotoneNumberEmbedding(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.n_layers,
            # norm_first makes torch's nested-tensor fast path inapplicable.
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.to_e = nn.Linear(config.d_model, config.e_dim)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        role_ids: torch.Tensor | None = None,
        numbers: torch.Tensor | None = None,
        number_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a padded batch of ``(batch, seq)`` lines, ``[CLS]`` first.

        Returns ``(e, hidden)``: the ``(batch, e_dim)`` bottleneck and the
        ``(batch, seq, d_model)`` token outputs the MLM auxiliary reads.
        """
        batch, seq = token_ids.shape
        positions = torch.arange(seq, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        if role_ids is not None:
            hidden = hidden + self.role_embedding(role_ids)
        if numbers is not None:
            numeric = self.number_embedding(numbers)
            if number_mask is not None:
                numeric = numeric * number_mask.unsqueeze(-1).to(numeric.dtype)
            hidden = hidden + numeric
        hidden = self.dropout(hidden)

        padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        hidden = self.norm(hidden)

        e = self.to_e(hidden[:, 0, :])
        return e, hidden
