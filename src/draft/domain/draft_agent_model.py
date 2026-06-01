"""Two-headed draft agent: SAB trunk + imitation policy + MC-regression critic.

Follows ``sealed/domain/picker_model.py`` conventions exactly (SAB trunk over a
card set, optional input projection, fail-fast ``d_model % n_heads`` check) but
is a distinct sibling: it adds typed tokens, a ``CONTEXT`` token, learned
recency/context embedding tables, and a *second* head. The ``SAB`` primitive is
imported from the scorer rather than re-implemented (FR-024).

Input layout (per example, before padding): one ``CONTEXT`` token followed by
the ``POOL``/``PACK``/``PASSED``/``TAKEN`` card tokens (data-model §3). The
policy head emits one logit per ``PACK`` token (masked softmax over ``PACK``
only); the critic head emits one scalar on the ``CONTEXT`` token (FR-027,
FR-028).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from draft.domain.draft_state import NUM_TYPES
from sealed.domain.scorer_model import SAB


class DraftAgentArchitectureError(ValueError):
    """Raised when draft-agent architecture knobs are inconsistent (FR-026)."""


@dataclass
class DraftAgentConfig:
    """Architecture record for one draft-agent instance (data-model §4).

    ``embedding_dim`` is the ``.npz`` card-vector width; ``d_model`` is the
    transformer's internal width. When ``d_model`` is left unset it defaults to
    the concat width (frozen card vector + 4-dim type one-hot + recency
    embeddings), so no input projection is inserted; an explicit non-default
    ``d_model`` inserts a single ``Linear(concat_width, d_model)`` (FR-025).
    ``packs`` and ``P`` (pack size) come from the corpus and size the
    ``pack_number`` / ``pick_number`` / ``pick_ago`` tables.
    """

    embedding_dim: int
    packs: int
    P: int
    d_model: int | None = None
    n_layers: int = 4
    n_heads: int = 8
    ff_dim: int | None = None
    dropout: float = 0.0
    d_packs_ago: int = 4
    d_pick_ago: int = 8

    def __post_init__(self) -> None:
        if self.d_model is None:
            self.d_model = self.concat_width
        if self.ff_dim is None:
            self.ff_dim = 4 * self.d_model
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_model % self.n_heads != 0:
            raise DraftAgentArchitectureError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}); pick a head count that divides the model width."
            )

    @property
    def concat_width(self) -> int:
        """Width of a card token before the (optional) input projection."""
        return self.embedding_dim + NUM_TYPES + self.d_packs_ago + self.d_pick_ago


class DraftAgentModel(nn.Module):
    """SAB trunk over [CONTEXT, cards…] with a policy head and a critic head."""

    def __init__(self, config: DraftAgentConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model

        if config.d_model != config.concat_width:
            self.input_projection: nn.Module = nn.Linear(config.concat_width, d_model)
        else:
            self.input_projection = nn.Identity()

        # Learned recency tables. packs_ago ∈ {0,1,2}; pick_ago ∈ {0,…,P-1}.
        self.packs_ago_embed = nn.Embedding(3, config.d_packs_ago)
        self.pick_ago_embed = nn.Embedding(config.P, config.d_pick_ago)
        # CONTEXT token = pack_number embed + pick_number embed (both d_model).
        # 1-based indices, so size tables one larger and index directly.
        self.pack_number_embed = nn.Embedding(config.packs + 1, d_model)
        self.pick_number_embed = nn.Embedding(config.P + 1, d_model)

        self.sab_layers = nn.ModuleList([
            SAB(d_model, config.n_heads, config.ff_dim, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])
        self.policy_head = nn.Linear(d_model, 1)
        self.critic_head = nn.Linear(d_model, 1)

    def forward(
        self,
        card_emb: torch.Tensor,      # (B, N, embedding_dim) frozen card vectors
        type_idx: torch.Tensor,      # (B, N) long, TYPE_* indices
        packs_ago: torch.Tensor,     # (B, N) long
        pick_ago: torch.Tensor,      # (B, N) long
        card_mask: torch.Tensor,     # (B, N) bool, True = real card
        pack_number: torch.Tensor,   # (B,) long, 1-based
        pick_number: torch.Tensor,   # (B,) long, 1-based
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits (B, N), critic (B,))``.

        ``policy_logits`` is meaningful only at ``PACK`` positions; the caller
        masks the rest before softmax/CE. ``critic`` reads the ``CONTEXT`` token.
        """
        type_onehot = F.one_hot(type_idx, NUM_TYPES).to(card_emb.dtype)
        card_feat = torch.cat(
            [card_emb, type_onehot, self.packs_ago_embed(packs_ago),
             self.pick_ago_embed(pick_ago)],
            dim=-1,
        )
        card_tok = self.input_projection(card_feat)  # (B, N, d_model)

        context_tok = (
            self.pack_number_embed(pack_number) + self.pick_number_embed(pick_number)
        ).unsqueeze(1)  # (B, 1, d_model)

        x = torch.cat([context_tok, card_tok], dim=1)  # (B, 1+N, d_model)

        # key_padding_mask: True = ignore. CONTEXT (col 0) is always real.
        ctx_real = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
        attn_padding_mask = torch.cat([ctx_real, ~card_mask], dim=1)
        for sab in self.sab_layers:
            x = sab(x, attn_padding_mask=attn_padding_mask)

        critic = self.critic_head(x[:, 0, :]).squeeze(-1)        # (B,)
        policy_logits = self.policy_head(x[:, 1:, :]).squeeze(-1)  # (B, N)
        return policy_logits, critic
