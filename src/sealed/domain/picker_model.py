"""One-shot sealed deck picker: SAB trunk + per-card head + aux pool-quality head.

The picker is a policy transformer over a sealed pool. It emits one logit per
pool card (the per-card head) plus a single per-pool scalar (the aux head that
predicts the per-pool mean reward). It mirrors ``ScorerConfig`` /
``SetTransformerScorer`` but is a distinct model: no PMA, a per-token head
instead of a pooled scalar, and an extra aux head. The SAB primitive is
imported from the scorer rather than re-implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from sealed.domain.card_embedding_layout import is_land_embedding
from sealed.domain.greedy_deck_builder import NONLAND_DECK_SIZE
from sealed.domain.scorer_model import SAB


class PickerArchitectureError(ValueError):
    """Raised when picker architecture knobs are inconsistent (FR-033)."""


@dataclass
class PickerConfig:
    """Architecture record for one picker instance.

    ``embedding_dim`` is the width of the ``.npz`` cache rows the picker
    consumes (the scorer's ``ScorerConfig.d_model``). ``d_model`` is the
    picker's internal width: when it equals ``embedding_dim`` no input
    projection is inserted, otherwise a single ``Linear(embedding_dim,
    d_model)`` is inserted ahead of the first SAB (FR-002). ``d_model`` and
    ``d_ff`` resolve from the other fields in ``__post_init__`` when left
    unset, so a checkpoint round-trip (which stores resolved ints) reconstructs
    the same architecture.
    """

    embedding_dim: int
    d_model: int | None = None
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model is None:
            self.d_model = self.embedding_dim
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_model % self.n_heads != 0:
            raise PickerArchitectureError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}); pick a head count that divides the picker "
                "width."
            )


class PickerModel(nn.Module):
    """Policy transformer over a sealed pool.

    ``forward(pool_cards, pool_mask)`` returns ``(logits, aux_pred)`` where
    ``logits`` is one real value per pool card ``(B, N)`` and ``aux_pred`` is
    one scalar per pool ``(B,)``. The aux head is structurally always present;
    callers decide whether to use it.
    """

    def __init__(self, config: PickerConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        if config.d_model != config.embedding_dim:
            self.input_projection: nn.Module = nn.Linear(config.embedding_dim, d_model)
        else:
            self.input_projection = nn.Identity()
        self.sab_layers = nn.ModuleList([
            SAB(d_model, config.n_heads, config.d_ff, dropout=config.dropout)
            for _ in range(config.n_layers)
        ])
        self.per_card_head = nn.Linear(d_model, 1)
        self.aux_head = nn.Linear(d_model, 1)

    def forward(
        self, pool_cards: torch.Tensor, pool_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward pass over a batch of pools.

        Args:
            pool_cards: ``(B, max_N, embedding_dim)`` card vectors.
            pool_mask: ``(B, max_N)`` boolean, True = real pool card.

        Returns:
            ``(logits (B, max_N), aux_pred (B,))``. Logits at padding positions
            are not meaningful and MUST be masked by the caller.
        """
        x = self.input_projection(pool_cards)
        # nn.MultiheadAttention's key_padding_mask: True = ignore.
        attn_padding_mask = ~pool_mask
        for sab in self.sab_layers:
            x = sab(x, attn_padding_mask=attn_padding_mask)

        logits = self.per_card_head(x).squeeze(-1)

        mask_f = pool_mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1.0)
        aux_pred = self.aux_head(summed / counts).squeeze(-1)
        return logits, aux_pred


def walk_pick_indices(logits: torch.Tensor, is_land_flags) -> list[int]:
    """Deterministic pick-decomposition walk (spec § 1.1), returning indices.

    Sorts pool cards by logit (descending) and walks the order: each spell is
    taken until the 23-spell quota fills, and each nonbasic land encountered
    before the quota fills is also taken. The walk halts as soon as the spell
    quota is met.

    Args:
        logits: 1D logit tensor of length ``N`` (one pool).
        is_land_flags: indexable of ``N`` booleans, True = nonbasic land.

    Returns:
        Chosen pool indices: spells first (ranked order), then any nonbasic
        lands taken along the way. No index appears twice.
    """
    order = torch.argsort(logits, descending=True).tolist()
    chosen_spells: list[int] = []
    chosen_lands: list[int] = []
    for idx in order:
        if len(chosen_spells) >= NONLAND_DECK_SIZE:
            break
        if is_land_flags[idx]:
            chosen_lands.append(idx)
        else:
            chosen_spells.append(idx)
    return chosen_spells + chosen_lands


def decompose_picks(
    logits: torch.Tensor,
    pool_embeddings,
    pool_names: list[str],
) -> list[str]:
    """Deterministic pick-decomposition walk (spec § 1.1), returning card names.

    Thin wrapper over ``walk_pick_indices`` that derives the land/spell
    partition from each embedding's ``is_land_embedding`` flag and maps the
    chosen indices back to names.
    """
    is_land_flags = [
        is_land_embedding(pool_embeddings[i]) for i in range(len(pool_names))
    ]
    chosen = walk_pick_indices(logits, is_land_flags)
    return [pool_names[i] for i in chosen]
