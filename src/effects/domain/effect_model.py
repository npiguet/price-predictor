"""The effect head: a game state plus an ability, to what that ability does.

A transformer trunk over the token surface, with three shipped heads: one
per-entity head shared over every ``[CARD]`` and ``[PLAYER]`` output (a gate,
then fields where the gate fires), a created-objects head at ``[GLOBAL]``, and
a verdict head at ``[ACT]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from effects.domain.ability_encoder import ENCODER_D_MODEL
from effects.domain.effect_head_input import ACT_SLOT, SlotKind
from effects.domain.effect_head_layout import (
    CREATED_OBJECTS_WIDTH,
    PER_ENTITY_WIDTH,
    VALUE_WIDTH,
    VERDICT_WIDTH,
)

# Hardcoded architecture (contracts/cli.md: "Hardcoded, not flags").
TRUNK_D_MODEL = 256
TRUNK_N_LAYERS = 6
TRUNK_N_HEADS = 4
FF_MULTIPLIER = 4
DROPOUT = 0.1
MAX_POSITION = 512


#: An ordinary ability row, and a charm mode's ``option`` row (FR-063a).
OPTION_KINDS = 2


@dataclass(frozen=True, slots=True)
class EffectModelConfig:
    """Architecture of the effect head."""

    global_features: int
    act_features: int
    player_features: int
    card_features: int
    e_dim: int = 64
    vocab_size: int = 0
    n_api_types: int = 0
    n_param_keys: int = 0
    #: Sizes the MLM head, which reads the encoder's token outputs (FR-060b).
    encoder_d_model: int = ENCODER_D_MODEL
    d_model: int = TRUNK_D_MODEL
    n_layers: int = TRUNK_N_LAYERS
    n_heads: int = TRUNK_N_HEADS
    ff_dim: int = TRUNK_D_MODEL * FF_MULTIPLIER
    dropout: float = DROPOUT
    max_position: int = MAX_POSITION

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )


def _mlp(width_in: int, hidden: int, width_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width_in, hidden), nn.GELU(), nn.Linear(hidden, width_out),
    )


class EffectModel(nn.Module):
    """Trunk plus the three shipped heads and the training-only auxiliaries."""

    def __init__(self, config: EffectModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.global_proj = nn.Linear(config.global_features, d)
        self.act_proj = nn.Linear(config.act_features, d)
        self.player_proj = nn.Linear(config.player_features, d)
        self.card_proj = nn.Linear(config.card_features, d)
        self.e_proj = nn.Linear(config.e_dim, d)
        self.slot_kind_embedding = nn.Embedding(len(SlotKind), d)
        # Zero rows: a checkpoint saved before option rows existed loads
        # without this key and must read every row as it trained (FR-063a).
        self.option_kind_embedding = nn.Embedding(OPTION_KINDS, d)
        nn.init.zeros_(self.option_kind_embedding.weight)
        self.position_embedding = nn.Embedding(config.max_position, d)
        self.dropout = nn.Dropout(config.dropout)

        self.trunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d, nhead=config.n_heads, dim_feedforward=config.ff_dim,
                dropout=config.dropout, batch_first=True, norm_first=True,
            ),
            num_layers=config.n_layers,
            # norm_first rules the nested-tensor path out; this silences the warning.
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d)

        # ── shipped heads ──
        self.per_entity_head = _mlp(d, d, PER_ENTITY_WIDTH)
        self.created_objects_head = _mlp(d, d, CREATED_OBJECTS_WIDTH)
        self.verdict_head = _mlp(d, d, VERDICT_WIDTH)

        # ── training-only auxiliaries (FR-063), filtered at save time by
        # ``aux_heads.TRAINING_ONLY_MODULES`` ──
        self.mlm_head = (
            nn.Linear(config.encoder_d_model, config.vocab_size)
            if config.vocab_size else None
        )
        self.api_type_head = (
            nn.Linear(config.e_dim, config.n_api_types)
            if config.n_api_types else None
        )
        self.param_key_head = (
            nn.Linear(config.e_dim, config.n_param_keys)
            if config.n_param_keys else None
        )
        # Amounts and costs the script states, read from e alone (FR-058).
        self.value_head = _mlp(config.e_dim, d, VALUE_WIDTH)

    def forward(
        self,
        slot_features: dict[SlotKind, torch.Tensor],
        slot_kinds: torch.Tensor,
        positions: torch.Tensor,
        e_vectors: torch.Tensor,
        attention_mask: torch.Tensor,
        option_kinds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the trunk. Returns ``(batch, slots, d_model)``.

        ``slot_features`` maps each kind to a ``(batch, slots, width)`` tensor
        zero at the slots of other kinds. Without ``option_kinds`` every row
        reads as an ordinary one.
        """
        hidden = (
            self.global_proj(slot_features[SlotKind.GLOBAL])
            + self.act_proj(slot_features[SlotKind.ACT])
            + self.player_proj(slot_features[SlotKind.PLAYER])
            + self.card_proj(slot_features[SlotKind.CARD])
            + self.e_proj(e_vectors)
            + self.slot_kind_embedding(slot_kinds)
            + self.position_embedding(positions)
        )
        if option_kinds is not None:
            hidden = hidden + self.option_kind_embedding(option_kinds)
        hidden = self.dropout(hidden)
        hidden = self.trunk(hidden, src_key_padding_mask=attention_mask == 0)
        return self.norm(hidden)

    def per_entity(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map the shared per-entity head over every slot; callers select the
        ``[CARD]`` and ``[PLAYER]`` positions."""
        return self.per_entity_head(hidden)

    def created_objects(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.created_objects_head(hidden[:, 0, :])

    def verdict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.verdict_head(hidden[:, ACT_SLOT, :])
