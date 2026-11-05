"""The effect head: a game state plus an ability, to what that ability does.

A transformer trunk over the token surface, with four shipped heads: one
per-entity head shared over every ``[CARD]`` and ``[PLAYER]`` output (a gate,
then fields where the gate fires), a created-objects head and a global head
(turn structure, day and night) at ``[GLOBAL]``, and a verdict head at
``[ACT]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from effects.domain.ability_encoder import DROPOUT, ENCODER_D_MODEL, FF_MULTIPLIER
from effects.domain.effect_head_input import (
    ACT_SLOT,
    COUNTER_ROWS,
    ID_PADDING,
    SUBTYPE_ROWS,
    SlotKind,
)
from effects.domain.effect_head_layout import (
    CREATED_DISPERSED,
    CREATED_OBJECTS_WIDTH,
    DISPERSED_FIELDS,
    GLOBAL_LAYOUT,
    GLOBAL_WIDTH,
    PER_ENTITY_WIDTH,
    VALUE_DISPERSED,
    VALUE_WIDTH,
    VERDICT_LAYOUT,
    VERDICT_WIDTH,
)
from effects.domain.field_kinds import INITIAL_LOG_DISPERSION, bounded_log_dispersion
from effects.domain.relation_attention import RelationEncoderLayer
from effects.domain.relation_types import RELATION_TYPES
from effects.domain.surface_collation import TrunkInputs, gather_entities

# Hardcoded architecture (contracts/cli.md: "Hardcoded, not flags").
TRUNK_D_MODEL = 256
TRUNK_N_LAYERS = 6
TRUNK_N_HEADS = 4
MAX_POSITION = 512


#: An ordinary ability row, and a charm mode's ``option`` row (FR-063a).
OPTION_KINDS = 2
#: The standard deviation the named-counter rows start at: each is scaled by
#: a count up to the magnitude cap, so it starts small.
COUNTER_INIT_STD = 0.02


@dataclass(frozen=True, slots=True)
class EffectModelConfig:
    """Architecture of the effect head."""

    global_features: int
    act_features: int
    player_features: int
    card_features: int
    #: The ``[ABILITY]`` slot's line flags; 0 builds no projection for them.
    ability_features: int = 0
    e_dim: int = 256
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
    #: Rows of each layer's relation tables: one per relation type (FR-047).
    relation_types: int = len(RELATION_TYPES)
    #: ``--relation-values``: off keeps the value tables at zero, untrained.
    relation_values: bool = True

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


class LogDispersion(nn.Module):
    """One head's learned log-dispersions, one per field a negative binomial
    scores (FR-063), in the head's own order of those fields. A module rather
    than a bare parameter, so its state key carries a prefix the save filter
    can name."""

    def __init__(self, fields: int) -> None:
        super().__init__()
        self.log_dispersion = nn.Parameter(torch.full((fields,), INITIAL_LOG_DISPERSION))

    def forward(self) -> torch.Tensor:
        """The log-dispersions as every loss reads them, bounded."""
        return bounded_log_dispersion(self.log_dispersion)


class RelationTrunk(nn.Module):
    """The gen-3 trunk: ``RelationEncoderLayer``s in sequence (root spec § 10.1).

    Kept under ``trunk.layers.N``, the names ``nn.TransformerEncoder`` gave the
    stock layers, so the parameter names a run reports stay as they were.
    """

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float,
                 n_layers: int, *, relation_types: int, relation_values: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            RelationEncoderLayer(d_model, n_heads, ff_dim, dropout,
                                 relation_types=relation_types, relation_values=relation_values)
            for _ in range(n_layers))

    def forward(self, hidden: torch.Tensor, src_key_padding_mask: torch.Tensor,
                relations: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden, src_key_padding_mask=src_key_padding_mask,
                           relations=relations)
        return hidden


class EffectModel(nn.Module):
    """Trunk plus the four shipped heads and the training-only auxiliaries."""

    def __init__(self, config: EffectModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.global_proj = nn.Linear(config.global_features, d)
        self.act_proj = nn.Linear(config.act_features, d)
        self.player_proj = nn.Linear(config.player_features, d)
        self.card_proj = nn.Linear(config.card_features, d)
        self.ability_proj = (
            nn.Linear(config.ability_features, d) if config.ability_features else None
        )
        self.e_proj = nn.Linear(config.e_dim, d)
        # One row per subtype, summed into a card slot; a chosen creature type
        # reads the same rows through its own map (root spec § 9.4).
        self.subtype_embedding = nn.Embedding(SUBTYPE_ROWS, d, padding_idx=ID_PADDING)
        self.chosen_type_proj = nn.Linear(d, d, bias=False)
        # One scalar input per named counter, kept sparse: a row per counter
        # type, its first half scaled by the count and its second by log1p of
        # it, which is a linear map of the dense raw-plus-log1p block.
        self.counter_embedding = nn.Embedding(COUNTER_ROWS, 2 * d, padding_idx=ID_PADDING)
        nn.init.normal_(self.counter_embedding.weight, std=COUNTER_INIT_STD)
        with torch.no_grad():
            self.counter_embedding.weight[ID_PADDING].zero_()
        self.slot_kind_embedding = nn.Embedding(len(SlotKind), d)
        # Zero rows: a checkpoint saved before option rows existed loads
        # without this key and must read every row as it trained (FR-063a).
        self.option_kind_embedding = nn.Embedding(OPTION_KINDS, d)
        nn.init.zeros_(self.option_kind_embedding.weight)
        self.position_embedding = nn.Embedding(config.max_position, d)
        self.dropout = nn.Dropout(config.dropout)

        self.trunk = RelationTrunk(
            d, config.n_heads, config.ff_dim, config.dropout, config.n_layers,
            relation_types=config.relation_types, relation_values=config.relation_values,
        )
        self.norm = nn.LayerNorm(d)

        # ── shipped heads ──
        self.per_entity_head = _mlp(d, d, PER_ENTITY_WIDTH)
        self.created_objects_head = _mlp(d, d, CREATED_OBJECTS_WIDTH)
        self.verdict_head = _mlp(d, d, VERDICT_WIDTH)
        self.global_head = _mlp(d, d, GLOBAL_WIDTH)
        # Each head's negative-binomial dispersions, in the ``head`` group.
        self.per_entity_dispersion = LogDispersion(len(DISPERSED_FIELDS))
        self.created_objects_dispersion = LogDispersion(len(CREATED_DISPERSED))
        self.verdict_dispersion = LogDispersion(len(VERDICT_LAYOUT.dispersed))
        self.global_dispersion = LogDispersion(len(GLOBAL_LAYOUT.dispersed))

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
        self.value_dispersion = LogDispersion(len(VALUE_DISPERSED))

    def forward(self, inputs: TrunkInputs) -> torch.Tensor:
        """Run the trunk. Returns ``(batch, slots, d_model)``.

        ``inputs.e_vectors`` is read as it stands: resolve ``e_rows`` first
        (:meth:`TrunkInputs.with_e`).
        """
        features = inputs.slot_features
        hidden = (
            self.global_proj(features[SlotKind.GLOBAL])
            + self.act_proj(features[SlotKind.ACT])
            + self.player_proj(features[SlotKind.PLAYER])
            + self.card_proj(features[SlotKind.CARD])
            + self.e_proj(inputs.e_vectors)
            + self.slot_kind_embedding(inputs.slot_kinds)
            + self.position_embedding(inputs.positions)
        )
        if inputs.option_kinds is not None:
            hidden = hidden + self.option_kind_embedding(inputs.option_kinds)
        if self.ability_proj is not None and SlotKind.ABILITY in features:
            hidden = hidden + self.ability_proj(features[SlotKind.ABILITY])
        hidden = hidden + self._card_ids(inputs)
        hidden = self.dropout(hidden)
        hidden = self.trunk(hidden, src_key_padding_mask=inputs.attention_mask == 0,
                            relations=inputs.relations)
        return self.norm(hidden)

    def _card_ids(self, inputs: TrunkInputs) -> torch.Tensor | float:
        """What a card slot's id lists add: its subtypes' rows, its chosen
        types' rows through their map, and each named counter's row scaled by
        the count (raw and log1p). Padding rows are zero."""
        added: torch.Tensor | float = 0.0
        if inputs.subtype_ids is not None:
            added = added + self.subtype_embedding(inputs.subtype_ids).sum(-2)
        if inputs.chosen_type_ids is not None:
            added = added + self.chosen_type_proj(
                self.subtype_embedding(inputs.chosen_type_ids).sum(-2))
        if inputs.counter_ids is not None:
            raw, logged = self.counter_embedding(inputs.counter_ids).chunk(2, dim=-1)
            counts = inputs.counter_counts.unsqueeze(-1)
            scaled = raw * counts + logged * torch.sign(counts) * torch.log1p(counts.abs())
            added = added + scaled.sum(-2)
        return added

    def per_entity(self, hidden: torch.Tensor) -> torch.Tensor:
        """The shared per-entity head over every trunk output it is handed."""
        return self.per_entity_head(hidden)

    def per_entity_at(self, hidden: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """``(batch, entities, PER_ENTITY_WIDTH)``: the per-entity head at the
        trunk columns ``index`` names. The head reads each position alone, so
        gathering first gives the same outputs and never runs it over an
        ability slot."""
        return self.per_entity_head(gather_entities(hidden, index))

    def created_objects(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.created_objects_head(hidden[:, 0, :])

    def verdict(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.verdict_head(hidden[:, ACT_SLOT, :])

    def global_outputs(self, hidden: torch.Tensor) -> torch.Tensor:
        """The global head at ``[GLOBAL]``: what belongs to no card or player."""
        return self.global_head(hidden[:, 0, :])
