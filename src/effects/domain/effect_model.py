"""The effect head: a game state plus an ability, to what that ability does.

A transformer trunk over the token surface
:mod:`effects.domain.effect_head_input` builds, with three output heads:

- a **per-entity head** mapped over every ``[CARD]`` *and* ``[PLAYER]`` output —
  one shared head, so "what happens to this thing" is learned once rather than
  once per board position;
- a **created-objects head** at ``[GLOBAL]``, because a token a spell makes
  belongs to no entity that was on the board;
- a **verdict head** at ``[ACT]``, carrying the rules-level judgments about the
  acting line itself.

The per-entity head is a **gate then conditional fields**: predict whether this
entity is affected at all, and only where it is, predict how. Most entities on
most boards are unaffected by most abilities, so a head that predicted every
field for every entity would spend nearly all its loss learning to output zero.

Field losses follow what the field *is* (FR-080). A count is Poisson, because
damage and cards drawn are counts and squared error on a count punishes the
wrong tail. A signed delta decomposes into a direction and a magnitude, because
"gains life" and "loses life" are a different question from "how much". A closed
vocabulary is categorical. A gate is binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as functional

from effects.domain.ability_encoder import ENCODER_D_MODEL
from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    OVERLAY_KEYWORDS,
    SlotKind,
)
from effects.domain.state_snapshot import COLORS

# Hardcoded architecture (contracts/cli.md: "Hardcoded, not flags").
TRUNK_D_MODEL = 256
TRUNK_N_LAYERS = 6
TRUNK_N_HEADS = 4
FF_MULTIPLIER = 4
DROPOUT = 0.1
MAX_POSITION = 512

#: Zone outcomes, a closed vocabulary (FR-077).
ZONE_OUTCOMES: tuple[str, ...] = (
    "stayed", "died", "exiled", "to_hand", "library_top", "library_bottom",
    "transformed", "face_changed", "phased_out", "blinked", "countered",
)
#: How long a delta lasts. Closed, and read alongside every delta that has one.
DURATIONS: tuple[str, ...] = ("instant", "end_of_turn", "permanent")
#: Library manipulations a player can undergo, as a multi-hot.
LIBRARY_EVENTS: tuple[str, ...] = ("scry", "surveil", "tutor", "reveal", "reorder")

#: The eight sampling classes, which are also the granularity FR-085 speaks in:
#: a field whose classes are all absent from the corpus contributes no loss.
CLASS_RESOLUTION_COST = "resolution-cost"
CLASS_RESOLUTION_EFFECT = "resolution-effect"
CLASS_REWRITE = "rewrite"
CLASS_CONTINUOUS = "continuous"
CLASS_TRIGGER = "trigger"
CLASS_PLAYABILITY_DECISION = "playability-decision"
CLASS_PLAYABILITY_LEGALITY = "playability-legality"
CLASS_COMBAT = "combat"

SAMPLING_CLASSES: tuple[str, ...] = (
    CLASS_RESOLUTION_EFFECT, CLASS_COMBAT, CLASS_CONTINUOUS,
    CLASS_PLAYABILITY_DECISION, CLASS_RESOLUTION_COST, CLASS_TRIGGER,
    CLASS_REWRITE, CLASS_PLAYABILITY_LEGALITY,
)

#: Classes whose records describe an outcome for an arbitrary entity. Used
#: below to say "every effect-shaped record supervises this field".
_EFFECTFUL = frozenset({
    CLASS_RESOLUTION_EFFECT, CLASS_REWRITE, CLASS_RESOLUTION_COST, CLASS_COMBAT,
})


class FieldType(StrEnum):
    BINARY = "binary"
    MULTI_BINARY = "multi_binary"
    COUNT = "count"
    SIGNED_DELTA = "signed_delta"
    CATEGORICAL = "categorical"


class FieldGroup(StrEnum):
    """When a field starts training.

    The sparse group is the set FR-082 names: fields that fire on a small
    fraction of records, and which a head learns to suppress entirely if they
    compete with the dense ones from step zero.
    """

    DENSE = "dense"
    SPARSE = "sparse"


class FieldScope(StrEnum):
    """Which slot kind a field is predicted at."""

    PERMANENT = "permanent"
    PLAYER = "player"
    LEGALITY = "legality"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One output field of the per-entity head."""

    name: str
    type: FieldType
    scope: FieldScope
    #: Categorical classes / multi-binary members. 1 for the scalar types.
    arity: int = 1
    group: FieldGroup = FieldGroup.DENSE
    #: Sampling classes whose records carry a target for this field.
    supervised_by: frozenset[str] = field(default_factory=lambda: _EFFECTFUL)

    @property
    def width(self) -> int:
        """How many head outputs this field occupies."""
        match self.type:
            case FieldType.BINARY | FieldType.COUNT:
                return 1
            case FieldType.MULTI_BINARY | FieldType.CATEGORICAL:
                return self.arity
            case FieldType.SIGNED_DELTA:
                # Three direction logits (negative, zero, positive) plus one
                # nonnegative magnitude under the count loss.
                return 4
        raise ValueError(f"unhandled field type {self.type}")


def _counter_fields() -> tuple[FieldSpec, ...]:
    """One signed delta per tracked counter type, plus a catch-all."""
    return tuple(
        FieldSpec(
            name=f"counters_delta_{name.lower()}",
            type=FieldType.SIGNED_DELTA,
            scope=FieldScope.PERMANENT,
            group=FieldGroup.SPARSE,
        )
        for name in (*COUNTER_TYPES, "OTHER")
    )


def _mana_fields() -> tuple[FieldSpec, ...]:
    return tuple(
        FieldSpec(
            name=f"mana_delta_{color.lower()}",
            type=FieldType.SIGNED_DELTA,
            scope=FieldScope.PLAYER,
        )
        for color in COLORS
    )


#: The per-entity head's full field list. Order is the head's output layout, so
#: appending is safe and reordering is not.
PER_ENTITY_FIELDS: tuple[FieldSpec, ...] = (
    # ── permanents and stack entities ──
    FieldSpec("zone_outcome", FieldType.CATEGORICAL, FieldScope.PERMANENT,
              arity=len(ZONE_OUTCOMES)),
    FieldSpec("tap_state", FieldType.BINARY, FieldScope.PERMANENT),
    FieldSpec("damage_taken", FieldType.COUNT, FieldScope.PERMANENT),
    FieldSpec("power_delta", FieldType.SIGNED_DELTA, FieldScope.PERMANENT),
    FieldSpec("toughness_delta", FieldType.SIGNED_DELTA, FieldScope.PERMANENT),
    FieldSpec("pt_duration", FieldType.CATEGORICAL, FieldScope.PERMANENT,
              arity=len(DURATIONS), group=FieldGroup.SPARSE),
    *_counter_fields(),
    FieldSpec("types_gained", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(CARD_TYPES), group=FieldGroup.SPARSE),
    FieldSpec("types_lost", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(CARD_TYPES), group=FieldGroup.SPARSE),
    FieldSpec("colors_gained", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(COLORS), group=FieldGroup.SPARSE),
    FieldSpec("colors_lost", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(COLORS), group=FieldGroup.SPARSE),
    FieldSpec("type_color_duration", FieldType.CATEGORICAL, FieldScope.PERMANENT,
              arity=len(DURATIONS), group=FieldGroup.SPARSE),
    FieldSpec("keywords_gained", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(OVERLAY_KEYWORDS), group=FieldGroup.SPARSE),
    FieldSpec("keywords_lost", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(OVERLAY_KEYWORDS), group=FieldGroup.SPARSE),
    FieldSpec("control_change", FieldType.BINARY, FieldScope.PERMANENT,
              group=FieldGroup.SPARSE),
    FieldSpec("attached_to_change", FieldType.BINARY, FieldScope.PERMANENT,
              group=FieldGroup.SPARSE),
    # ── players ──
    FieldSpec("life_delta", FieldType.SIGNED_DELTA, FieldScope.PLAYER),
    FieldSpec("cards_drawn", FieldType.COUNT, FieldScope.PLAYER),
    FieldSpec("cards_discarded", FieldType.COUNT, FieldScope.PLAYER),
    FieldSpec("cards_milled", FieldType.COUNT, FieldScope.PLAYER),
    FieldSpec("library_events", FieldType.MULTI_BINARY, FieldScope.PLAYER,
              arity=len(LIBRARY_EVENTS)),
    # Poison is not in FR-077's player group, but the snapshot tracks it and
    # gate 2's infect row has nowhere else to land: infect's whole player-side
    # effect is poison, so without this field that row could never fire.
    FieldSpec("poison_delta", FieldType.SIGNED_DELTA, FieldScope.PLAYER),
    *_mana_fields(),
    # ── legality bits ──
    FieldSpec("target_legal", FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({CLASS_PLAYABILITY_DECISION})),
    FieldSpec("attacker_legal", FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({CLASS_PLAYABILITY_LEGALITY})),
    FieldSpec("blocker_legal", FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({CLASS_PLAYABILITY_LEGALITY})),
    FieldSpec("min_blockers", FieldType.COUNT, FieldScope.LEGALITY,
              supervised_by=frozenset({CLASS_PLAYABILITY_LEGALITY})),
)

#: ``{field name: (start, end)}`` into the per-entity head's output vector.
FIELD_SLICES: dict[str, tuple[int, int]] = {}
_offset = 1  # index 0 is the affected/unaffected gate
for _spec in PER_ENTITY_FIELDS:
    FIELD_SLICES[_spec.name] = (_offset, _offset + _spec.width)
    _offset += _spec.width
PER_ENTITY_WIDTH = _offset
GATE_INDEX = 0

FIELDS_BY_NAME: dict[str, FieldSpec] = {s.name: s for s in PER_ENTITY_FIELDS}

#: Created-objects head (FR-078): K group slots plus one overflow flag.
CREATED_OBJECT_SLOTS = 4
#: Per slot: present, scripted flag, count, power, toughness, then type and
#: keyword flags for a characteristics-only group.
CREATED_SLOT_WIDTH = 3 + 2 + len(CARD_TYPES) + len(OVERLAY_KEYWORDS)
CREATED_OBJECTS_WIDTH = CREATED_OBJECT_SLOTS * CREATED_SLOT_WIDTH + 1

#: Verdict head (FR-079): three verdict bits, cost paid per colour, trigger fired.
VERDICT_BITS: tuple[str, ...] = ("can_play", "affordable", "has_legal_target")
VERDICT_WIDTH = len(VERDICT_BITS) + len(COLORS) + 1


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


class EffectModel(nn.Module):
    """Trunk plus the three shipped heads and the training-only auxiliaries."""

    def __init__(self, config: EffectModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        # One projection per slot kind: the kinds carry different features and
        # a shared projection would have to pad them to a common width.
        self.global_proj = nn.Linear(config.global_features, d)
        self.act_proj = nn.Linear(config.act_features, d)
        self.player_proj = nn.Linear(config.player_features, d)
        self.card_proj = nn.Linear(config.card_features, d)
        self.e_proj = nn.Linear(config.e_dim, d)
        self.slot_kind_embedding = nn.Embedding(len(SlotKind), d)
        self.position_embedding = nn.Embedding(config.max_position, d)
        self.dropout = nn.Dropout(config.dropout)

        self.trunk = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d, nhead=config.n_heads, dim_feedforward=config.ff_dim,
                dropout=config.dropout, batch_first=True, norm_first=True,
            ),
            num_layers=config.n_layers,
            # norm_first makes torch's nested-tensor fast path inapplicable; say
            # so rather than letting it warn once per construction.
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d)

        # ── shipped heads ──
        self.per_entity_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, PER_ENTITY_WIDTH),
        )
        self.created_objects_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, CREATED_OBJECTS_WIDTH),
        )
        self.verdict_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, VERDICT_WIDTH),
        )

        # ── training-only auxiliaries (FR-063), filtered at save time ──
        self.mlm_head = (
            nn.Linear(ENCODER_D_MODEL, config.vocab_size)
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
        # Stage four's paired-encoding projection; see `pairing_loss`.
        self.pairing_proj = nn.Linear(config.e_dim, config.e_dim)

    #: Head names filtered out of the saved artifact (FR-076).
    TRAINING_ONLY_HEADS: tuple[str, ...] = (
        "mlm_head", "api_type_head", "param_key_head", "pairing_proj",
    )

    def forward(
        self,
        slot_features: dict[SlotKind, torch.Tensor],
        slot_kinds: torch.Tensor,
        positions: torch.Tensor,
        e_vectors: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the trunk. Returns ``(batch, slots, d_model)``.

        ``slot_features`` maps each kind to a ``(batch, slots, width)`` tensor
        already zero-padded at the slots of other kinds, so one add assembles
        the surface without a per-slot Python loop.
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
        hidden = self.dropout(hidden)
        hidden = self.trunk(hidden, src_key_padding_mask=attention_mask == 0)
        return self.norm(hidden)

    def per_entity(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map the shared per-entity head over every slot.

        Callers select the ``[CARD]`` and ``[PLAYER]`` positions; mapping over
        all of them and selecting afterwards keeps the head one batched matmul.
        """
        return self.per_entity_head(hidden)

    def created_objects(self, hidden: torch.Tensor) -> torch.Tensor:
        """Read the created-objects head at ``[GLOBAL]`` (slot 0)."""
        return self.created_objects_head(hidden[:, 0, :])

    def verdict(self, hidden: torch.Tensor, act_index: int = 1) -> torch.Tensor:
        """Read the verdict head at ``[ACT]``."""
        return self.verdict_head(hidden[:, act_index, :])


# ── losses (FR-080, FR-081, FR-082, FR-085) ─────────────────────────────


def per_entity_fields_of_type(*types: FieldType) -> tuple[FieldSpec, ...]:
    """Every per-entity field of one of ``types``.

    Gate 1's deviance is read over the count-valued ones, and a caller that
    enumerated them by name would drift the moment a field was added.
    """
    wanted = set(types)
    return tuple(spec for spec in PER_ENTITY_FIELDS if spec.type in wanted)


def active_fields(
    *, present_classes: frozenset[str], step: int, curriculum_step: int,
) -> tuple[FieldSpec, ...]:
    """Fields that contribute loss right now.

    Two filters, for two different reasons. A field none of the corpus's record
    kinds supervises is dropped because nothing could teach it — that is what
    makes a stage-one corpus with three of the eight classes trainable on the
    same heads as a full one (FR-085). A sparse field is dropped before
    ``curriculum_step`` because it fires rarely, and a head that meets it
    alongside the dense fields from step zero learns to output zero for it and
    never recovers (FR-082).
    """
    return tuple(
        spec for spec in PER_ENTITY_FIELDS
        if spec.supervised_by & present_classes
        and (spec.group is FieldGroup.DENSE or step >= curriculum_step)
    )


def _poisson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Poisson NLL on a log-rate prediction, summed."""
    return functional.poisson_nll_loss(
        prediction, target, log_input=True, full=False, reduction="sum",
    )


def _signed_delta_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Direction (categorical) plus magnitude (count), summed.

    Splitting them is what makes "gains 2 life" and "loses 2 life" different
    predictions rather than a regression that can hedge between them.
    """
    direction_logits, log_magnitude = prediction[..., :3], prediction[..., 3]
    direction = torch.sign(target).long() + 1  # -1,0,1 -> 0,1,2
    loss = functional.cross_entropy(
        direction_logits.reshape(-1, 3), direction.reshape(-1), reduction="sum",
    )
    return loss + _poisson(log_magnitude, target.abs())


def field_loss(
    spec: FieldSpec, prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """The loss for one field, chosen by what the field is (FR-080)."""
    match spec.type:
        case FieldType.BINARY:
            return functional.binary_cross_entropy_with_logits(
                prediction.squeeze(-1), target.float(), reduction="sum",
            )
        case FieldType.MULTI_BINARY:
            return functional.binary_cross_entropy_with_logits(
                prediction, target.float(), reduction="sum",
            )
        case FieldType.COUNT:
            return _poisson(prediction.squeeze(-1), target.float())
        case FieldType.SIGNED_DELTA:
            return _signed_delta_loss(prediction, target.float())
        case FieldType.CATEGORICAL:
            return functional.cross_entropy(
                prediction.reshape(-1, spec.arity), target.reshape(-1).long(),
                reduction="sum",
            )
    raise ValueError(f"unhandled field type {spec.type}")


def per_entity_loss(
    outputs: torch.Tensor,
    gate_target: torch.Tensor,
    field_targets: dict[str, torch.Tensor],
    entity_mask: torch.Tensor,
    *,
    fields: tuple[FieldSpec, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    """The per-entity head's loss, normalized per record over entity count.

    Args:
        outputs: ``(batch, entities, PER_ENTITY_WIDTH)``.
        gate_target: ``(batch, entities)``, 1 where the entity was affected.
        field_targets: per-field ``(batch, entities, …)`` targets.
        entity_mask: ``(batch, entities)``, 1 for a real entity.
        fields: the fields active at this step, from :func:`active_fields`.

    The gate trains on every entity — players included, since the head is mapped
    over ``[PLAYER]`` outputs too. Conditional fields train only where the
    gate's *target* fires, not where its prediction does: supervising on the
    prediction would let a head that gates everything off escape the field loss
    entirely.

    Normalizing per record rather than per entity keeps a twelve-permanent board
    from outweighing a two-permanent one; each record is one observation of an
    ability, whatever the board size.
    """
    parts: dict[str, float] = {}
    mask = entity_mask.bool()
    gate_logits = outputs[..., GATE_INDEX]
    gate = functional.binary_cross_entropy_with_logits(
        gate_logits[mask], gate_target[mask].float(), reduction="sum",
    )
    parts["gate"] = float(gate.detach())
    total = gate

    affected = mask & gate_target.bool()
    for spec in fields:
        target = field_targets.get(spec.name)
        if target is None or not affected.any():
            continue
        start, end = FIELD_SLICES[spec.name]
        loss = field_loss(spec, outputs[..., start:end][affected], target[affected])
        parts[spec.name] = float(loss.detach())
        total = total + loss

    records = max(int(outputs.shape[0]), 1)
    return total / records, parts


def created_objects_loss(
    prediction: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Binary presence/flags plus count regression over the K slots.

    Slots are in canonical order (FR-078), so slot *i* means the same group on
    both sides and the head is not asked to solve an assignment problem.
    """
    total = prediction.new_zeros(())
    for slot in range(CREATED_OBJECT_SLOTS):
        base = slot * CREATED_SLOT_WIDTH
        total = total + functional.binary_cross_entropy_with_logits(
            prediction[:, base:base + 2], target[:, base:base + 2],
            reduction="sum",
        )
        total = total + _poisson(
            prediction[:, base + 2], target[:, base + 2],
        )
        rest = slice(base + 3, base + CREATED_SLOT_WIDTH)
        total = total + functional.binary_cross_entropy_with_logits(
            prediction[:, rest], target[:, rest], reduction="sum",
        )
    total = total + functional.binary_cross_entropy_with_logits(
        prediction[:, -1], target[:, -1], reduction="sum",
    )
    return total / max(int(prediction.shape[0]), 1)


def verdict_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Verdict bits and the trigger bit binary; cost paid Poisson per colour."""
    if not mask.any():
        return prediction.new_zeros(())
    prediction, target = prediction[mask], target[mask]
    bits = len(VERDICT_BITS)
    total = functional.binary_cross_entropy_with_logits(
        prediction[:, :bits], target[:, :bits], reduction="sum",
    )
    total = total + _poisson(
        prediction[:, bits:bits + len(COLORS)], target[:, bits:bits + len(COLORS)],
    )
    total = total + functional.binary_cross_entropy_with_logits(
        prediction[:, -1], target[:, -1], reduction="sum",
    )
    return total / max(int(prediction.shape[0]), 1)


def mlm_loss(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy at the masked positions only.

    Training-only: it teaches the encoder the language of card text, which the
    effect objective alone would only reach through whichever words happened to
    change an outcome.
    """
    if not mask.any():
        return logits.new_zeros(())
    return functional.cross_entropy(logits[mask], targets[mask])


def api_loss(
    type_logits: torch.Tensor | None,
    type_targets: torch.Tensor | None,
    key_logits: torch.Tensor | None,
    key_targets: torch.Tensor | None,
) -> torch.Tensor:
    """Script-API classification from ``e``: the API type and its parameter keys.

    This is not stage-four work. Through stage three the encoder reads prose
    only, and this head is what stands in for the script structure prose does
    not spell out — so the MVP needs it.
    """
    total = None
    if type_logits is not None and type_targets is not None:
        total = functional.cross_entropy(type_logits, type_targets)
    if key_logits is not None and key_targets is not None:
        keys = functional.binary_cross_entropy_with_logits(
            key_logits, key_targets.float(),
        )
        total = keys if total is None else total + keys
    if total is None:
        raise ValueError("api_loss needs at least one of its two targets")
    return total


def collate_surfaces(
    surfaces: list, *, e_dim: int, widths: dict[SlotKind, int],
) -> dict[str, torch.Tensor]:
    """Pad a batch of token surfaces into the trunk's keyword arguments.

    Each slot kind's features go into its own zero-padded tensor, so the model
    can project all four with four matmuls rather than gathering per slot. The
    batch pads to its own longest surface: board sizes vary a great deal and
    padding to a fixed width would spend most of the compute on empty slots.
    """
    batch = len(surfaces)
    width = max((len(s.slots) for s in surfaces), default=1)
    features = {
        kind: torch.zeros(batch, width, size) for kind, size in widths.items()
    }
    slot_kinds = torch.zeros(batch, width, dtype=torch.long)
    positions = torch.zeros(batch, width, dtype=torch.long)
    e_vectors = torch.zeros(batch, width, e_dim)
    attention = torch.zeros(batch, width, dtype=torch.long)

    for row, surface in enumerate(surfaces):
        for column, slot in enumerate(surface.slots):
            slot_kinds[row, column] = int(slot.kind)
            positions[row, column] = slot.position
            attention[row, column] = 1
            if slot.features and slot.kind in features:
                values = torch.tensor(slot.features, dtype=torch.float32)
                features[slot.kind][row, column, : values.shape[0]] = values
            if slot.e is not None:
                vector = torch.tensor(slot.e, dtype=torch.float32)
                e_vectors[row, column, : vector.shape[0]] = vector

    return {
        "slot_features": features,
        "slot_kinds": slot_kinds,
        "positions": positions,
        "e_vectors": e_vectors,
        "attention_mask": attention,
    }


def entity_target_tensors(
    surfaces: list, targets_per_surface: list[dict], fields: tuple[FieldSpec, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Gather per-entity outputs and their targets into aligned tensors.

    Returns ``(gate_target, field_targets, entity_mask, entity_index)``, where
    ``entity_index`` selects the ``[CARD]`` and ``[PLAYER]`` columns of the
    trunk output. A record that mentions an entity no slot represents is simply
    not gathered — the surface is the authority on what exists.
    """
    batch = len(surfaces)
    width = max((len(s.entity_slots) for s in surfaces), default=1)
    gate = torch.zeros(batch, width)
    mask = torch.zeros(batch, width)
    index = torch.zeros(batch, width, dtype=torch.long)
    collected: dict[str, torch.Tensor] = {}

    for spec in fields:
        shape = (batch, width) if spec.width <= 4 else (batch, width, spec.arity)
        if spec.type is FieldType.MULTI_BINARY:
            shape = (batch, width, spec.arity)
        collected[spec.name] = torch.zeros(*shape)

    for row, (surface, targets) in enumerate(zip(surfaces, targets_per_surface)):
        for column, slot_index in enumerate(surface.entity_slots):
            slot = surface.slots[slot_index]
            key = slot.entity_id or slot.player_id
            mask[row, column] = 1
            index[row, column] = slot_index
            entry = targets.get(key)
            if entry is None:
                continue
            gate[row, column] = 1.0 if entry.affected else 0.0
            for spec in fields:
                value = entry.fields.get(spec.name)
                if value is None:
                    continue
                if isinstance(value, list):
                    collected[spec.name][row, column, : len(value)] = torch.tensor(
                        value, dtype=torch.float32,
                    )
                else:
                    collected[spec.name][row, column] = float(value)
    return gate, collected, mask, index


def pairing_loss(
    script_e: torch.Tensor, prose_e: torch.Tensor,
) -> torch.Tensor:
    """Stage four: pull the prose encoding toward the script encoding.

    Asymmetric with a stop-gradient on the script side, so prose learns to agree
    with the mechanism rather than the two meeting somewhere in between. A
    synthetic variant has only the script surface and contributes nothing here.
    """
    return functional.mse_loss(prose_e, script_e.detach())
