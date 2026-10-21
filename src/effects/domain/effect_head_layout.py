"""The effect head's output layout: every field, slot and bit, and where it sits.

Field losses follow what the field *is* (FR-080). A count is Poisson, because
damage and cards drawn are counts and squared error on a count punishes the
wrong tail. A signed delta decomposes into a direction and a magnitude, because
"gains life" and "loses life" are a different question from "how much". A closed
vocabulary is categorical. A gate is binary.

Torch-free: the targets, the losses, the floor and the model all read the
layout, and none of them owns it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    OVERLAY_KEYWORDS,
)
from effects.domain.sampling import (
    CLASS_COMBAT,
    CLASS_PLAYABILITY_DECISION,
    CLASS_PLAYABILITY_LEGALITY,
    CLASS_RESOLUTION_COST,
    CLASS_RESOLUTION_EFFECT,
    CLASS_REWRITE,
)
from effects.domain.state_snapshot import COLORS
from effects.domain.value_targets import VALUE_TARGETS, ValueKind

#: Zone outcomes, a closed vocabulary (FR-077).
ZONE_OUTCOMES: tuple[str, ...] = (
    "stayed", "died", "exiled", "to_hand", "library_top", "library_bottom",
    "transformed", "face_changed", "phased_out", "blinked", "countered",
)
#: How long a delta lasts. Closed, and read alongside every delta that has one.
DURATIONS: tuple[str, ...] = ("instant", "end_of_turn", "permanent")
#: Library manipulations a player can undergo, as a multi-hot.
LIBRARY_EVENTS: tuple[str, ...] = ("scry", "surveil", "tutor", "reveal", "reorder")


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


class CreatedSlot:
    """Offsets within one created-objects slot.

    Two binary flags, then three magnitudes, then one flag per card type and
    per overlay keyword for a characteristics-only group.
    """

    PRESENT = 0
    SCRIPTED = 1
    COUNT = 2
    POWER = 3
    TOUGHNESS = 4
    TYPES = 5
    KEYWORDS = TYPES + len(CARD_TYPES)


CREATED_SLOT_WIDTH = CreatedSlot.KEYWORDS + len(OVERLAY_KEYWORDS)
CREATED_OBJECTS_WIDTH = CREATED_OBJECT_SLOTS * CREATED_SLOT_WIDTH + 1

#: Verdict head (FR-079): three verdict bits, cost paid per colour, trigger fired.
VERDICT_BITS: tuple[str, ...] = ("can_play", "affordable", "has_legal_target")
VERDICT_WIDTH = len(VERDICT_BITS) + len(COLORS) + 1

#: Value head (FR-058): one output per count or binary target, four per signed
#: target (the direction-and-magnitude split every signed field uses).
_VALUE_WIDTHS: dict[ValueKind, int] = {
    ValueKind.COUNT: 1, ValueKind.BINARY: 1, ValueKind.SIGNED: 4,
}
#: ``target -> (start, end)`` into the value head's output.
VALUE_SLICES: dict[str, tuple[int, int]] = {}
_value_offset = 0
for _name, _kind in VALUE_TARGETS:
    VALUE_SLICES[_name] = (_value_offset, _value_offset + _VALUE_WIDTHS[_kind])
    _value_offset += _VALUE_WIDTHS[_kind]

#: Steps of the at-least targets (FR-058b): beside each stated amount, the head
#: predicts whether its magnitude is at least k for k = 1 … 40, one yes/no each —
#: the magnitude cap, and the board's thermometer length, so both sides of a
#: comparison count on one scale.
#: A single layer reading every such bit from ``e`` pushes ``e`` to carry the
#: amount as a thermometer — the form the trunk's attention compares against a
#: toughness or a life total in one dot product (the board side carries the
#: same code, ``effect_head_input.thermometer``). Cumulative-link ordinal
#: regression, kept beside the count loss rather than replacing it.
AT_LEAST_STEPS = 40
#: ``amount target -> (start, end)`` of its at-least block, after every
#: :data:`VALUE_SLICES` block so the earlier layout of the head is a prefix.
AT_LEAST_SLICES: dict[str, tuple[int, int]] = {}
for _name, _kind in VALUE_TARGETS:
    if _kind is not ValueKind.BINARY:
        AT_LEAST_SLICES[_name] = (_value_offset, _value_offset + AT_LEAST_STEPS)
        _value_offset += AT_LEAST_STEPS
VALUE_WIDTH = _value_offset



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

