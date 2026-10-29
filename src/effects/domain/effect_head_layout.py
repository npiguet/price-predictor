"""The effect head's output layout: every field, slot and bit, and where it sits.

Field losses follow what the field *is* (FR-080). A count is Poisson, because
damage and cards drawn are counts and squared error on a count punishes the
wrong tail. A signed delta decomposes into a direction and a magnitude, because
"gains life" and "loses life" are a different question from "how much". A closed
vocabulary is categorical. A gate is binary.

A field is scored on an affected entity only where the record says something
about it (FR-077, FR-081). Two things make a field *present*: a value the
record wrote — an event, or a payload's legality mark — or the field's
**silent** value, which is what "affected, and nothing said" means for a field
whose absence is itself an observation. A creature a burn spell damaged and did
not kill stayed where it was and gained no counters; that is known. Whether it
was tapped is not, and a tapped creature is not the opposite of one no event
mentioned. So a field takes its silent value only where the record's sampling
class supervises it and the entity is the kind the field describes — a player
is never scored on a zone, a permanent never on life — and the tap state, the
durations and the legality bits have no silent value at all.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import torch

from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    OVERLAY_KEYWORDS,
)
from effects.domain.sampling import SamplingClass
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


#: Classes whose records list every outcome event of what they describe, so an
#: affected entity no event of a kind names underwent none of that kind.
_EFFECTFUL = frozenset({
    SamplingClass.RESOLUTION_EFFECT, SamplingClass.REWRITE, SamplingClass.RESOLUTION_COST,
    SamplingClass.COMBAT,
})
#: The effectful classes plus ``continuous``, whose contribution names every
#: P/T, keyword, type and colour change its static makes to the entity.
_STATIC_CHANNELS = _EFFECTFUL | {SamplingClass.CONTINUOUS}


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
    """Which kind of entity a field describes, and so where its silent value applies."""

    PERMANENT = "permanent"
    PLAYER = "player"
    #: The legality bits, which only a payload's own marks ever set.
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
    supervised_by: frozenset[SamplingClass] = field(default_factory=lambda: _EFFECTFUL)
    #: The value an affected entity of ``scope`` takes when a record of a
    #: supervising class names no change to the field; every member of a
    #: multi-binary field. None for a field only a written value supervises.
    silent: float | None = None

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

    def supervised_value(
        self, written: Mapping[str, object], unknown: frozenset[str] | set[str],
        *, record_class: str | None, scope: FieldScope | None,
    ):
        """This field's target for one affected entity, or None where it is absent.

        ``written`` holds what the record set; ``unknown`` the fields it
        declares it cannot say, which no silent value overrides.
        """
        value = written.get(self.name)
        if value is not None:
            return value
        if (
            self.silent is None or self.name in unknown
            or scope is not self.scope or record_class not in self.supervised_by
        ):
            return None
        if self.type is FieldType.MULTI_BINARY:
            return [self.silent] * self.arity
        return self.silent


def _counter_fields() -> tuple[FieldSpec, ...]:
    """One signed delta per tracked counter type, plus a catch-all."""
    return tuple(
        FieldSpec(
            name=f"counters_delta_{name.lower()}",
            type=FieldType.SIGNED_DELTA,
            scope=FieldScope.PERMANENT,
            group=FieldGroup.SPARSE,
            silent=0.0,
        )
        for name in (*COUNTER_TYPES, "OTHER")
    )


def _mana_fields() -> tuple[FieldSpec, ...]:
    return tuple(
        FieldSpec(
            name=f"mana_delta_{color.lower()}",
            type=FieldType.SIGNED_DELTA,
            scope=FieldScope.PLAYER,
            silent=0.0,
        )
        for color in COLORS
    )


#: The per-entity head's full field list. Order is the head's output layout, so
#: appending is safe and reordering is not.
PER_ENTITY_FIELDS: tuple[FieldSpec, ...] = (
    # ── permanents and stack entities ──
    FieldSpec("zone_outcome", FieldType.CATEGORICAL, FieldScope.PERMANENT,
              arity=len(ZONE_OUTCOMES), silent=float(ZONE_OUTCOMES.index("stayed"))),
    # Tapped or untapped only where an event says so: a creature no event
    # mentions is neither.
    FieldSpec("tap_state", FieldType.BINARY, FieldScope.PERMANENT),
    FieldSpec("damage_taken", FieldType.COUNT, FieldScope.PERMANENT, silent=0.0),
    FieldSpec("power_delta", FieldType.SIGNED_DELTA, FieldScope.PERMANENT,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("toughness_delta", FieldType.SIGNED_DELTA, FieldScope.PERMANENT,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("pt_duration", FieldType.CATEGORICAL, FieldScope.PERMANENT,
              arity=len(DURATIONS), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS),
    *_counter_fields(),
    FieldSpec("types_gained", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(CARD_TYPES), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("types_lost", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(CARD_TYPES), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("colors_gained", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(COLORS), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("colors_lost", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(COLORS), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("type_color_duration", FieldType.CATEGORICAL, FieldScope.PERMANENT,
              arity=len(DURATIONS), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS),
    FieldSpec("keywords_gained", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(OVERLAY_KEYWORDS), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("keywords_lost", FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              arity=len(OVERLAY_KEYWORDS), group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec("control_change", FieldType.BINARY, FieldScope.PERMANENT,
              group=FieldGroup.SPARSE, silent=0.0),
    FieldSpec("attached_to_change", FieldType.BINARY, FieldScope.PERMANENT,
              group=FieldGroup.SPARSE, silent=0.0),
    # ── players ──
    FieldSpec("life_delta", FieldType.SIGNED_DELTA, FieldScope.PLAYER, silent=0.0),
    FieldSpec("cards_drawn", FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    FieldSpec("cards_discarded", FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    FieldSpec("cards_milled", FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    FieldSpec("library_events", FieldType.MULTI_BINARY, FieldScope.PLAYER,
              arity=len(LIBRARY_EVENTS), silent=0.0),
    # Poison is not in FR-077's player group, but the snapshot tracks it and
    # gate 2's infect row has nowhere else to land: infect's whole player-side
    # effect is poison, so without this field that row could never fire.
    FieldSpec("poison_delta", FieldType.SIGNED_DELTA, FieldScope.PLAYER, silent=0.0),
    *_mana_fields(),
    # ── legality bits: only a payload's own marks set them ──
    FieldSpec("target_legal", FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_DECISION})),
    FieldSpec("attacker_legal", FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_LEGALITY})),
    FieldSpec("blocker_legal", FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_LEGALITY})),
    FieldSpec("min_blockers", FieldType.COUNT, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_LEGALITY})),
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



@dataclass(frozen=True)
class EntityFieldTargets:
    """A batch's per-entity field targets, each beside where it is present.

    ``values[name]`` is ``(batch, entities)`` — ``(batch, entities, arity)`` for
    a multi-binary field — and ``present[name]`` the ``(batch, entities)`` bool
    mask of the entities whose record supervises the field. A field is scored
    only where the gate's target fires *and* it is present.

    ``present`` stays on the host whatever device the values move to: the loss
    picks its rows there, so no step waits for the device to say which.
    """

    values: Mapping[str, torch.Tensor]
    present: Mapping[str, torch.Tensor]

    @classmethod
    def everywhere(cls, values: Mapping[str, torch.Tensor]) -> EntityFieldTargets:
        """``values`` present at every entity."""
        return cls(
            dict(values),
            {
                name: torch.ones(value.shape[:2], dtype=torch.bool)
                for name, value in values.items()
            },
        )

    def __contains__(self, name: str) -> bool:
        return name in self.values

    def to(self, device, *, non_blocking: bool = False) -> EntityFieldTargets:
        """The values on ``device``; the presence masks stay where they are."""
        return EntityFieldTargets(
            {
                name: value.to(device, non_blocking=non_blocking)
                for name, value in self.values.items()
            },
            self.present,
        )

    def select(self, rows) -> EntityFieldTargets:
        """The records ``rows`` indexes, every field and mask alike."""
        return EntityFieldTargets(
            {name: value[rows] for name, value in self.values.items()},
            {name: mask[rows] for name, mask in self.present.items()},
        )
