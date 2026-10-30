"""The effect head's output layout: every field, slot and bit, and where it sits.

A field is scored on an affected entity only where present (FR-077, FR-081):
the record wrote a value, or the field has a **silent** value — what "affected,
and nothing said" means where absence is itself an observation (no damage, no
counters). The silent value applies only where the record's sampling class
supervises the field and the entity is the field's scope; the tap state, the
durations and the legality bits have none.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from effects.domain.effect_head_input import (
    CARD_TYPES,
    COUNTER_TYPES,
    OVERLAY_KEYWORDS,
    THERMOMETER_STEPS,
)
from effects.domain.sampling import SamplingClass
from effects.domain.state_snapshot import COLORS
from effects.domain.value_targets import VALUE_TARGETS, ValueKind


class ZoneOutcome(StrEnum):
    """Where an entity ends up, a closed vocabulary (FR-077)."""

    STAYED = "stayed"
    DIED = "died"
    EXILED = "exiled"
    TO_HAND = "to_hand"
    LIBRARY_TOP = "library_top"
    LIBRARY_BOTTOM = "library_bottom"
    TRANSFORMED = "transformed"
    FACE_CHANGED = "face_changed"
    PHASED_OUT = "phased_out"
    BLINKED = "blinked"
    COUNTERED = "countered"


#: The zone outcomes in the head's column order, which a target stores as an index.
ZONE_OUTCOMES: tuple[ZoneOutcome, ...] = tuple(ZoneOutcome)
DURATIONS: tuple[str, ...] = ("instant", "end_of_turn", "permanent")
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


#: A signed delta's direction classes, in the head's column order; a target's
#: sign plus one is its class.
DIRECTIONS: tuple[str, ...] = ("negative", "zero", "positive")
#: A signed delta's columns: a logit per direction, then the magnitude's log
#: rate under the count loss.
SIGNED_DIRECTIONS = slice(0, len(DIRECTIONS))
SIGNED_MAGNITUDE = len(DIRECTIONS)
SIGNED_WIDTH = SIGNED_MAGNITUDE + 1


class FieldType(StrEnum):
    BINARY = "binary"
    MULTI_BINARY = "multi_binary"
    COUNT = "count"
    SIGNED_DELTA = "signed_delta"
    CATEGORICAL = "categorical"


class HeadField(StrEnum):
    """The per-entity fields code names one by one; the counter and mana
    fields are generated per counter type and colour, so their builders name them."""

    ZONE_OUTCOME = "zone_outcome"
    TAP_STATE = "tap_state"
    DAMAGE_TAKEN = "damage_taken"
    POWER_DELTA = "power_delta"
    TOUGHNESS_DELTA = "toughness_delta"
    PT_DURATION = "pt_duration"
    #: The generated counter field a counter type the layout does not list lands in.
    COUNTERS_DELTA_OTHER = "counters_delta_other"
    TYPES_GAINED = "types_gained"
    TYPES_LOST = "types_lost"
    COLORS_GAINED = "colors_gained"
    COLORS_LOST = "colors_lost"
    TYPE_COLOR_DURATION = "type_color_duration"
    KEYWORDS_GAINED = "keywords_gained"
    KEYWORDS_LOST = "keywords_lost"
    CONTROL_CHANGE = "control_change"
    ATTACHED_TO_CHANGE = "attached_to_change"
    LIFE_DELTA = "life_delta"
    CARDS_DRAWN = "cards_drawn"
    CARDS_DISCARDED = "cards_discarded"
    CARDS_MILLED = "cards_milled"
    LIBRARY_EVENTS = "library_events"
    POISON_DELTA = "poison_delta"
    TARGET_LEGAL = "target_legal"
    ATTACKER_LEGAL = "attacker_legal"
    BLOCKER_LEGAL = "blocker_legal"
    MIN_BLOCKERS = "min_blockers"


class FieldGroup(StrEnum):
    """When a field starts training: the sparse group (FR-082) waits for the
    curriculum epoch."""

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
    #: A categorical field's classes or a multi-binary field's members, in the
    #: head's column order; empty for the scalar types.
    classes: tuple[str, ...] = ()
    group: FieldGroup = FieldGroup.DENSE
    #: Sampling classes whose records carry a target for this field.
    supervised_by: frozenset[SamplingClass] = field(default_factory=lambda: _EFFECTFUL)
    #: The value an affected entity of ``scope`` takes when a record of a
    #: supervising class names no change to the field; every member of a
    #: multi-binary field. None for a field only a written value supervises.
    silent: float | None = None

    @property
    def arity(self) -> int:
        return len(self.classes) or 1

    @property
    def width(self) -> int:
        match self.type:
            case FieldType.BINARY | FieldType.COUNT:
                return 1
            case FieldType.MULTI_BINARY | FieldType.CATEGORICAL:
                return self.arity
            case FieldType.SIGNED_DELTA:
                return SIGNED_WIDTH
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


def counter_field(counter_type: str) -> str:
    """The generated per-entity field one counter type's changes land in."""
    return f"counters_delta_{counter_type.lower()}"


def mana_field(color: str) -> str:
    """The generated per-entity field one colour's mana changes land in."""
    return f"mana_delta_{color.lower()}"


def _counter_fields() -> tuple[FieldSpec, ...]:
    return tuple(
        FieldSpec(
            name=counter_field(name),
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
            name=mana_field(color),
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
    FieldSpec(HeadField.ZONE_OUTCOME, FieldType.CATEGORICAL, FieldScope.PERMANENT,
              classes=ZONE_OUTCOMES, silent=float(ZONE_OUTCOMES.index(ZoneOutcome.STAYED))),
    FieldSpec(HeadField.TAP_STATE, FieldType.BINARY, FieldScope.PERMANENT),
    FieldSpec(HeadField.DAMAGE_TAKEN, FieldType.COUNT, FieldScope.PERMANENT, silent=0.0),
    FieldSpec(HeadField.POWER_DELTA, FieldType.SIGNED_DELTA, FieldScope.PERMANENT,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.TOUGHNESS_DELTA, FieldType.SIGNED_DELTA, FieldScope.PERMANENT,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.PT_DURATION, FieldType.CATEGORICAL, FieldScope.PERMANENT,
              classes=DURATIONS, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS),
    *_counter_fields(),
    FieldSpec(HeadField.TYPES_GAINED, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=CARD_TYPES, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.TYPES_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=CARD_TYPES, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.COLORS_GAINED, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=COLORS, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.COLORS_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=COLORS, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.TYPE_COLOR_DURATION, FieldType.CATEGORICAL, FieldScope.PERMANENT,
              classes=DURATIONS, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS),
    FieldSpec(HeadField.KEYWORDS_GAINED, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=OVERLAY_KEYWORDS, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.KEYWORDS_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=OVERLAY_KEYWORDS, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.CONTROL_CHANGE, FieldType.BINARY, FieldScope.PERMANENT,
              group=FieldGroup.SPARSE, silent=0.0),
    FieldSpec(HeadField.ATTACHED_TO_CHANGE, FieldType.BINARY, FieldScope.PERMANENT,
              group=FieldGroup.SPARSE, silent=0.0),
    # ── players ──
    FieldSpec(HeadField.LIFE_DELTA, FieldType.SIGNED_DELTA, FieldScope.PLAYER, silent=0.0),
    FieldSpec(HeadField.CARDS_DRAWN, FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    FieldSpec(HeadField.CARDS_DISCARDED, FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    FieldSpec(HeadField.CARDS_MILLED, FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    FieldSpec(HeadField.LIBRARY_EVENTS, FieldType.MULTI_BINARY, FieldScope.PLAYER,
              classes=LIBRARY_EVENTS, silent=0.0),
    # Not in FR-077's player group, but gate 2's infect row needs it.
    FieldSpec(HeadField.POISON_DELTA, FieldType.SIGNED_DELTA, FieldScope.PLAYER, silent=0.0),
    *_mana_fields(),
    # ── legality bits: only a payload's own marks set them ──
    FieldSpec(HeadField.TARGET_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_DECISION})),
    FieldSpec(HeadField.ATTACKER_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_LEGALITY})),
    FieldSpec(HeadField.BLOCKER_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=frozenset({SamplingClass.PLAYABILITY_LEGALITY})),
    FieldSpec(HeadField.MIN_BLOCKERS, FieldType.COUNT, FieldScope.LEGALITY,
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
    """Offsets within one created-objects slot."""

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

#: Value head (FR-058) widths per target kind.
_VALUE_WIDTHS: dict[ValueKind, int] = {
    ValueKind.COUNT: 1, ValueKind.BINARY: 1, ValueKind.SIGNED: SIGNED_WIDTH,
}
#: ``target -> (start, end)`` into the value head's output.
VALUE_SLICES: dict[str, tuple[int, int]] = {}
_value_offset = 0
for _name, _kind in VALUE_TARGETS:
    VALUE_SLICES[_name] = (_value_offset, _value_offset + _VALUE_WIDTHS[_kind])
    _value_offset += _VALUE_WIDTHS[_kind]

#: Steps of the at-least targets (FR-058b): whether each stated amount is at
#: least k for k = 1 … 40, the board's thermometer length, which pushes ``e``
#: to carry amounts in the code the board side carries.
AT_LEAST_STEPS = THERMOMETER_STEPS
#: ``amount target -> (start, end)`` of its at-least block, after every
#: :data:`VALUE_SLICES` block so the earlier layout of the head is a prefix.
AT_LEAST_SLICES: dict[str, tuple[int, int]] = {}
for _name, _kind in VALUE_TARGETS:
    if _kind is not ValueKind.BINARY:
        AT_LEAST_SLICES[_name] = (_value_offset, _value_offset + AT_LEAST_STEPS)
        _value_offset += AT_LEAST_STEPS
VALUE_WIDTH = _value_offset


def active_fields(
    present_classes: frozenset[str], *, sparse: bool,
) -> tuple[FieldSpec, ...]:
    """Fields that contribute loss right now: supervised by a class present
    (FR-085), and sparse ones only once ``sparse`` is on, since a head that
    meets them from step zero learns to output zero for good (FR-082)."""
    return tuple(
        spec for spec in PER_ENTITY_FIELDS
        if spec.supervised_by & present_classes
        and (spec.group is FieldGroup.DENSE or sparse)
    )
