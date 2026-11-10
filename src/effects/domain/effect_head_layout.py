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
    DESIGNATIONS,
    FACE_STATES,
    MANA_SYMBOLS,
    SUPERTYPES,
    THERMOMETER_STEPS,
)
from effects.domain.field_types import FieldType
from effects.domain.forge_vocabularies import NAMED_COUNTERS
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import PlayabilityCheck, ResolutionOutcome, RewriteResult
from effects.domain.sampling import SamplingClass
from effects.domain.state_snapshot import COLORS
from effects.domain.value_targets import VALUE_TARGETS


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
    # Gen-3 (root spec § 11.3), appended so every earlier class keeps its column.
    #: Onto the battlefield from anywhere but a blink: a permanent spell
    #: resolving, a reanimation, a land put onto the battlefield.
    ENTERED = "entered"
    SHUFFLED_INTO_LIBRARY = "shuffled_into_library"
    TO_COMMAND = "to_command"
    #: Moved to no zone at all (a token or a copy leaving the game's zones).
    CEASED_TO_EXIST = "ceased_to_exist"
    PHASED_IN = "phased_in"


#: The zone outcomes in the head's column order, which a target stores as an index.
ZONE_OUTCOMES: tuple[str, ...] = tuple(outcome.value for outcome in ZoneOutcome)


class Duration(StrEnum):
    """How long a lasting change lasts, in the record's spelling (FR-077)."""

    INSTANT = "instant"
    END_OF_TURN = "end_of_turn"
    PERMANENT = "permanent"


class LibraryEvent(StrEnum):
    """What a player's library underwent, the library-events field's members."""

    SCRY = "scry"
    SURVEIL = "surveil"
    TUTOR = "tutor"
    REVEAL = "reveal"
    REORDER = "reorder"
    SHUFFLE = "shuffle"


#: The durations and library events in the head's column order. Append to
#: the enums, never reorder: a target stores a position.
DURATIONS: tuple[str, ...] = tuple(duration.value for duration in Duration)
LIBRARY_EVENTS: tuple[str, ...] = tuple(event.value for event in LibraryEvent)


#: Classes whose records list every outcome event of what they describe, so an
#: affected entity no event of a kind names underwent none of that kind.
_EFFECTFUL = frozenset({
    SamplingClass.RESOLUTION_EFFECT, SamplingClass.REWRITE, SamplingClass.RESOLUTION_COST,
    SamplingClass.COMBAT,
})
#: The effectful classes plus ``continuous``, whose contribution names every
#: P/T, keyword, type and colour change its static makes to the entity.
_STATIC_CHANNELS = _EFFECTFUL | {SamplingClass.CONTINUOUS}


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
    # ── gen-3's outputs (root spec § 11.2, § 11.4), appended ──
    COPIED = "copied"
    SPELL_COPIES = "spell_copies"
    TARGETS_CHANGED = "targets_changed"
    DESIGNATIONS_GAINED = "designations_gained"
    DESIGNATIONS_LOST = "designations_lost"
    CLASS_LEVELS_GAINED = "class_levels_gained"
    DOOR_UNLOCKED = "door_unlocked"
    PAIRED = "paired"
    PROTECTOR_CHANGED = "protector_changed"
    MERGED = "merged"
    FACE_AFTER = "face_after"
    REMOVED_FROM_COMBAT = "removed_from_combat"
    BECAME_BLOCKED = "became_blocked"
    ENTERED_IN_COMBAT = "entered_in_combat"
    SUBTYPES_GAINED = "subtypes_gained"
    SUBTYPES_LOST = "subtypes_lost"
    SUPERTYPES_GAINED = "supertypes_gained"
    SUPERTYPES_LOST = "supertypes_lost"
    ABILITIES_GAINED = "abilities_gained"
    ABILITIES_LOST = "abilities_lost"
    TEXT_CHANGED = "text_changed"
    BASE_POWER_DELTA = "base_power_delta"
    BASE_TOUGHNESS_DELTA = "base_toughness_delta"
    REGENERATED = "regenerated"
    DAMAGE_PREVENTED = "damage_prevented"
    BECAME_MONARCH = "became_monarch"
    TOOK_INITIATIVE = "took_initiative"
    GAINED_CITY_BLESSING = "gained_city_blessing"
    VENTURED = "ventured"
    COMPLETED_DUNGEON = "completed_dungeon"
    SPEED_GAINED = "speed_gained"
    GAME_LOST = "game_lost"
    RAD_DELTA = "rad_delta"
    ENERGY_DELTA = "energy_delta"
    ATTACK_TARGET_LEGAL = "attack_target_legal"
    COMBAT_REQUIREMENT = "combat_requirement"


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
    #: Cards and players both (``damage_prevented``).
    ENTITY = "entity"

    def covers(self, scope: FieldScope) -> bool:
        """Whether an entity of ``scope`` takes this field's silent value."""
        return scope is self or (
            self is FieldScope.ENTITY and scope in (FieldScope.PERMANENT, FieldScope.PLAYER)
        )


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
    #: Whether the field trains only where the affected gate's target fires.
    #: The legality outputs do not (root spec § 11.4): an entity can be a
    #: legal target that nothing affected, and a playability record trains
    #: no gate at all.
    gated: bool = True

    def __post_init__(self) -> None:
        # The type carries the field's width and target shape, so a spec that
        # names no type would fail far from where it was written.
        if not isinstance(self.type, FieldType):
            raise ValueError(f"unhandled field type {self.type}")

    @property
    def arity(self) -> int:
        return len(self.classes) or 1

    @property
    def width(self) -> int:
        return self.type.width(self.arity)

    def supervised_value(
        self, written: Mapping[str, object], unknown: frozenset[str] | set[str],
        *, record_class: SamplingClass, scope: FieldScope,
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
            or not self.scope.covers(scope) or record_class not in self.supervised_by
        ):
            return None
        if self.type.per_member:
            return [self.silent] * self.arity
        return self.silent


#: The counter type named for the generated field every counter type the
#: layout does not list lands in, :attr:`HeadField.COUNTERS_DELTA_OTHER`.
OTHER_COUNTER = "OTHER"


def counter_field(counter_type: str) -> str:
    """The generated per-entity field one counter type's changes land in."""
    return f"counters_delta_{counter_type.lower().replace(' ', '_')}"


#: The counter types with a signed-delta output each (root spec § 11.3):
#: every named counter Forge defines. A keyword counter has none; what it
#: grants reaches ``keywords_gained`` through the keyword change it makes, and
#: its count lands in :attr:`HeadField.COUNTERS_DELTA_OTHER` with every type
#: this list lacks.
OUTPUT_COUNTER_TYPES: tuple[str, ...] = NAMED_COUNTERS


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
        for name in (*OUTPUT_COUNTER_TYPES, OTHER_COUNTER)
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


#: Decision records, which supervise ``target_legal``.
_DECISIONS = frozenset({SamplingClass.PLAYABILITY_DECISION})
#: Attackers and blockers records, which supervise the combat legality bits
#: and, on a declared anchor, its ``tap_state`` (root spec § 11.3).
_DECLARATIONS = frozenset({SamplingClass.PLAYABILITY_LEGALITY})


def _sparse(name: str, field_type: FieldType, scope: FieldScope, **settings) -> FieldSpec:
    """A field of the sparse group (FR-082): every gen-3 output but the
    legality bits is rare, and a head meeting it from step zero learns to
    answer zero for good."""
    return FieldSpec(name, field_type, scope, group=FieldGroup.SPARSE, **settings)


#: The per-entity head's full field list. Order is the head's output layout, so
#: appending is safe and reordering is not.
#:
#: A gen-3 field takes a silent value only where the collector writes the
#: event it reads for every occurrence in the records of its supervising
#: classes; a field whose event is missing or names no subject (``abilities_*``
#: on an Animate, a designation Forge sets without an event) trains only where
#: a record writes it, so a record that cannot say teaches no zero.
PER_ENTITY_FIELDS: tuple[FieldSpec, ...] = (
    # ── permanents and stack entities ──
    FieldSpec(HeadField.ZONE_OUTCOME, FieldType.CATEGORICAL, FieldScope.PERMANENT,
              classes=ZONE_OUTCOMES, silent=float(ZONE_OUTCOMES.index(ZoneOutcome.STAYED))),
    FieldSpec(HeadField.TAP_STATE, FieldType.BINARY, FieldScope.PERMANENT,
              supervised_by=_EFFECTFUL | _DECLARATIONS),
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
              classes=KEYWORDS_CHANGED, group=FieldGroup.SPARSE,
              supervised_by=_STATIC_CHANNELS, silent=0.0),
    FieldSpec(HeadField.KEYWORDS_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
              classes=KEYWORDS_CHANGED, group=FieldGroup.SPARSE,
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
    # ── legality bits: only a payload's own marks set them, gate or no gate ──
    FieldSpec(HeadField.TARGET_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=_DECISIONS, gated=False),
    FieldSpec(HeadField.ATTACKER_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=_DECLARATIONS, gated=False),
    FieldSpec(HeadField.BLOCKER_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=_DECLARATIONS, gated=False),
    FieldSpec(HeadField.MIN_BLOCKERS, FieldType.COUNT, FieldScope.LEGALITY,
              supervised_by=_DECLARATIONS, gated=False),
    # ── gen-3: stack objects and cards (root spec § 11.2) ──
    _sparse(HeadField.COPIED, FieldType.BINARY, FieldScope.PERMANENT, silent=0.0),
    _sparse(HeadField.SPELL_COPIES, FieldType.COUNT, FieldScope.PERMANENT, silent=0.0),
    _sparse(HeadField.TARGETS_CHANGED, FieldType.BINARY, FieldScope.PERMANENT, silent=0.0),
    _sparse(HeadField.DESIGNATIONS_GAINED, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
            classes=DESIGNATIONS),
    _sparse(HeadField.DESIGNATIONS_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
            classes=DESIGNATIONS),
    _sparse(HeadField.CLASS_LEVELS_GAINED, FieldType.COUNT, FieldScope.PERMANENT),
    _sparse(HeadField.DOOR_UNLOCKED, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.PAIRED, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.PROTECTOR_CHANGED, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.MERGED, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.FACE_AFTER, FieldType.CATEGORICAL, FieldScope.PERMANENT,
            classes=FACE_STATES),
    _sparse(HeadField.REMOVED_FROM_COMBAT, FieldType.BINARY, FieldScope.PERMANENT, silent=0.0),
    _sparse(HeadField.BECAME_BLOCKED, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.ENTERED_IN_COMBAT, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.SUBTYPES_GAINED, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
            classes=SUBTYPES_CHANGED, supervised_by=_STATIC_CHANNELS, silent=0.0),
    _sparse(HeadField.SUBTYPES_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
            classes=SUBTYPES_CHANGED, supervised_by=_STATIC_CHANNELS, silent=0.0),
    _sparse(HeadField.SUPERTYPES_GAINED, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
            classes=SUPERTYPES, supervised_by=_STATIC_CHANNELS, silent=0.0),
    _sparse(HeadField.SUPERTYPES_LOST, FieldType.MULTI_BINARY, FieldScope.PERMANENT,
            classes=SUPERTYPES, supervised_by=_STATIC_CHANNELS, silent=0.0),
    _sparse(HeadField.ABILITIES_GAINED, FieldType.COUNT, FieldScope.PERMANENT),
    _sparse(HeadField.ABILITIES_LOST, FieldType.BINARY, FieldScope.PERMANENT),
    _sparse(HeadField.TEXT_CHANGED, FieldType.BINARY, FieldScope.PERMANENT, silent=0.0),
    _sparse(HeadField.BASE_POWER_DELTA, FieldType.SIGNED_DELTA, FieldScope.PERMANENT,
            supervised_by=_STATIC_CHANNELS),
    _sparse(HeadField.BASE_TOUGHNESS_DELTA, FieldType.SIGNED_DELTA, FieldScope.PERMANENT,
            supervised_by=_STATIC_CHANNELS),
    _sparse(HeadField.REGENERATED, FieldType.BINARY, FieldScope.PERMANENT, silent=0.0),
    _sparse(HeadField.DAMAGE_PREVENTED, FieldType.COUNT, FieldScope.ENTITY, silent=0.0),
    # ── gen-3: players ──
    _sparse(HeadField.BECAME_MONARCH, FieldType.BINARY, FieldScope.PLAYER, silent=0.0),
    _sparse(HeadField.TOOK_INITIATIVE, FieldType.BINARY, FieldScope.PLAYER, silent=0.0),
    _sparse(HeadField.GAINED_CITY_BLESSING, FieldType.BINARY, FieldScope.PLAYER),
    _sparse(HeadField.VENTURED, FieldType.BINARY, FieldScope.PLAYER, silent=0.0),
    _sparse(HeadField.COMPLETED_DUNGEON, FieldType.BINARY, FieldScope.PLAYER),
    _sparse(HeadField.SPEED_GAINED, FieldType.COUNT, FieldScope.PLAYER, silent=0.0),
    # No silent value: a burn spell's lethal resolution holds no game-end
    # event (the game-end rescue record does), so its silence is no "lost
    # nothing".
    _sparse(HeadField.GAME_LOST, FieldType.BINARY, FieldScope.PLAYER),
    _sparse(HeadField.RAD_DELTA, FieldType.SIGNED_DELTA, FieldScope.PLAYER, silent=0.0),
    _sparse(HeadField.ENERGY_DELTA, FieldType.SIGNED_DELTA, FieldScope.PLAYER, silent=0.0),
    # ── gen-3: legality (root spec § 11.4) ──
    FieldSpec(HeadField.ATTACK_TARGET_LEGAL, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=_DECLARATIONS, gated=False),
    FieldSpec(HeadField.COMBAT_REQUIREMENT, FieldType.BINARY, FieldScope.LEGALITY,
              supervised_by=_DECLARATIONS, gated=False),
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

#: The gate's name among the per-entity head's loss terms.
GATE_TERM = "gate"
#: The per-entity head's loss terms in the order a per-term vector holds them:
#: the gate, then every field in head order.
PER_ENTITY_TERMS: tuple[str, ...] = (GATE_TERM, *(str(s.name) for s in PER_ENTITY_FIELDS))
#: The per-entity fields a negative binomial scores, each with a learned
#: log-dispersion, in head order (FR-063).
DISPERSED_FIELDS: tuple[str, ...] = tuple(
    str(s.name) for s in PER_ENTITY_FIELDS if s.type.has_magnitude
)
DISPERSION_INDEX: dict[str, int] = {name: i for i, name in enumerate(DISPERSED_FIELDS)}

#: Created-objects head (FR-078): K group slots plus one overflow flag.
CREATED_OBJECT_SLOTS = 4


class CreatedKind(StrEnum):
    """What a created-objects group is (root spec § 11.3), in column order."""

    TOKEN = "token"
    EMBLEM = "emblem"
    #: An effect object Forge builds in the command zone for a lasting effect.
    EFFECT = "effect"
    #: A card made outside the game (``MakeCard``).
    CARD = "card"


CREATED_KINDS: tuple[str, ...] = tuple(kind.value for kind in CreatedKind)


class CreatedSlot:
    """Offsets within one created-objects slot: the two flags, the magnitudes
    (count, power, toughness, then counters per type), then the binaries."""

    PRESENT = 0
    SCRIPTED = 1
    COUNT = 2
    POWER = 3
    TOUGHNESS = 4
    #: One count per :data:`COUNTER_TYPES` member, then every other type.
    COUNTERS = 5
    OTHER_CONTROLLER = COUNTERS + len(COUNTER_TYPES) + 1
    KIND = OTHER_CONTROLLER + 1
    TYPES = KIND + len(CREATED_KINDS)
    SUBTYPES = TYPES + len(CARD_TYPES)
    KEYWORDS = SUBTYPES + len(SUBTYPES_CHANGED)


CREATED_SLOT_WIDTH = CreatedSlot.KEYWORDS + len(KEYWORDS_CHANGED)
CREATED_OBJECTS_WIDTH = CREATED_OBJECT_SLOTS * CREATED_SLOT_WIDTH + 1

#: A slot's columns by name, in :class:`CreatedSlot` order. A column is one
#: field of the created-objects head, the same over every slot.
CREATED_SLOT_COLUMNS: tuple[str, ...] = (
    "present", "scripted", "count", "power", "toughness",
    *(f"counters_{name.lower()}" for name in COUNTER_TYPES), "counters_other",
    "other_controller",
    *(f"kind_{kind}" for kind in CREATED_KINDS),
    *(f"type_{name}" for name in CARD_TYPES),
    *(f"subtype_{name}" for name in SUBTYPES_CHANGED),
    *(f"keyword_{name}" for name in KEYWORDS_CHANGED),
)
#: The overflow flag after the slots: more groups than slots were created.
CREATED_OVERFLOW = "overflow"
#: The created-objects head's loss terms, in the order a per-term vector holds
#: them: each slot column, then the overflow flag.
CREATED_TERMS: tuple[str, ...] = (*CREATED_SLOT_COLUMNS, CREATED_OVERFLOW)
#: A slot's magnitude columns, scored by a negative binomial: count, power,
#: toughness and the counters.
CREATED_MAGNITUDES = slice(CreatedSlot.COUNT, CreatedSlot.OTHER_CONTROLLER)
#: Their names, one learned log-dispersion each, shared over the slots.
CREATED_DISPERSED: tuple[str, ...] = CREATED_SLOT_COLUMNS[CREATED_MAGNITUDES]


@dataclass(frozen=True, slots=True)
class RecordField:
    """One output of a head with one row per record (the verdict and global
    heads): a field type, its classes, and its members, a member repeating
    the type over its own columns (one signed delta per mana symbol).

    A target holds one value per member (one, without members); a categorical's
    is a class index. Members share one mask column: they are supervised together.
    """

    name: str
    type: FieldType
    classes: tuple[str, ...] = ()
    members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type is FieldType.MULTI_BINARY:
            raise ValueError(f"{self.name}: a record field spells a multi-binary as members")

    @property
    def arity(self) -> int:
        return len(self.classes) or 1

    @property
    def count(self) -> int:
        """Target columns: one per member, else one."""
        return len(self.members) or 1

    @property
    def width(self) -> int:
        return self.type.width(self.arity) * self.count


class RecordHeadLayout:
    """A per-record head's fields, their output slices and target columns."""

    def __init__(self, fields: tuple[RecordField, ...]) -> None:
        self.fields = fields
        self.slices: dict[str, tuple[int, int]] = {}
        self.target_slices: dict[str, tuple[int, int]] = {}
        self.index: dict[str, int] = {}
        output = target = 0
        for position, spec in enumerate(fields):
            self.slices[spec.name] = (output, output + spec.width)
            self.target_slices[spec.name] = (target, target + spec.count)
            self.index[spec.name] = position
            output += spec.width
            target += spec.count
        self.width = output
        self.target_width = target
        self.by_name: dict[str, RecordField] = {spec.name: spec for spec in fields}
        #: The fields a negative binomial scores, one learned log-dispersion
        #: each, shared by its members.
        self.dispersed: tuple[str, ...] = tuple(
            str(spec.name) for spec in fields if spec.type.has_magnitude
        )
        self.dispersion_index: dict[str, int] = {
            name: i for i, name in enumerate(self.dispersed)
        }
        #: Field names in head order: a per-term vector's order.
        self.terms: tuple[str, ...] = tuple(str(spec.name) for spec in fields)

    def blank(self) -> tuple[list[float], list[bool]]:
        """An all-unsupervised target row and its per-field mask."""
        return [0.0] * self.target_width, [False] * len(self.fields)

    def put(self, row: tuple[list[float], list[bool]], name: str, value) -> None:
        """Supervise ``name`` with ``value``, one number per member."""
        values, mask = row
        start, end = self.target_slices[name]
        given = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(given) != end - start:
            raise ValueError(f"{name} takes {end - start} values, not {len(given)}")
        values[start:end] = [float(item) for item in given]
        mask[self.index[name]] = True


class VerdictField(StrEnum):
    """The verdict head's outputs at ``[ACT]`` (FR-079, root spec § 11.2, § 11.5)."""

    CAN_PLAY = "can_play"
    AFFORDABLE = "affordable"
    HAS_LEGAL_TARGET = "has_legal_target"
    MANA_PAID = "mana_paid"
    TRIGGER_FIRED = "trigger_fired"
    TIMING = "timing"
    FAILED_CHECKS = "failed_checks"
    COST_ADJUSTMENT = "cost_adjustment"
    RESOLUTION_OUTCOME = "resolution_outcome"
    EFFECTS_CREATED = "effects_created"
    REPLACEMENT_RESULT = "replacement_result"


#: The decision record's three verdict bits, the verdict head's first columns.
VERDICT_BITS: tuple[str, ...] = (
    VerdictField.CAN_PLAY.value, VerdictField.AFFORDABLE.value,
    VerdictField.HAS_LEGAL_TARGET.value,
)
#: One column per check a candidate can fail, in :class:`PlayabilityCheck`
#: order, ``script_restriction`` and ``land_play_limit`` included.
FAILED_CHECKS: tuple[str, ...] = tuple(check.value for check in PlayabilityCheck)
#: The effect half's outcomes (FR-052, root § 11.2): ``countered`` is no class,
#: a countered item writing no effect half.
RESOLUTION_OUTCOMES: tuple[str, ...] = tuple(
    outcome.value for outcome in ResolutionOutcome if outcome is not ResolutionOutcome.COUNTERED)
REPLACEMENT_RESULTS: tuple[str, ...] = tuple(result.value for result in RewriteResult)

VERDICT_LAYOUT = RecordHeadLayout((
    *(RecordField(bit, FieldType.BINARY) for bit in VERDICT_BITS),
    RecordField(VerdictField.MANA_PAID, FieldType.COUNT, members=COLORS),
    RecordField(VerdictField.TRIGGER_FIRED, FieldType.BINARY),
    RecordField(VerdictField.TIMING, FieldType.BINARY),
    RecordField(VerdictField.FAILED_CHECKS, FieldType.BINARY, members=FAILED_CHECKS),
    RecordField(VerdictField.COST_ADJUSTMENT, FieldType.SIGNED_DELTA, members=MANA_SYMBOLS),
    RecordField(VerdictField.RESOLUTION_OUTCOME, FieldType.CATEGORICAL,
                classes=RESOLUTION_OUTCOMES),
    RecordField(VerdictField.EFFECTS_CREATED, FieldType.COUNT),
    RecordField(VerdictField.REPLACEMENT_RESULT, FieldType.CATEGORICAL,
                classes=REPLACEMENT_RESULTS),
))
VERDICT_WIDTH = VERDICT_LAYOUT.width


class GlobalField(StrEnum):
    """The global head's outputs at ``[GLOBAL]``: what belongs to no card or
    player (root spec § 11.1, § 11.2)."""

    TURN_STRUCTURE_CHANGE = "turn_structure_change"
    DAY_NIGHT_CHANGE = "day_night_change"


#: ``turn_structure_change``'s members, each a count.
TURN_STRUCTURE_CHANGES: tuple[str, ...] = (
    "extra_turns", "turns_skipped", "extra_phases", "phases_skipped",
)


class DayNightChange(StrEnum):
    BECAME_DAY = "became_day"
    BECAME_NIGHT = "became_night"
    UNCHANGED = "unchanged"


DAY_NIGHT_CHANGES: tuple[str, ...] = tuple(change.value for change in DayNightChange)

GLOBAL_LAYOUT = RecordHeadLayout((
    RecordField(GlobalField.TURN_STRUCTURE_CHANGE, FieldType.COUNT,
                members=TURN_STRUCTURE_CHANGES),
    RecordField(GlobalField.DAY_NIGHT_CHANGE, FieldType.CATEGORICAL,
                classes=DAY_NIGHT_CHANGES),
))
GLOBAL_WIDTH = GLOBAL_LAYOUT.width

#: ``target -> (start, end)`` into the value head's output (FR-058), each
#: target as wide as a per-entity field of its type.
VALUE_SLICES: dict[str, tuple[int, int]] = {}
_value_offset = 0
for _name, _type in VALUE_TARGETS:
    VALUE_SLICES[_name] = (_value_offset, _value_offset + _type.width(1))
    _value_offset += _type.width(1)

#: Steps of the at-least targets (FR-058b): whether each stated amount is at
#: least k for k = 1 … 40, the board's thermometer length, which pushes ``e``
#: to carry amounts in the code the board side carries.
AT_LEAST_STEPS = THERMOMETER_STEPS
#: ``amount target -> (start, end)`` of its at-least block, after every
#: :data:`VALUE_SLICES` block so the earlier layout of the head is a prefix.
AT_LEAST_SLICES: dict[str, tuple[int, int]] = {}
for _name, _type in VALUE_TARGETS:
    if _type.has_magnitude:
        AT_LEAST_SLICES[_name] = (_value_offset, _value_offset + AT_LEAST_STEPS)
        _value_offset += AT_LEAST_STEPS
VALUE_WIDTH = _value_offset
#: The value head's targets a negative binomial scores, one learned
#: log-dispersion each.
VALUE_DISPERSED: tuple[str, ...] = tuple(
    name for name, field_type in VALUE_TARGETS if field_type.has_magnitude
)


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
