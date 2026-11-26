"""What a checkpoint records about the layout it was built on, and the
refusal of a checkpoint built on another.

Every ordered name list that fixes a row or a column of the model's tables
(an input one-hot, an id table, an output head's terms and classes) is
recorded as the sha256 of its names in order, so that reordering two members,
or regenerating a list at the same length, refuses the checkpoint instead of
loading it with shifted columns.
"""

from __future__ import annotations

from collections.abc import Iterable

from effects.domain.canonical_digest import canonical_sha256
from effects.domain.card_features import CANT_ATTACK_KEYWORDS, CANT_BLOCK_KEYWORDS, CARD_PARTS
from effects.domain.effect_head_layout import (
    COUNTER_TYPES,
    CREATED_KINDS,
    CREATED_TERMS,
    DAY_NIGHT_CHANGES,
    DURATIONS,
    FAILED_CHECKS,
    GATE_TERM,
    GLOBAL_LAYOUT,
    LIBRARY_EVENTS,
    PER_ENTITY_FIELDS,
    REPLACEMENT_RESULTS,
    RESOLUTION_OUTCOMES,
    TURN_STRUCTURE_CHANGES,
    VERDICT_LAYOUT,
    ZONE_OUTCOMES,
    RecordHeadLayout,
)
from effects.domain.field_types import DIRECTIONS, FieldType
from effects.domain.forge_vocabularies import KEYWORD_COUNTERS, NAMED_COUNTERS, SUBTYPES
from effects.domain.input_layout import SlotKind
from effects.domain.input_vocabularies import (
    CARD_TYPES,
    CHOSEN_DIRECTIONS,
    CHOSEN_PARITIES,
    COMBAT_SUBSTEPS,
    DAY_NIGHT,
    DESIGNATIONS,
    EFFECT_DURATIONS,
    ENTITY_ROLES,
    EVENT_TYPES,
    FACE_STATES,
    MANA_SYMBOLS,
    MOMENTS,
    PENDING_EVENT_MODES,
    PHASES,
    PLAYABILITY_SUBKINDS,
    RECORD_KINDS,
    SUPERTYPES,
    THIS_TURN_COUNTERS,
    TURN_COUNTS,
    ZONES,
)
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.relation_types import RELATION_TYPES, TARGET_INDICES
from effects.domain.state_snapshot import COLORS
from effects.domain.value_targets import VALUE_TARGETS

#: Where a checkpoint's ``training_settings`` record the layout it trained on.
INPUT_LAYOUT_KEY = "input_layout"

#: The input layout every training run builds and records, and the only one a
#: loader accepts (FR-045): a checkpoint recording another layout, or none, was
#: trained on inputs this code no longer builds. Gen-3 is the card's lines as
#: ability slots, its gen-3 card, player, global and act inputs, command-zone
#: slots and the relations (root spec § 9 – § 10).
INPUT_LAYOUT = "gen-3"

#: Where a checkpoint records the relations' ``targets`` index cap (research M4).
RELATION_TARGET_INDICES_KEY = "relation_target_indices"


class CheckpointLayoutError(ValueError):
    """A checkpoint trained on an input layout this code does not build."""


def _field_names(name: str, field_type: FieldType, classes: Iterable[str] = (),
                 members: Iterable[str] = ()) -> tuple[str, ...]:
    """One output field as names: itself with its type, then its classes and
    its members in column order."""
    return (f"{name}:{field_type.value}", *(f"{name}={cls}" for cls in classes),
            *(f"{name}.{member}" for member in members))


def _per_entity_head() -> tuple[str, ...]:
    return (GATE_TERM, *(
        name for spec in PER_ENTITY_FIELDS
        for name in _field_names(spec.name, spec.type, spec.classes)
    ))


def _record_head(layout: RecordHeadLayout) -> tuple[str, ...]:
    return tuple(
        name for spec in layout.fields
        for name in _field_names(spec.name, spec.type, spec.classes, spec.members)
    )


def _value_head() -> tuple[str, ...]:
    return tuple(name for target, field_type in VALUE_TARGETS
                 for name in _field_names(target, field_type))


#: Every name list whose order fixes a row or a column of the model's tables,
#: by the name a checkpoint records its digest under: the trunk's relation
#: types and slot kinds, the card, player, global and act slots' id lists,
#: flags and one-hots, a set by its sorted members, then every output head
#: with the class lists its fields read.
LAYOUT_VOCABULARIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("relation_types", RELATION_TYPES),
    ("slot_kinds", tuple(kind.name for kind in SlotKind)),
    ("subtypes", SUBTYPES),
    ("named_counters", NAMED_COUNTERS),
    ("keyword_counters", KEYWORD_COUNTERS),
    ("kept_keyword_flags", KEPT_KEYWORD_FLAGS),
    ("card_parts", CARD_PARTS),
    ("phases", PHASES),
    ("combat_substeps", COMBAT_SUBSTEPS),
    ("zones", ZONES),
    ("card_types", CARD_TYPES),
    ("supertypes", SUPERTYPES),
    ("colors", COLORS),
    ("this_turn_counters", THIS_TURN_COUNTERS),
    ("turn_counts", TURN_COUNTS),
    ("face_states", FACE_STATES),
    ("entity_roles", ENTITY_ROLES),
    ("day_night", DAY_NIGHT),
    ("effect_durations", EFFECT_DURATIONS),
    ("designations", DESIGNATIONS),
    ("mana_symbols", MANA_SYMBOLS),
    ("chosen_directions", CHOSEN_DIRECTIONS),
    ("chosen_parities", CHOSEN_PARITIES),
    ("event_types", EVENT_TYPES),
    ("pending_event_modes", PENDING_EVENT_MODES),
    ("record_kinds", RECORD_KINDS),
    ("moments", MOMENTS),
    ("playability_subkinds", PLAYABILITY_SUBKINDS),
    ("cant_block_keywords", tuple(sorted(CANT_BLOCK_KEYWORDS))),
    ("cant_attack_keywords", tuple(sorted(CANT_ATTACK_KEYWORDS))),
    ("directions", DIRECTIONS),
    ("zone_outcomes", ZONE_OUTCOMES),
    ("durations", DURATIONS),
    ("library_events", LIBRARY_EVENTS),
    ("subtypes_changed", SUBTYPES_CHANGED),
    ("keywords_changed", KEYWORDS_CHANGED),
    ("counter_types", COUNTER_TYPES),
    ("created_kinds", CREATED_KINDS),
    ("failed_checks", FAILED_CHECKS),
    ("resolution_outcomes", RESOLUTION_OUTCOMES),
    ("replacement_results", REPLACEMENT_RESULTS),
    ("turn_structure_changes", TURN_STRUCTURE_CHANGES),
    ("day_night_changes", DAY_NIGHT_CHANGES),
    ("per_entity_head", _per_entity_head()),
    ("created_objects_head", CREATED_TERMS),
    ("verdict_head", _record_head(VERDICT_LAYOUT)),
    ("global_head", _record_head(GLOBAL_LAYOUT)),
    ("value_head", _value_head()),
)

_DIGEST_SUFFIX = "_digest"


def layout_settings() -> dict:
    """What a run records about the layout it was built on: the input layout,
    the relations' target cap, and a digest of every
    :data:`LAYOUT_VOCABULARIES` list under ``<name>_digest``."""
    return {
        INPUT_LAYOUT_KEY: INPUT_LAYOUT,
        RELATION_TARGET_INDICES_KEY: TARGET_INDICES,
        **{f"{name}{_DIGEST_SUFFIX}": canonical_sha256(list(names))
           for name, names in LAYOUT_VOCABULARIES},
    }


def _layout_generation(recorded) -> str:
    """The generation a checkpoint's recorded layout names: a checkpoint that
    records none predates the layout key, which is gen-1."""
    return "gen-1" if recorded is None else str(recorded)


def _require_generation(recorded) -> None:
    if recorded != INPUT_LAYOUT:
        raise CheckpointLayoutError(
            f"the checkpoint is a {_layout_generation(recorded)} checkpoint "
            f"(input layout {recorded!r}); this code reads {INPUT_LAYOUT} "
            "checkpoints only. Retrain it on the current code."
        )


def _labels(key: str) -> tuple[str, str]:
    """What a refusal calls ``key`` when it is missing and when it differs."""
    if key.endswith(_DIGEST_SUFFIX):
        name = key.removesuffix(_DIGEST_SUFFIX)
        return f"{name} digest", f"{name} order"
    return key, key


def _require_recorded(settings, key: str, expected) -> None:
    missing, differs = _labels(key)
    if key not in settings:
        raise CheckpointLayoutError(
            f"the checkpoint records no {missing}: it was written before this code "
            "recorded one, so the columns it trained on are unknown. Retrain it."
        )
    if settings[key] != expected:
        raise CheckpointLayoutError(
            f"the checkpoint records another {differs} than this code's ({INPUT_LAYOUT!r} "
            "layout); its tables would be read in the wrong rows and columns. Retrain it."
        )


def require_input_layout(training_settings) -> None:
    """Refuse a checkpoint trained on another input layout, or whose recorded
    target cap or any vocabulary's order differs from this code's or is not
    recorded; a refusal names the generation, else the setting."""
    settings = training_settings or {}
    _require_generation(settings.get(INPUT_LAYOUT_KEY))
    for key, expected in layout_settings().items():
        _require_recorded(settings, key, expected)
