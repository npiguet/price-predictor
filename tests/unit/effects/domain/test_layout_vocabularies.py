"""Every layout tuple read off an enum is the literal tuple a checkpoint trained on.

The input one-hots and the head's columns index tensors by position, so an
enum that gained, lost or reordered a member would move every checkpoint's
reading without a shape changing. Each tuple is pinned here as the literal it
replaced, and to plain strings, so nothing serialized from it changes either.
"""

from __future__ import annotations

import pytest

from effects.domain.effect_head_input import (
    COMBAT_SUBSTEPS,
    ENTITY_ROLES,
    PENDING_EVENT_MODES,
    ZONES,
    SlotKind,
    card_features,
    card_segments,
    feature_widths,
)
from effects.domain.effect_head_layout import (
    AT_LEAST_SLICES,
    CREATED_OBJECTS_WIDTH,
    CREATED_SLOT_WIDTH,
    DURATIONS,
    GLOBAL_WIDTH,
    LIBRARY_EVENTS,
    OTHER_COUNTER,
    PER_ENTITY_WIDTH,
    VALUE_SLICES,
    VALUE_WIDTH,
    VERDICT_WIDTH,
    ZONE_OUTCOMES,
    HeadField,
    counter_field,
)
from effects.domain.field_types import DIRECTIONS
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records

_PINNED = [
    (COMBAT_SUBSTEPS, ("first_strike", "regular")),
    (ZONES, ("battlefield", "stack", "hand", "graveyard", "library", "exile", "command")),
    (ENTITY_ROLES, ("tier", "library_card", "exiled_with", "mutate_component", "last_known")),
    (DURATIONS, ("instant", "end_of_turn", "permanent")),
    (LIBRARY_EVENTS, ("scry", "surveil", "tutor", "reveal", "reorder", "shuffle")),
    (ZONE_OUTCOMES, (
        "stayed", "died", "exiled", "to_hand", "library_top", "library_bottom",
        "face_changed", "phased_out", "blinked", "countered",
        "entered", "shuffled_into_library", "to_command", "ceased_to_exist", "phased_in",
    )),
    (DIRECTIONS, ("negative", "zero", "positive")),
]


@pytest.mark.parametrize(("derived", "literal"), _PINNED)
def test_a_derived_vocabulary_is_its_literal_tuple_of_plain_strings(derived, literal):
    assert derived == literal
    assert all(type(member) is str for member in derived)


def test_the_generated_counter_fields_keep_their_names():
    assert counter_field("M1M1") == "counters_delta_m1m1"
    assert counter_field(OTHER_COUNTER) == HeadField.COUNTERS_DELTA_OTHER == "counters_delta_other"


def test_the_heads_keep_their_widths():
    """Gen-3's widths (root spec § 11.3) with the pilot-generated lists of
    181 subtypes and 70 keywords: the per-entity head's 1,634 columns are
    932 for the 233 counter fields, a created-objects slot is 285 columns,
    the verdict head's 169 are 136 for the cost adjustment's 34 mana symbols
    and 4 for the resolution outcome (FR-052: resolved, fizzled, partially
    fizzled, declined). The widths move only when the lists are regenerated
    from the collected corpus."""
    assert (len(SUBTYPES_CHANGED), len(KEYWORDS_CHANGED)) == (181, 70)
    assert PER_ENTITY_WIDTH == 1634
    assert CREATED_SLOT_WIDTH == 285
    assert CREATED_OBJECTS_WIDTH == 1141
    assert VERDICT_WIDTH == 169
    assert GLOBAL_WIDTH == 7
    assert VALUE_WIDTH == 541


def test_the_card_slot_keeps_its_width_and_parts():
    """A card slot is 575 features in 20 parts. A combat record's damage
    assignment is no part of it (D3): it reads the first-strike rule
    without the keyword. The role is (finding 4.7): six columns, the five
    of :data:`ENTITY_ROLES` and "other". The pending event's part is six
    columns: a pending event, the card is its subject, its source, its
    defender, among its blockers, its cause; a face-down card's real face
    is 27 (types, colours, mana value, printed power and toughness, the
    subtype count); the hidden keywords' two: can't block, can't attack (finding
    4.8). The width moves with the kept keyword flags and the other lists
    the parts one-hot over."""
    record = record_from_dict(gen3_fixture_records()[0])
    entity = record.state.entities[0]
    assert len(card_features(entity, record)) == 575
    assert list(card_segments(entity, record)) == [
        "characteristics", "face_state", "real_face", "mana_cost", "printed_loyalty_defense",
        "status", "role", "keyword_flags", "hidden_keywords", "relation_to_act",
        "designations", "x_paid",
        "chosen", "costs_paid", "history", "order_rank", "stack_extras",
        "pending_event", "command_zone", "comparable_amounts",
    ]


def test_the_global_and_player_slots_keep_their_widths():
    """The ``[GLOBAL]`` slot is 226 features, 83 of them the pending event's
    mode, one-hot over the 81 :data:`PENDING_EVENT_MODES` and "other", and
    its combat bit, and 16 its from- and to-zone, each one-hot over
    :data:`ZONES` and "other"; a player slot is 116, three the pending
    event's subject, defender and cause (finding 4.8)."""
    assert len(PENDING_EVENT_MODES) == 81
    widths = feature_widths([record_from_dict(gen3_fixture_records()[0])])
    assert (widths[SlotKind.GLOBAL], widths[SlotKind.PLAYER]) == (226, 116)


def test_the_value_head_keeps_its_slices():
    """Each value target as wide as a per-entity field of its type: one column
    for a count or a cost flag, four for a signed change."""
    assert VALUE_SLICES == {
        "damage": (0, 1), "power_change": (1, 5), "toughness_change": (5, 9),
        "counters_placed": (9, 10), "counters_removed": (10, 11), "cards_drawn": (11, 12),
        "mana_w": (12, 13), "mana_u": (13, 14), "mana_b": (14, 15), "mana_r": (15, 16),
        "mana_g": (16, 17), "mana_c": (17, 18), "mana_generic": (18, 19),
        "cost_taps": (19, 20), "cost_sacrifices": (20, 21),
    }
    assert list(AT_LEAST_SLICES) == [
        "damage", "power_change", "toughness_change", "counters_placed",
        "counters_removed", "cards_drawn", "mana_w", "mana_u", "mana_b", "mana_r",
        "mana_g", "mana_c", "mana_generic",
    ]
    assert AT_LEAST_SLICES["damage"] == (21, 61)
