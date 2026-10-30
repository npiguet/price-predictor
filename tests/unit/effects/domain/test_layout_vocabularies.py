"""Every layout tuple read off an enum is the literal tuple a checkpoint trained on.

The input one-hots and the head's columns index tensors by position, so an
enum that gained, lost or reordered a member would move every checkpoint's
reading without a shape changing. Each tuple is pinned here as the literal it
replaced, and to plain strings, so nothing serialized from it changes either.
"""

from __future__ import annotations

import pytest

from effects.domain.effect_head_input import COMBAT_SUBSTEPS, ZONES
from effects.domain.effect_head_layout import (
    AT_LEAST_SLICES,
    DURATIONS,
    LIBRARY_EVENTS,
    OTHER_COUNTER,
    PER_ENTITY_WIDTH,
    VALUE_SLICES,
    VALUE_WIDTH,
    HeadField,
    counter_field,
)
from effects.domain.field_types import DIRECTIONS

_PINNED = [
    (COMBAT_SUBSTEPS, ("first_strike", "regular")),
    (ZONES, ("battlefield", "stack", "hand", "graveyard", "library", "exile", "command")),
    (DURATIONS, ("instant", "end_of_turn", "permanent")),
    (LIBRARY_EVENTS, ("scry", "surveil", "tutor", "reveal", "reorder")),
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
    assert PER_ENTITY_WIDTH == 220
    assert VALUE_WIDTH == 541


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
