"""``relation_types``: § 10.3's families as table rows (T099).

The family names are also the Java collector's ``LinkFamily`` spellings, which
a snapshot's ``links[].type`` carries; ``LinkFamilyContractTest`` reads this
side's source the other way (T058).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from effects.domain.relation_types import (
    FAMILIES,
    RELATION_TYPES,
    TARGET_INDICES,
    Direction,
    relation_type,
    reverse,
)

_JAVA = (Path(__file__).resolve().parents[4] / "forge-connector" / "src" / "main" / "java"
         / "com" / "pricepredictor" / "connector" / "effects" / "LinkFamily.java")


def test_every_family_is_two_types_and_targets_four_per_direction():
    assert len(RELATION_TYPES) == 2 * (len(FAMILIES) - 1) + 2 * TARGET_INDICES
    assert len(set(RELATION_TYPES)) == len(RELATION_TYPES)


def test_the_table_s_families_are_all_there():
    for family in ("line_of", "attached_to", "blocks", "targets", "affects", "linked_by_text",
                   "controls", "copy_of", "has_the_same_name", "last_known_state_of"):
        assert family in FAMILIES


def test_directions_are_distinct_types():
    forward = relation_type("attached_to")
    backward = relation_type("attached_to", Direction.REVERSE)
    assert forward != backward
    assert reverse(forward) == backward and reverse(backward) == forward


@pytest.mark.parametrize("index, shared", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (9, 3)])
def test_target_index_three_and_above_share_the_last_type(index, shared):
    assert relation_type("targets", target_index=index) == relation_type(
        "targets", target_index=shared)


def test_a_target_index_is_required_for_targets_and_refused_elsewhere():
    with pytest.raises(ValueError):
        relation_type("targets")
    with pytest.raises(ValueError):
        relation_type("blocks", target_index=0)
    with pytest.raises(ValueError):
        relation_type("no_such_family")


def test_the_order_is_frozen():
    """A checkpoint stores this list; reordering it silently remaps every table row."""
    assert RELATION_TYPES[:4] == (
        "line_of:forward", "line_of:reverse",
        "line_of_the_same_card:forward", "line_of_the_same_card:reverse")
    assert RELATION_TYPES[-2:] == ("last_known_state_of:forward", "last_known_state_of:reverse")


def test_java_and_python_spell_the_families_alike_in_table_order():
    constants = re.findall(r'^\s+[A-Z_]+\("([a-z_]+)", (?:true|false)\)[,;]',
                           _JAVA.read_text(encoding="utf-8"), flags=re.M)
    assert tuple(constants) == FAMILIES
