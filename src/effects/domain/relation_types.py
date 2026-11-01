"""The relation types the gen-3 trunk reads (root spec § 10.3, research M3, M4).

A relation is a directed, typed link from one slot of a record to another,
acting only inside attention. Every family of § 10.3's table is two types,
one per direction: the forward one is the direction the table names ("from the
line", "from a card to a player"), the reverse one reads the same link from
the other end. ``targets`` has one type per target index 0, 1 and 2 and one
shared by index 3 and above, in each direction.

The order of :data:`RELATION_TYPES` fixes each type's row in every layer's
relation tables, so a checkpoint records it as a list of names and a loader
refuses one whose list differs. A family is only ever appended.

Prior art: ``effect_head_input.SlotKind``, the closed enum whose order fixes an
embedding table; this is a new enum of the same kind, not a variant of it.

Torch-free.
"""

from __future__ import annotations

from enum import StrEnum

#: How many ``targets`` types each direction has; index 3 and above share the last.
TARGET_INDICES = 4

#: § 10.3's families, in the table's order: each becomes a forward and a reverse type.
FAMILIES: tuple[str, ...] = (
    "line_of", "line_of_the_same_card", "mode_of", "merged_with",
    "attached_to",
    "attacks", "blocks", "in_band_with", "must_block", "goaded_by",
    "targets", "source_of", "triggered_by",
    "affects", "created_by",
    "pending_on",
    "linked_by_text",
    "bound_to",
    "controls", "owns", "control_granted_by",
    "copy_of",
    "paired_with", "protects",
    "chose",
    "damaged_this_turn", "fought_this_turn", "crewed_by",
    "has_the_same_name",
    "in_dungeon",
    "last_known_state_of",
)


class Direction(StrEnum):
    FORWARD = "forward"
    REVERSE = "reverse"


def _names() -> tuple[str, ...]:
    names: list[str] = []
    for family in FAMILIES:
        for direction in Direction:
            if family == "targets":
                names += [f"targets_{index}:{direction}" for index in range(TARGET_INDICES)]
            else:
                names.append(f"{family}:{direction}")
    return tuple(names)


#: Every relation type's name, in table-row order.
RELATION_TYPES: tuple[str, ...] = _names()
_INDEX = {name: index for index, name in enumerate(RELATION_TYPES)}


def relation_type(family: str, direction: Direction = Direction.FORWARD,
                  target_index: int | None = None) -> int:
    """The table row of one family's type in one direction.

    ``target_index`` is required for ``targets`` and refused elsewhere; an index
    at or past :data:`TARGET_INDICES` - 1 reads the shared last type.
    """
    if (family == "targets") != (target_index is not None):
        raise ValueError(f"target_index is given for targets and only for targets ({family})")
    if family == "targets":
        if target_index < 0:
            raise ValueError(f"target index {target_index}")
        name = f"targets_{min(target_index, TARGET_INDICES - 1)}:{Direction(direction)}"
    else:
        name = f"{family}:{Direction(direction)}"
    try:
        return _INDEX[name]
    except KeyError:
        raise ValueError(f"no relation family {family!r}") from None


def reverse(row: int) -> int:
    """The type that reads the same link from the other end."""
    name = RELATION_TYPES[row]
    stem, _, direction = name.rpartition(":")
    other = Direction.REVERSE if direction == Direction.FORWARD else Direction.FORWARD
    return _INDEX[f"{stem}:{other}"]
