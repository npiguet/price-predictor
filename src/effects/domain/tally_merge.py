"""Adding one tally into another, field by field, by what each field holds.

One function rather than a hand-written ``merge`` per tally class, so a new
field cannot be forgotten in one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields, is_dataclass

#: Field metadata naming a field that identifies one part rather than counting
#: anything in it, such as a shard's name: :func:`merge_into` leaves it alone.
NOT_MERGED = {"merge": False}


def merge_into(target, other) -> None:
    """Add every field of the dataclass ``other`` into ``target``'s.

    A ``Counter`` adds, a set unions, a dict unions its set values key by key,
    an ``int`` sums, an object with its own ``merge`` merges through it, and a
    nested dataclass recurses. Anything else raises ``TypeError`` rather than
    keeping one side's value.
    """
    for spec in fields(target):
        if not spec.metadata.get("merge", True):
            continue
        name = spec.name
        mine, theirs = getattr(target, name), getattr(other, name)
        if isinstance(mine, Counter):
            mine.update(theirs)
        elif isinstance(mine, set):
            mine |= theirs
        elif isinstance(mine, dict):
            for key, values in theirs.items():
                mine.setdefault(key, set()).update(values)
        elif isinstance(mine, int):
            setattr(target, name, mine + theirs)
        elif hasattr(mine, "merge"):
            mine.merge(theirs)
        elif is_dataclass(mine):
            merge_into(mine, theirs)
        else:
            raise TypeError(
                f"{type(target).__name__}.{name}: cannot merge a "
                f"{type(mine).__name__}"
            )
