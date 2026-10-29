"""Adding one tally into another, field by field, by what each field holds.

The corpus passes count per shard in a worker and fold the shards together in
the main process. Every tally they keep is a dataclass of counters, sets, maps
of sets and plain counts, and each kind adds one way, so one function adds
them all rather than a hand-written ``merge`` per class that a new field can
be forgotten in.
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
    an ``int`` sums, an object with a ``merge`` method of its own merges
    through it (one that has an invariant to check, such as a heap size), and
    a nested dataclass merges field by field. Anything else is a field this
    function does not know how to add, and raises rather than keeping one
    side's value.
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
