"""Split a budget equally across members, passing on what a full member leaves.

Curation uses it at every level FR-049 balances: across the rule families of a
class, across the outcome signatures of a family, across the real and what-if
halves of the legality class, and across the ability texts of one cell. One
allocator rather than four, because the four must agree on what "equal" means
when a member runs out — a member that cannot take its share is written at
capacity and the remainder is split equally again over the members that still
can, until every member is full or the budget is spent.

The result is an exact integer split: the shares sum to ``min(budget, total
capacity)``. The remainder of an integer division goes one unit at a time to
the members in sorted key order, so the split is a pure function of its inputs.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def allocate(budget: int, capacities: Mapping[K, int]) -> dict[K, int]:
    """``member -> share``: equal shares, capped by capacity, redistributed.

    A member with zero capacity takes nothing; with no members the result is
    empty. A budget above the total capacity fills every member and the rest
    goes unspent — the caller reads the shortfall from the difference.
    """
    shares: dict[K, int] = {key: 0 for key in capacities}
    open_members = sorted(
        (key for key, capacity in capacities.items() if capacity > 0), key=_order,
    )
    remaining = max(0, int(budget))
    while remaining > 0 and open_members:
        each, extra = divmod(remaining, len(open_members))
        still_open = []
        spent = 0
        for position, key in enumerate(open_members):
            offer = each + (1 if position < extra else 0)
            room = capacities[key] - shares[key]
            take = min(offer, room)
            shares[key] += take
            spent += take
            if shares[key] < capacities[key]:
                still_open.append(key)
        remaining -= spent
        if spent == 0:
            break
        open_members = still_open
    return shares


def _order(key) -> tuple[str, str]:
    """Sort key over arbitrary hashable members, deterministic across runs.

    A set's ``repr`` follows string hashing, which is salted per process, so
    its members are sorted first; everything else sorts by its ``repr``.
    """
    if isinstance(key, (frozenset, set)):
        return (type(key).__name__, repr(sorted(key, key=repr)))
    if isinstance(key, tuple):
        return (type(key).__name__, repr(tuple(_order(part) for part in key)))
    return (type(key).__name__, repr(key))
