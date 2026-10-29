"""Split a budget equally across members, passing on what a full member leaves.

One allocator for every level FR-049 balances, so they agree on what "equal"
means when a member runs out.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def allocate(budget: int, capacities: Mapping[K, int]) -> dict[K, int]:
    """``member -> share``: equal shares, capped by capacity, redistributed.

    The shares sum to ``min(budget, total capacity)`` exactly; an integer
    division's remainder goes one unit at a time in sorted key order, so the
    split is a pure function of its inputs.
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

    A set's members are sorted first, since its ``repr`` follows the salted
    string hash. A string enum sorts as its string, so a sentinel family sorts
    among the script words rather than ahead of them.
    """
    if isinstance(key, str):
        return ("str", repr(str(key)))
    if isinstance(key, (frozenset, set)):
        return (type(key).__name__, repr(sorted(key, key=repr)))
    if isinstance(key, tuple):
        return (type(key).__name__, repr(tuple(_order(part) for part in key)))
    return (type(key).__name__, repr(key))
