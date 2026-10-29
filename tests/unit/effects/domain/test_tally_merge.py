"""merge_into: one tally added into another, field by field."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import pytest

from effects.domain.tally_merge import NOT_MERGED, merge_into


@dataclass(slots=True)
class _Inner:
    count: int = 0


@dataclass(slots=True)
class _Tally:
    name: str = field(default="", metadata=NOT_MERGED)
    counts: Counter[str] = field(default_factory=Counter)
    seen: set[str] = field(default_factory=set)
    by_key: dict[str, set[int]] = field(default_factory=dict)
    total: int = 0
    inner: _Inner = field(default_factory=_Inner)


def test_each_field_adds_by_what_it_holds():
    left = _Tally("left", Counter(a=1), {"x"}, {"k": {1}}, 2, _Inner(3))
    right = _Tally("right", Counter(a=2, b=1), {"y"}, {"k": {2}, "j": {3}}, 5, _Inner(4))

    merge_into(left, right)

    assert left == _Tally(
        "left", Counter(a=3, b=1), {"x", "y"}, {"k": {1, 2}, "j": {3}}, 7, _Inner(7),
    )
    assert right.by_key["j"] is not left.by_key["j"], "a merged set aliases its source"


def test_a_field_it_cannot_add_raises():
    @dataclass
    class _Odd:
        label: str = ""

    with pytest.raises(TypeError, match="label"):
        merge_into(_Odd("a"), _Odd("b"))
