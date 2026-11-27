"""A fork record's ``fork_kind`` through the codec.

The connector writes it right after ``fork`` on a fork's records alone; the
decoder reads it into its closed list and refuses a kind outside it, as it
refuses an unknown ``kind`` or ``mode``.
"""

from __future__ import annotations

import json

import pytest

from effects.domain.records import ForkKind
from effects.infrastructure.record_codec import record_from_dict, record_to_dict
from tests.unit.effects.builders import JAVA_GOLDENS


def _golden(name: str, *, fork: bool) -> dict:
    """The first raw record of a Java golden whose ``fork`` flag is ``fork``."""
    lines = (JAVA_GOLDENS / name).read_text(encoding="utf-8").splitlines()
    return next(raw for raw in map(json.loads, filter(str.strip, lines)) if raw["fork"] is fork)


@pytest.mark.parametrize(("golden", "kind"), [
    ("record-intervention.jsonl", ForkKind.INTERVENTION),
    ("record-combat-probe.jsonl", ForkKind.PROBE),
    ("record-protection-fork.jsonl", ForkKind.PROTECTION),
    ("record-directed-fork.jsonl", ForkKind.DIRECTED),
])
def test_each_fork_kind_reads_into_the_closed_list_and_writes_after_fork(golden, kind):
    record = record_from_dict(_golden(golden, fork=True))
    assert record.fork_kind is kind
    written = record_to_dict(record)
    keys = list(written)
    assert written["fork_kind"] == kind.value
    assert keys.index("fork_kind") == keys.index("fork") + 1


def test_a_live_record_reads_and_writes_no_fork_kind():
    record = record_from_dict(_golden("record-combat.jsonl", fork=False))
    assert record.fork_kind is None
    assert "fork_kind" not in record_to_dict(record)


def test_a_fork_kind_outside_the_closed_list_is_refused():
    raw = _golden("record-combat-probe.jsonl", fork=True) | {"fork_kind": "replay"}
    with pytest.raises(ValueError, match="replay"):
        record_from_dict(raw)
