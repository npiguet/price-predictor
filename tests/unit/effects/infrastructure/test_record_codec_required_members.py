"""A gen-3 record carries every collection member on every line, so the
decoder reads each strictly: a missing one is a decode error naming it, not a
default. Each case is a real gen-3 record less one member."""

from __future__ import annotations

import copy

import pytest

from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records


def _record() -> dict:
    return copy.deepcopy(gen3_fixture_records()[0])


@pytest.mark.parametrize(
    "member", ["mode", "interventional", "fork", "synthetic", "random_seat"])
def test_a_record_without_a_collection_member_does_not_decode(member):
    raw = _record()
    del raw[member]

    with pytest.raises(KeyError, match=member):
        record_from_dict(raw)


def test_an_entity_without_lines_does_not_decode():
    raw = _record()
    del raw["state"]["entities"][0]["lines"]

    with pytest.raises(KeyError, match="lines"):
        record_from_dict(raw)
