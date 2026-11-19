"""The directed fork tally a worker prints per JVM lifetime (DF01), summed into
``run.json``: ``DirectedForkTally.java`` prints the line, read back by JSON."""

from __future__ import annotations

import json

from price_predictor.domain.directed_fork_tally import (
    DIRECTED_FORK_TALLY_PREFIX,
    DirectedForkTallyCounts,
    directed_fork_tally_document,
)
from tests.unit.forge_sources import connector_source

START = "Effect records: C:/out/run.0-{life}.jsonl.gz [mode=patched]"


def _line(tally: dict) -> str:
    return DIRECTED_FORK_TALLY_PREFIX + json.dumps(tally)


def _counts(**given: int) -> dict[str, int]:
    names = ("matched", "drawn", "budget_blocked", "taken", "kept", "precondition_threw")
    return {name: given.get(name, 0) for name in names}


def test_the_prefix_is_the_one_the_connector_prints():
    source = connector_source("effects/DirectedForkTally.java")
    assert f'LINE = "{DIRECTED_FORK_TALLY_PREFIX}"' in source


def test_a_lifetime_s_latest_line_stands_and_lifetimes_add_up():
    slot = DirectedForkTallyCounts()
    for line in (
        START.format(life=0), _line({"704.5f": _counts(matched=3, taken=1)}),
        _line({"704.5f": _counts(matched=5, taken=2)}),
        START.format(life=1), _line({"704.5f": _counts(matched=4), "118.12b": _counts(drawn=1)}),
    ):
        slot.add_line(line)
    assert slot.totals() == {
        "704.5f": _counts(matched=9, taken=2), "118.12b": _counts(drawn=1)}


def test_slots_add_up_in_rule_name_order():
    first, second = DirectedForkTallyCounts(), DirectedForkTallyCounts()
    first.add_line(_line({"704.5f": _counts(matched=1)}))
    second.add_line(_line({"118.12b": _counts(matched=2), "704.5f": _counts(matched=3)}))
    document = directed_fork_tally_document([first, second])
    assert list(document) == ["118.12b", "704.5f"]
    assert document["704.5f"]["matched"] == 4


def test_a_rule_id_with_commas_and_equals_signs_is_read_whole():
    slot = DirectedForkTallyCounts()
    slot.add_line("stray text " + _line({"a, b=c": _counts(matched=2)}))
    assert slot.totals() == {"a, b=c": _counts(matched=2)}


def test_a_line_that_is_no_json_object_is_skipped():
    slot = DirectedForkTallyCounts()
    slot.add_line(DIRECTED_FORK_TALLY_PREFIX + "{not json")
    slot.add_line(DIRECTED_FORK_TALLY_PREFIX + "[1]")
    slot.add_line(_line({"704.5f": {"matched": "many"}}))
    assert slot.totals() == {}


def test_no_tally_is_an_empty_document():
    assert directed_fork_tally_document([]) == {}
