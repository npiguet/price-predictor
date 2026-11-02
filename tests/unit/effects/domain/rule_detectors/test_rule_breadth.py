"""The breadth check: the pool a detector's rule is cut from, on real gen-2 records (T036)."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.domain.records import Moment, RecordKind
from effects.domain.rule_breadth import (
    POOL,
    TAGGED,
    BreadthRow,
    breadth_counts,
    breadth_rows,
    flagged_rows,
    pool_params,
)
from effects.domain.rule_detectors import DetectorIndex, families
from effects.domain.rule_detectors.acting_line import acting_line
from tests.unit.effects.domain.rule_detectors.builders import catalogue, declaration, scenario


def test_every_family_s_general_parameters_are_its_own():
    for family in families().values():
        assert family.general and family.general <= family.params, family.id


def test_the_pool_keeps_the_rule_s_subject_and_drops_its_situation():
    family = families()["outcome_event"]
    narrowed = declaration("outcome_event", "resolution", event="zone_change",
                           to_zone="battlefield", api="OpenAttraction",
                           script_param={"Origin": "Library"})
    assert pool_params(narrowed, family) == {"event": "zone_change", "api": "OpenAttraction"}


def _zone_change_effect(gen2_records, gen2_sidecars):
    """A real effect half moving a card, its acting line's API and the zone
    the first move went to."""
    for record in gen2_records:
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            continue
        line = acting_line(record, gen2_sidecars)
        moves = [e for e in record.payload.listed_events() if e.type.value == "zone_change"
                 and (e.params or {}).get("to_zone")]
        if line is not None and line.script_api_type and moves:
            return record, line.script_api_type, moves[0].params["to_zone"]
    pytest.skip("no effect half in the fixture moves a card")


def _index(*declared):
    return DetectorIndex.of(catalogue(*(
        scenario(sid, (sid.split("-")[0],), "resolution", detector)
        for sid, detector in declared)))


def test_a_move_to_another_zone_is_in_the_pool_and_not_tagged(gen2_records, gen2_sidecars):
    record, api, zone = _zone_change_effect(gen2_records, gen2_sidecars)
    elsewhere = "command" if zone != "command" else "ante"
    index = _index(
        ("1.1-1", declaration("outcome_event", "resolution", event="zone_change", api=api,
                              to_zone=zone)),
        ("1.2-1", declaration("outcome_event", "resolution", event="zone_change", api=api,
                              to_zone=elsewhere)),
        ("1.3-1", declaration("outcome_event", "resolution", event="zone_change",
                              api=api + "Elsewhere")))
    found = breadth_counts(record, index, gen2_sidecars)
    assert found[("1.1-1", POOL)] == found[("1.1-1", TAGGED)] == 1
    assert found[("1.2-1", POOL)] == 1 and found[("1.2-1", TAGGED)] == 0
    assert not found[("1.3-1", POOL)]  # another rule's subject


def test_a_declaration_needing_a_gen3_field_is_skipped_on_gen2(gen2_records, gen2_sidecars):
    record, _, _ = _zone_change_effect(gen2_records, gen2_sidecars)
    index = _index(("1.1-1", declaration("outcome_event", "resolution", needs=("owner",),
                                         event="zone_change")))
    assert not breadth_counts(record, index, gen2_sidecars)


def test_rows_carry_every_declared_scenario_and_what_it_dropped():
    index = _index(
        ("1.1-1", declaration("outcome_event", "resolution", event="zone_change", api="X",
                              to_zone="exile")),
        ("1.2-1", declaration("sba_zone_move", "resolution", check="toughness_zero")))
    rows = {row.scenario: row for row in breadth_rows(
        index, Counter({("1.1-1", POOL): 400, ("1.1-1", TAGGED): 2}))}
    assert rows["1.1-1"].dropped == ("to_zone",) and rows["1.1-1"].share == pytest.approx(0.005)
    assert rows["1.2-1"].dropped == () and rows["1.2-1"].pool == 0


def test_a_small_share_of_a_large_pool_is_flagged_and_nothing_else():
    rows = [BreadthRow("a-1", ("a",), ("api",), 1000, 3),   # flagged
            BreadthRow("b-1", ("b",), ("api",), 1000, 50),  # a fair share
            BreadthRow("c-1", ("c",), ("api",), 50, 0),     # a small pool
            BreadthRow("d-1", ("d",), (), 1000, 3)]         # nothing dropped
    assert [row.scenario for row in flagged_rows(rows, min_pool=200, max_share=0.01)] == ["a-1"]
