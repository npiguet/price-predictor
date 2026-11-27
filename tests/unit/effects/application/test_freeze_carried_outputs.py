"""A rule test of a per-entity rule counts a row only where the record carries
one of the rule's effective outputs at the row's outcome entities
(contracts/selection-set.md, Freeze step 4).

The control score drops a request for an output the record defines no target
for (``untargeted``), so the freeze does not count such a row. The records are
the real gen-3 fixture's, their games edited into the two halves.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.application.freeze_selection_set import (
    FreezePorts,
    RuleTestRules,
    SelectionSetRefused,
    freeze_selection_set,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import RuleTag
from effects.domain.rule_output_carriage import carried_outputs
from effects.domain.selection_set import NO_OUTCOME_ENTITY
from tests.unit.effects.application.test_freeze_selection_set import (
    GD,
    LENIENT,
    _Grid,
    _Losses,
    _manifest,
    _Probes,
    _reader,
)

FIRES, NOT = Polarity.FIRES, Polarity.DOES_NOT_FIRE
RULE = "704.5m"
OUTPUTS = ("zone_outcome",)
HALVES = {GD: {"selection": ["s"], "reporting": ["r"]}}


def _eligible():
    from tests.unit.effects.builders import curated_fixture_records

    return [r for r in curated_fixture_records()
            if not (r.fork or r.interventional or r.synthetic or r.aimed_at is not None
                    or r.directed is not None)]


def _carrier(record):
    """An entity of ``record`` whose zone outcome it carries, or None."""
    return next((e.id for e in record.state.entities
                 if carried_outputs(record, [e.id], OUTPUTS)), None)


@pytest.fixture(scope="module")
def carrying():
    found = [(r, _carrier(r)) for r in _eligible()]
    return [(r, e) for r, e in found if e is not None]


@pytest.fixture(scope="module")
def reading_nothing():
    return [r for r in _eligible()
            if not carried_outputs(r, [r.actor_player], OUTPUTS)]


def _rules(tags):
    return RuleTestRules(tag=lambda record: tags.get(record.record_id, []),
                         declared=frozenset({RULE}), trained=frozenset({RULE}),
                         entity_read={RULE: OUTPUTS})


def _tag(polarity, outcome):
    return RuleTag(RULE, polarity, observed=polarity is FIRES, condition=(), outcome=outcome,
                   scenario=f"{RULE}-1")


def test_a_row_naming_a_player_for_a_zone_output_is_not_counted(carrying, reading_nothing):
    record, carrier = carrying[0]
    tags = {record.record_id: [_tag(FIRES, (record.actor_player,))]}
    assert _rules(tags).tests_of(record) == [(RULE, FIRES, ())]
    tags = {record.record_id: [_tag(FIRES, (carrier,))]}
    assert _rules(tags).tests_of(record) == [(RULE, FIRES, (carrier,))]


def test_a_polarity_with_no_row_reading_an_output_leaves_the_rule_out(carrying, reading_nothing):
    """``fires`` rows carry the output; every ``does_not_fire`` row names an
    entity the record defines no zone outcome for."""
    assert len(carrying) >= 2 and len(reading_nothing) >= 2
    records, tags = [], {}
    for half in ("s", "r"):
        for record, carrier in carrying:
            copy = replace(record, record_id=f"{record.record_id}~{half}", game_id=half)
            records.append(copy)
            tags[copy.record_id] = [_tag(FIRES, (carrier,))]
        for record in reading_nothing[:4]:
            copy = replace(record, record_id=f"{record.record_id}~{half}", game_id=half)
            records.append(copy)
            tags[copy.record_id] = [_tag(NOT, (record.actor_player,))]
    with pytest.raises(SelectionSetRefused) as refused:
        freeze_selection_set(_manifest(HALVES), {GD: _reader(records)},
                             ports=FreezePorts(rules=_rules(tags), probes=_Probes(),
                                               grids=[_Grid()], deviance=_Losses()),
                             thresholds=LENIENT)
    for half in ("selection", "reporting"):
        left = refused.value.selection.left_out[GD][half]["rule_tests"]
        assert [item.item for item in left] == [RULE]
        assert left[0].reason.startswith(NO_OUTCOME_ENTITY)
        assert "0 of 4 does_not_fire name one" in left[0].reason
