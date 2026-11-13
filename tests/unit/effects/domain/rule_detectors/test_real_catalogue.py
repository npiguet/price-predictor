"""The detectors over the whole real catalogue under ``rules/scenarios`` (TP4's missing test).

Every other detector test passes hand-written parameters to one family; these
build the index the commands build, from every scenario the catalogue holds.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

from effects.domain.records import RecordKind
from effects.domain.rule_detectors import DetectorIndex
from effects.domain.rule_detectors.emittable_events import EMITTABLE, carries
from effects.infrastructure.rules_catalogue_io import load_scenarios

RULES = Path(__file__).parents[5] / "rules"

#: The declarations no record can show (V5 #14), each reported, none tagging:
#: an event no collector writes on the kind they ask.
UNTAGGABLE = {
    # game_drawn: no collector writes it
    "104.4a-1", "104.4c-1",
    # removed_from_combat on a combat record: a clause hook's alone
    "701.8c-2",
}


@pytest.fixture(scope="module")
def catalogue():
    return load_scenarios(RULES)


@pytest.fixture(scope="module")
def index(catalogue):
    return DetectorIndex.of(catalogue)


def test_every_declaration_of_the_real_catalogue_is_readable(index):
    """``DetectorIndex.of`` refuses an undeclared parameter or a value its
    family cannot read (V5 #22); building it over the real tree is the test."""
    assert sum(len(found) for found in index.by_kind.values()) > 700


def test_exactly_the_known_declarations_are_untaggable(index):
    assert set(index.untaggable) == UNTAGGABLE
    for sid, reason in index.untaggable.items():
        assert "never lists" in reason or "game-end record" in reason, sid


def test_no_taggable_declaration_asks_its_kind_for_an_event_it_never_lists(index):
    for kind, found in index.by_kind.items():
        for scenario, declaration, family in found:
            for event in getattr(family, "events", lambda _: ())(declaration.params):
                assert carries(kind, event), (scenario.id, event)


def test_every_record_kind_a_declaration_names_has_a_row(index, catalogue):
    kinds = {s.detector.record_kind for s in catalogue.scenarios
             if hasattr(s.detector, "record_kind")}
    assert {RecordKind(k) for k in kinds} <= set(EMITTABLE)


def test_no_two_rules_share_an_identical_declaration(catalogue):
    """D4: scenarios of different rules declaring the same family, record kind
    and parameters tag one population, so the rules are indistinguishable;
    a scenario stating another's fact declares itself ``same_as`` it (exempt
    here), and what the record cannot show declares ``none``."""
    from effects.domain.rule_catalogue import DetectorDeclaration

    groups = defaultdict(set)
    for scenario in catalogue.scenarios:
        declaration = scenario.detector
        if isinstance(declaration, DetectorDeclaration):
            key = (declaration.family, declaration.record_kind,
                   json.dumps(declaration.params, sort_keys=True))
            groups[key].add(scenario.rules)
    shared = {key: sorted(rules) for key, rules in groups.items() if len(rules) > 1}
    assert not shared, shared


def test_every_same_as_scenario_points_at_declaring_scenarios(index, catalogue):
    """Building the index checks the targets; each duplicate is reported under
    rules of its own targets, never its own."""
    for sid, duplicate in index.duplicates.items():
        assert duplicate.counted_under, sid
        assert not set(duplicate.rules) & set(duplicate.counted_under), sid
