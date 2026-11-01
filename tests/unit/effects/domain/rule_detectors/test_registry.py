"""The detector framework: registry, per-rule merging, not-countable rules and review (T024)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from effects.domain.rule_catalogue import NoDetector, Polarity
from effects.domain.rule_detectors import (
    DetectorIndex,
    FamilyMatch,
    RuleTag,
    families,
    tag_record,
)
from tests.unit.effects.builders import make_record
from tests.unit.effects.domain.rule_detectors.builders import catalogue, declaration, scenario


@dataclass(frozen=True)
class _Fixed:
    """A family that answers the same for every record, so the merge is what is tested."""

    answer: FamilyMatch | None
    id: str = "fixed"
    params: frozenset = frozenset()
    gen3_fields: frozenset = frozenset()

    def needs(self, params):
        return self.gen3_fields

    def match(self, record, lines, params):
        return self.answer


def _index(*entries):
    """``(scenario id, rules, family)`` triples, as one index of resolution scenarios."""
    by_kind = tuple(
        (scenario(sid, rules, "resolution", declaration(family.id, "resolution")),
         declaration(family.id, "resolution"), family)
        for sid, rules, family in entries)
    return DetectorIndex({"resolution": by_kind},
                         frozenset(r for _, rules, _ in entries for r in rules))


def test_two_scenarios_of_one_rule_merge_to_fires_with_every_entity():
    index = _index(
        ("704.5f-1", ("704.5f",), _Fixed(FamilyMatch(Polarity.DOES_NOT_FIRE, False, ("E1",)))),
        ("704.5f-2", ("704.5f",), _Fixed(FamilyMatch(Polarity.FIRES, True, ("E2",)))),
    )
    (tag,) = tag_record(make_record(), index).tags
    assert (tag.rule, tag.polarity, tag.observed) == ("704.5f", Polarity.FIRES, True)
    assert tag.entities == ("E1", "E2")
    assert tag.scenario == "704.5f-2"


def test_does_not_fire_wins_over_other():
    index = _index(
        ("1-1", ("100.1",), _Fixed(FamilyMatch(Polarity.OTHER, False))),
        ("1-2", ("100.1",), _Fixed(FamilyMatch(Polarity.DOES_NOT_FIRE, False))),
    )
    assert tag_record(make_record(), index).tags[0].polarity is Polarity.DOES_NOT_FIRE


def test_a_rule_needing_a_gen3_field_is_not_countable_on_gen2_and_counted_on_gen3():
    """Story 1 acceptance 4: not countable, never zero (FR-007)."""
    family = _Fixed(FamilyMatch(Polarity.FIRES, True), gen3_fields=frozenset({"failed_checks"}))
    index = _index(("601.3-1", ("601.3",), family))
    gen2 = tag_record(make_record(), index)
    assert gen2.tags == () and gen2.not_countable == {"601.3"}
    gen3 = tag_record(make_record(generation=3), index)
    assert [t.rule for t in gen3.tags] == ["601.3"] and not gen3.not_countable


def test_a_declaration_s_own_needs_also_make_a_rule_not_countable():
    family = _Fixed(FamilyMatch(Polarity.FIRES, True))
    decl = declaration(family.id, "resolution", needs=("timing",))
    index = DetectorIndex({"resolution": ((scenario("x-1", ("1.1",), "resolution", decl),
                                           decl, family),)})
    assert tag_record(make_record(), index).not_countable == {"1.1"}


def test_a_scenario_declaring_no_detector_tags_nothing():
    """Story 1 acceptance 3."""
    none = scenario("613.1-1", ("613.1",), "continuous", NoDetector("layer order is not recorded"))
    index = DetectorIndex.of(catalogue(none))
    assert index.by_kind == {}
    assert tag_record(make_record(), index).tags == ()


def test_a_record_whose_outcome_disagrees_with_its_side_goes_on_the_review_list():
    """Story 1 acceptance 5: ``observed != (polarity == fires)``."""
    index = _index(("704.5f-1", ("704.5f",), _Fixed(FamilyMatch(Polarity.DOES_NOT_FIRE, True))))
    result = tag_record(make_record(), index)
    assert [t.scenario for t in result.review] == ["704.5f-1"]
    assert RuleTag("r", Polarity.FIRES, True, (), "s").needs_review is False
    assert RuleTag("r", Polarity.FIRES, False, (), "s").needs_review is True
    assert RuleTag("r", Polarity.OTHER, True, (), "s").needs_review is False


def test_only_the_record_s_own_kind_is_matched():
    index = _index(("1-1", ("1.1",), _Fixed(FamilyMatch(Polarity.FIRES, True))))
    from effects.domain.records import RecordKind

    assert tag_record(make_record(RecordKind.COMBAT), index).tags == ()


def test_an_unregistered_family_is_refused_when_the_index_is_built():
    unknown = scenario("1-1", ("1.1",), "resolution", declaration("no_such_family", "resolution"))
    with pytest.raises(KeyError, match="no_such_family"):
        DetectorIndex.of(catalogue(unknown))


_SCENARIO_SPECIFIC = re.compile(r"card|name|cards|names|entity|object", re.IGNORECASE)


@pytest.mark.parametrize("family", sorted(families().values(), key=lambda f: f.id),
                         ids=lambda f: f.id)
def test_no_family_takes_a_scenario_specific_parameter(family):
    """A detector states the rule's condition; a card name would make it match
    one scenario's board and almost nothing in the corpus."""
    assert not [p for p in family.params if _SCENARIO_SPECIFIC.search(p)]


def test_every_family_module_registers_its_family():
    from effects.domain.rule_detectors import FAMILY_MODULES

    assert len(families()) == len(FAMILY_MODULES)
