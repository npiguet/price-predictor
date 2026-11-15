"""The detector framework: registry, per-rule merging, not-countable rules and review (T024)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from effects.domain.rule_catalogue import NoDetector, Polarity, SameAs
from effects.domain.rule_detectors import (
    DeclarationError,
    DetectorIndex,
    Duplicate,
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


def test_two_scenarios_of_one_rule_merge_to_fires_with_each_role_unioned():
    index = _index(
        ("704.5f-1", ("704.5f",),
         _Fixed(FamilyMatch(Polarity.DOES_NOT_FIRE, False, ("E1",), ("E3",)))),
        ("704.5f-2", ("704.5f",), _Fixed(FamilyMatch(Polarity.FIRES, True, ("E2",), ("E1",)))),
    )
    (tag,) = tag_record(make_record(), index).tags
    assert (tag.rule, tag.polarity, tag.observed) == ("704.5f", Polarity.FIRES, True)
    assert tag.condition == ("E1", "E2")
    assert tag.outcome == ("E3", "E1")
    assert tag.entities == ("E1", "E2", "E3")
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
    gen2 = tag_record(make_record(generation=None), index)
    assert gen2.tags == () and gen2.not_countable == {"601.3"}
    gen3 = tag_record(make_record(generation=3), index)
    assert [t.rule for t in gen3.tags] == ["601.3"] and not gen3.not_countable


def test_a_declaration_s_own_needs_also_make_a_rule_not_countable():
    family = _Fixed(FamilyMatch(Polarity.FIRES, True))
    decl = declaration(family.id, "resolution", needs=("timing",))
    index = DetectorIndex({"resolution": ((scenario("x-1", ("1.1",), "resolution", decl),
                                           decl, family),)})
    assert tag_record(make_record(generation=None), index).not_countable == {"1.1"}


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
    assert RuleTag("r", Polarity.FIRES, True, (), (), "s").needs_review is False
    assert RuleTag("r", Polarity.FIRES, False, (), (), "s").needs_review is True
    assert RuleTag("r", Polarity.OTHER, True, (), (), "s").needs_review is False


def test_an_outcome_the_record_cannot_show_goes_on_no_review_list():
    """A family answering ``observed=None`` (the outcome is filed in another
    record) counts the record's side and disagrees with nothing."""
    index = _index(("704.5a-1", ("704.5a",), _Fixed(FamilyMatch(Polarity.FIRES, None, ("P1",)))))
    result = tag_record(make_record(), index)
    (tag,) = result.tags
    assert tag.polarity is Polarity.FIRES and tag.observed is None
    assert not result.review


def test_a_readable_outcome_beside_an_unreadable_one_decides_the_merge():
    index = _index(
        ("704.5a-1", ("704.5a",), _Fixed(FamilyMatch(Polarity.FIRES, None))),
        ("704.5a-2", ("704.5a",), _Fixed(FamilyMatch(Polarity.FIRES, True))))
    (tag,) = tag_record(make_record(), index).tags
    assert tag.observed is True


def test_only_the_record_s_own_kind_is_matched():
    index = _index(("1-1", ("1.1",), _Fixed(FamilyMatch(Polarity.FIRES, True))))
    from effects.domain.records import RecordKind

    assert tag_record(make_record(RecordKind.COMBAT), index).tags == ()


def test_an_unregistered_family_is_refused_when_the_index_is_built():
    unknown = scenario("1-1", ("1.1",), "resolution", declaration("no_such_family", "resolution"))
    with pytest.raises(DeclarationError, match="no_such_family"):
        DetectorIndex.of(catalogue(unknown))


@pytest.mark.parametrize(("family", "params", "message"), [
    ("sba_zone_move", {"check": "lethal_damage", "chek": "x"}, "no parameter"),
    ("sba_zone_move", {"check": "lethal_damag"}, "unknown check"),
    ("verdict_bit", {"bit": "timming"}, "unknown bit"),
    ("legality_membership", {"list": "legal_blocker"}, "unknown list"),
    ("outcome_event", {"event": "zone_changed"}, "zone_changed"),
    ("outcome_event", {"event": "zone_change", "to": "graveyard"}, "no parameter"),
    ("trigger_mode", {"keyword": "Prowess", "mode": "SpellCast"}, "keyword"),
    ("resolution_outcome", {"outcome": "fizzle"}, "fizzle"),
])
def test_a_misspelt_declaration_is_refused_when_the_index_is_built(family, params, message):
    """V5 #22: a parameter name the family does not declare, or a value it
    cannot read, stops the command at the index, naming the scenario, rather
    than tagging nothing or raising mid-run."""
    bad = scenario("9.9-1", ("9.9",), "resolution",
                   declaration(family, "resolution", **params))
    with pytest.raises(DeclarationError, match=f"9.9-1.*{message}"):
        DetectorIndex.of(catalogue(bad))


def test_a_declaration_asking_its_kind_for_an_event_it_never_lists_is_untaggable():
    """V5 #14: a combat never lists ``removed_from_combat`` (a clause hook's
    event), so a combat record must not read as the rule's near miss."""
    asks = scenario("701.8c-2", ("701.8c",), "combat", declaration(
        "outcome_event", "combat", event="regenerated", with_event="removed_from_combat"))
    index = DetectorIndex.of(catalogue(asks))
    assert index.by_kind == {} and "701.8c" not in index.declared_rules
    assert "removed_from_combat" in index.untaggable["701.8c-2"]
    assert index.untaggable_rules["701.8c-2"] == ("701.8c",)
    on_a_resolution = scenario("1-1", ("1.1",), "resolution", declaration(
        "outcome_event", "resolution", event="face_change", api="SetState"))
    assert DetectorIndex.of(catalogue(on_a_resolution)).untaggable == {}


def test_the_game_s_end_asked_of_a_resolution_acting_through_a_line_is_untaggable():
    """The game's end is filed in the game-end record, which acts through no
    line: a Draw resolution would read as 121.4's near miss every time."""
    scoped = scenario("121.4-1", ("121.4",), "resolution", declaration(
        "outcome_event", "resolution", event="player_lost", api="Draw"))
    assert "game-end record" in DetectorIndex.of(catalogue(scoped)).untaggable["121.4-1"]
    unscoped = scenario("704.5b-1", ("704.5b",), "resolution", declaration(
        "outcome_event", "resolution", event="player_lost", reason="Milled"))
    assert DetectorIndex.of(catalogue(unscoped)).untaggable == {}


def test_a_same_as_scenario_tags_nothing_and_points_at_its_scenario():
    flying = scenario("702.9b-1", ("702.9b",), "playability", declaration(
        "legality_membership", "playability", list="legal_blockers"))
    again = scenario("702.17b-1", ("702.17b",), "playability",
                     SameAs(("702.9b-1",), "both say a flier is blocked only by flying or reach"))
    index = DetectorIndex.of(catalogue(flying, again))
    assert index.declared_rules == {"702.9b"}
    assert index.duplicates["702.17b-1"] == Duplicate(("702.17b",), ("702.9b-1",), ("702.9b",))
    union = scenario("1-1", ("1.1",), "playability", SameAs(("702.9b-1", "702.17b-1"), "x"))
    with pytest.raises(DeclarationError, match="702.17b-1"):
        DetectorIndex.of(catalogue(flying, again, union))
    dangling = scenario("1-1", ("1.1",), "playability", SameAs(("2-1",), "x"))
    with pytest.raises(DeclarationError, match="2-1"):
        DetectorIndex.of(catalogue(dangling))


def test_a_same_as_naming_a_scenario_of_its_own_rule_is_refused():
    """A duplicate is counted under its targets' rules, never its own: two
    scenarios of one rule stating one fact each declare it."""
    first = scenario("702.103e-1", ("702.103e",), "resolution", declaration(
        "legality_membership", "playability", list="legal_blockers"))
    second = scenario("702.103e-2", ("702.103e",), "resolution",
                      SameAs(("702.103e-1",), "the same fact"))
    with pytest.raises(DeclarationError, match="702.103e-2.*own rule"):
        DetectorIndex.of(catalogue(first, second))


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


def test_the_game_end_marker_is_tagged_by_no_detector():
    """The marker holds no event and no board change of its own (WP7): no
    declaration of its kind, an absence's included, may read it."""
    from effects.domain.records import (
        RecordKind,
        TurnActionPayload,
        TurnActionSubkind,
    )

    family = _Fixed(FamilyMatch(Polarity.FIRES, True))
    entry = (scenario("t-1", ("100.1",), "turn_action", declaration(family.id, "turn_action")),
             declaration(family.id, "turn_action"), family)
    index = DetectorIndex({"turn_action": (entry,)}, frozenset({"100.1"}))
    draw = make_record(RecordKind.TURN_ACTION, subkind=TurnActionSubkind.DRAW,
                       ability=None, payload=TurnActionPayload(), generation=3)
    marker = make_record(RecordKind.TURN_ACTION, subkind=TurnActionSubkind.GAME_END,
                         ability=None, generation=3,
                         payload=TurnActionPayload(records_withheld=0))
    assert [t.rule for t in tag_record(draw, index).tags] == ["100.1"]
    assert tag_record(marker, index) == tag_record(marker, DetectorIndex({}))
    assert tag_record(marker, index).tags == ()
