"""A family states the record kinds and events it reads; trigger zones compare as sets.

``rewrite_event`` reads rewrite records, so a declaration of it on a
resolution reads nothing; a script's ``Origin$ Any`` or comma list of
destinations holds a declared zone.
"""

from __future__ import annotations

import pytest

from effects.domain.rule_detectors import DeclarationError, DetectorIndex, families
from effects.domain.rule_detectors import trigger_mode as trigger_module
from effects.domain.rule_detectors.emittable_events import carries
from effects.domain.rule_detectors.trigger_mode import FAMILY
from tests.unit.effects.domain.rule_detectors.builders import catalogue, declaration, scenario


def test_a_rewrite_declaration_on_a_resolution_is_refused():
    bad = scenario("603.6d-1", ("603.6d",), "resolution",
                   declaration("rewrite_event", "resolution", event="zone_change"))
    with pytest.raises(DeclarationError, match=r"603.6d-1.*rewrite_event.*rewrite"):
        DetectorIndex.of(catalogue(bad))


@pytest.mark.parametrize("family", sorted(families()))
def test_every_family_states_what_it_reads(family):
    found = families()[family]
    assert found.kinds and all(isinstance(kind, str) for kind in found.kinds)
    assert callable(found.events) and callable(found.reads_acting_line)


@pytest.mark.parametrize("script", [
    {"Mode": "ChangesZone", "Origin": "Battlefield", "Destination": "Graveyard"},
    {"Mode": "ChangesZone", "Origin": "Any", "Destination": "Graveyard"},
    {"Mode": "ChangesZone", "Origin": "battlefield", "Destination": "Graveyard,Exile"},
    {"Mode": "ChangesZone", "Origin": "Battlefield,Hand", "Destination": "Exile, Graveyard"},
])
def test_a_dies_declaration_matches_a_script_that_holds_its_zones(by_kind, monkeypatch, script):
    record = next(r for r in by_kind("trigger") if r.payload.fired)
    monkeypatch.setattr(trigger_module, "acting_root_params", lambda record, lines: script)
    found = FAMILY.match(record, None, {"mode": "ChangesZone", "origin": "Battlefield",
                                        "destination": "Graveyard"})
    assert found is not None and found.observed


@pytest.mark.parametrize("script", [
    {"Mode": "ChangesZone", "Origin": "Hand", "Destination": "Graveyard"},
    {"Mode": "ChangesZone", "Origin": "Battlefield", "Destination": "Exile,Hand"},
    {"Mode": "ChangesZone", "Destination": "Graveyard"},
])
def test_a_dies_declaration_leaves_a_script_without_its_zones(by_kind, monkeypatch, script):
    record = next(r for r in by_kind("trigger") if r.payload.fired)
    monkeypatch.setattr(trigger_module, "acting_root_params", lambda record, lines: script)
    assert FAMILY.match(record, None, {"mode": "ChangesZone", "origin": "Battlefield",
                                       "destination": "Graveyard"}) is None


def test_a_combat_record_carries_a_replacement_s_application():
    assert carries("combat", "replacement_applied")


@pytest.mark.parametrize("event", ["phased", "day_night_changed", "untapped", "card_drawn",
                                   "card_discarded", "zone_change", "counter_change"])
def test_a_turn_action_record_lists_what_its_six_actions_make(event):
    assert carries("turn_action", event)


@pytest.mark.parametrize("event", ["damage_dealt", "life_change", "token_created", "tapped"])
def test_a_turn_action_record_lists_nothing_else(event):
    assert not carries("turn_action", event)


@pytest.mark.parametrize(("script", "declared"), [
    ({"Origin": "Any", "Destination": "Battlefield"}, {"origin": "Battlefield"}),
    ({"Origin": "Any", "Destination": "Battlefield"}, {"origin": "AttractionDeck",
                                                      "destination": "Battlefield"}),
])
def test_any_holds_no_zone_the_move_cannot_come_from_nor_a_rare_zone(by_kind, monkeypatch,
                                                                    script, declared):
    record = next(r for r in by_kind("trigger") if r.payload.fired)
    script = {"Mode": "ChangesZone", **script}
    monkeypatch.setattr(trigger_module, "acting_root_params", lambda record, lines: script)
    assert FAMILY.match(record, None, {"mode": "ChangesZone", **declared}) is None
