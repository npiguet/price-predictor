"""``outcome_event`` on real gen-2 effect halves."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from effects.domain.event_schema import EventType
from effects.domain.records import ResolutionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.acting_line import acting_line, acting_root_params
from effects.domain.rule_detectors.outcome_event import (
    FAMILY,
    acting_api,
    source_is_token_of_kind,
    token_kind_of,
)

PREDEFINED = {"treasure", "food", "clue", "blood", "map", "gold", "powerstone", "shard",
              "incubator", "junk"}


def _token_maker(by_kind, sidecars):
    return next(r for r in by_kind("resolution")
                if r.moment.value == "resolution" and acting_api(r, sidecars)
                and any(e.type is EventType.TOKEN_CREATED for e in r.payload.events))


def test_a_resolution_creating_a_token_fires(by_kind, gen2_sidecars):
    record = _token_maker(by_kind, gen2_sidecars)
    found = FAMILY.match(record, gen2_sidecars, {"event": "token_created"})
    assert found.polarity is Polarity.FIRES and found.observed


def test_the_same_resolution_edited_to_create_nothing_is_the_near_miss_of_its_api(
        by_kind, gen2_sidecars):
    record = _token_maker(by_kind, gen2_sidecars)
    api = acting_api(record, gen2_sidecars)
    edited = replace(record, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
    found = FAMILY.match(edited, gen2_sidecars, {"event": "token_created", "api": api})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
    assert FAMILY.match(edited, gen2_sidecars, {"event": "token_created"}) is None


def test_another_api_shows_nothing(by_kind, gen2_sidecars):
    record = _token_maker(by_kind, gen2_sidecars)
    assert FAMILY.match(record, gen2_sidecars,
                        {"event": "token_created", "api": "NoSuchApi"}) is None


def test_zones_and_count_narrow_the_event(by_kind, gen2_sidecars):
    record = next(r for r in by_kind("resolution") if r.moment.value == "resolution"
                  and any(e.type is EventType.ZONE_CHANGE for e in r.payload.events))
    move = next(e for e in record.payload.events if e.type is EventType.ZONE_CHANGE)
    zones = {k: move.params[k] for k in ("from_zone", "to_zone") if k in move.params}
    assert FAMILY.match(record, None, {"event": "zone_change", **zones}).polarity is Polarity.FIRES
    assert FAMILY.match(record, None, {"event": "zone_change", "to_zone": "nowhere"}) is None
    assert FAMILY.match(record, None, {"event": "zone_change", "min_count": 999}) is None


def _loyalty_change(gen2_records):
    return next(r for r in gen2_records if r.kind.value == "resolution"
                and r.moment.value == "resolution" and any(
                    e.type is EventType.COUNTER_CHANGE
                    and str(e.params.get("counter_type")).upper() == "LOYALTY"
                    for e in r.payload.listed_events()))


class TestEventParameters:
    def test_a_parameter_of_the_event_narrows_it_without_case(self, gen2_records):
        record = _loyalty_change(gen2_records)
        declared = {"event": "counter_change", "counter_type": "loyalty"}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES

    def test_another_value_shows_nothing(self, gen2_records):
        record = _loyalty_change(gen2_records)
        assert FAMILY.match(record, None, {"event": "counter_change",
                                           "counter_type": "P1P1"}) is None

    def test_a_parameter_the_event_does_not_carry_is_refused(self, gen2_records):
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.match(gen2_records[0], None, {"event": "zone_change", "counter_type": "x"})
        with pytest.raises(ValueError, match="no parameter"):
            FAMILY.match(gen2_records[0], None, {"event": "zone_change", "cause": "E1"})


class TestAbsent:
    def test_the_half_without_the_event_is_the_fires_side(self, by_kind, gen2_sidecars):
        record = _token_maker(by_kind, gen2_sidecars)
        api = acting_api(record, gen2_sidecars)
        declared = {"event": "token_created", "api": api, "absent": True}
        found = FAMILY.match(record, gen2_sidecars, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        edited = replace(record, payload=ResolutionPayload(events=tuple(
            e for e in record.payload.events if e.type is not EventType.TOKEN_CREATED)))
        found = FAMILY.match(edited, gen2_sidecars, declared)
        assert found.polarity is Polarity.FIRES and found.observed

    def test_absent_needs_an_api(self, gen2_records):
        with pytest.raises(ValueError, match="needs api"):
            FAMILY.match(gen2_records[0], None, {"event": "token_created", "absent": True})


class TestTokenKinds:
    def test_a_predefined_token_made_shows_by_its_kind(self, gen2_records):
        record, kind = next(
            (r, sorted(token_kind_of(e) & PREDEFINED)[0])
            for r in gen2_records if r.kind.value == "resolution"
            for e in r.payload.listed_events()
            if e.type is EventType.TOKEN_CREATED and token_kind_of(e) & PREDEFINED)
        assert FAMILY.match(record, None, {"event": "token_created",
                                           "token_kind": kind.title()}).observed
        assert FAMILY.match(record, None, {"event": "token_created",
                                           "token_kind": "Nonesuch"}) is None

    def test_token_kind_reads_token_created_only(self, gen2_records):
        with pytest.raises(ValueError, match="token_kind"):
            FAMILY.match(gen2_records[0], None, {"event": "zone_change", "token_kind": "Food"})

    def test_an_ability_of_a_predefined_token_shows_by_its_source(self, gen2_records):
        record, kind = next(
            (r, sorted({s.lower() for s in source.subtypes} & PREDEFINED)[0])
            for r in gen2_records if r.kind.value == "resolution"
            and r.moment.value == "resolution"
            for source in r.state.entities
            if source.id == r.state.refs.source and source.token_script_id
            and {s.lower() for s in source.subtypes} & PREDEFINED)
        event = next(e for e in record.payload.listed_events()).type.value
        assert source_is_token_of_kind(record.state, kind)
        assert FAMILY.match(record, None, {"event": event, "source_token_kind": kind}) is not None
        assert FAMILY.match(record, None, {"event": event,
                                           "source_token_kind": "nonesuch"}) is None


def _effect_halves(by_kind):
    return [r for r in by_kind("resolution") if r.moment.value == "resolution"]


class TestKeyword:
    @staticmethod
    def _keyword_resolution(by_kind, sidecars):
        return next(r for r in _effect_halves(by_kind) if r.payload.events
                    and acting_line(r, sidecars) is not None
                    and acting_line(r, sidecars).keyword_display_name)

    def test_an_ability_a_keyword_generates_shows_by_its_keyword(self, by_kind, gen2_sidecars):
        record = self._keyword_resolution(by_kind, gen2_sidecars)
        name = acting_line(record, gen2_sidecars).keyword_display_name
        event = record.payload.events[0].type.value
        found = FAMILY.match(record, gen2_sidecars, {"event": event, "keyword": name})
        assert found.polarity is Polarity.FIRES and found.observed
        spelled = name.lower().replace(" ", "_")
        assert FAMILY.match(record, gen2_sidecars, {"event": event, "keyword": spelled})
        assert FAMILY.match(record, gen2_sidecars,
                            {"event": event, "keyword": "no such keyword"}) is None

    def test_the_keyword_scopes_the_near_miss_as_api_does(self, by_kind, gen2_sidecars):
        record = self._keyword_resolution(by_kind, gen2_sidecars)
        name = acting_line(record, gen2_sidecars).keyword_display_name
        event = record.payload.events[0].type
        edited = replace(record, payload=ResolutionPayload(events=tuple(
            e for e in record.payload.events if e.type is not event)))
        found = FAMILY.match(edited, gen2_sidecars, {"event": event.value, "keyword": name})
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed
        absent = FAMILY.match(edited, gen2_sidecars,
                              {"event": event.value, "keyword": name, "absent": True})
        assert absent.polarity is Polarity.FIRES

    def test_a_script_line_is_no_keyword_line(self, by_kind, gen2_sidecars):
        record = _token_maker(by_kind, gen2_sidecars)
        assert FAMILY.match(record, gen2_sidecars,
                            {"event": "token_created", "keyword": "Equip"}) is None


class TestScriptParam:
    @staticmethod
    def _with_param(by_kind, sidecars):
        for record in _effect_halves(by_kind):
            found = acting_root_params(record, sidecars) or {}
            name = next((k for k in found if k not in ("SP", "AB", "DB") and found[k]), None)
            if record.payload.events and name:
                return record, name, found[name], record.payload.events[0].type.value
        raise AssertionError("no effect half acting through a script with a parameter")

    def test_a_parameter_present_or_holding_a_value_narrows_the_line(
            self, by_kind, gen2_sidecars):
        record, name, value, event = self._with_param(by_kind, gen2_sidecars)
        for spec in (name, {name: "*"}, {name: value.upper()}):
            assert FAMILY.match(record, gen2_sidecars,
                                {"event": event, "script_param": spec}).observed
        for spec in ("NoSuchParam", {name: value + "-not"}):
            assert FAMILY.match(record, gen2_sidecars,
                                {"event": event, "script_param": spec}) is None

    def test_a_malformed_parameter_is_refused(self, by_kind, gen2_sidecars):
        record, _, _, event = self._with_param(by_kind, gen2_sidecars)
        with pytest.raises(ValueError, match="script_param"):
            FAMILY.match(record, gen2_sidecars, {"event": event, "script_param": ["a"]})


def _in_combat(record, subject):
    entity = record.state.entity(subject)
    return entity is not None and entity.combat is not None and bool(
        entity.combat.attacking or entity.combat.blocking)


class TestQualifiers:
    def test_a_creature_leaving_combat_fires_and_one_out_of_combat_is_the_near_miss(
            self, by_kind):
        record = next(r for r in _effect_halves(by_kind) for e in r.payload.events
                      if e.type is EventType.ZONE_CHANGE
                      and e.params.get("from_zone") == "battlefield"
                      and any(_in_combat(r, s) for s in e.subjects))
        declared = {"event": "zone_change", "from_zone": "battlefield",
                    "subject_in_combat": True}
        found = FAMILY.match(record, None, declared)
        assert found.polarity is Polarity.FIRES and found.observed
        assert any(_in_combat(record, s) for s in found.entities)
        out_of_combat = replace(record, state=replace(record.state, entities=tuple(
            replace(e, combat=None) for e in record.state.entities)))
        found = FAMILY.match(out_of_combat, None, declared)
        assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed

    def test_a_token_subject_reads_off_the_snapshot(self, by_kind):
        no_token = SimpleNamespace(token_script_id=None)
        record, event = next(
            (r, e) for r in _effect_halves(by_kind) for e in r.payload.events
            if any((r.state.entity(s) or no_token).token_script_id for s in e.subjects))
        declared = {"event": event.type.value, "subject_is_token": True}
        assert FAMILY.match(record, None, declared).polarity is Polarity.FIRES
        carded = replace(record, state=replace(record.state, entities=tuple(
            replace(e, token_script_id=None) for e in record.state.entities)))
        assert FAMILY.match(carded, None, declared).polarity is Polarity.DOES_NOT_FIRE

    def test_another_event_in_the_record_is_with_or_without(self, by_kind):
        record = next(r for r in _effect_halves(by_kind)
                      if len({e.type for e in r.payload.events}) >= 2)
        first, other = list(dict.fromkeys(e.type.value for e in record.payload.events))[:2]
        missing = next(t.value for t in EventType
                       if t not in {e.type for e in record.payload.events})
        sides = {
            ("with_event", other): Polarity.FIRES,
            ("without_event", other): Polarity.DOES_NOT_FIRE,
            ("with_event", missing): Polarity.DOES_NOT_FIRE,
            ("without_event", missing): Polarity.FIRES,
        }
        for (name, value), polarity in sides.items():
            found = FAMILY.match(record, None, {"event": first, name: value})
            assert found.polarity is polarity, (name, value)

    def test_controller_is_owner_needs_the_gen3_owner(self):
        assert FAMILY.needs({"event": "zone_change", "controller_is_owner": False}) == {"owner"}
        assert FAMILY.needs({"event": "zone_change"}) == frozenset()

    def test_a_qualifier_and_absent_are_refused_together(self, gen2_records):
        with pytest.raises(ValueError, match="qualifier"):
            FAMILY.match(gen2_records[0], None, {"event": "zone_change", "api": "Draw",
                                                 "absent": True, "subject_is_token": True})


def test_a_combat_records_events_show_too(by_kind):
    record = next(r for r in by_kind("combat") if not r.fork and any(
        e.type is EventType.DAMAGE_DEALT for e in r.payload.events))
    assert FAMILY.match(record, None, {"event": "damage_dealt"}).polarity is Polarity.FIRES
    assert FAMILY.match(record, None, {"event": "damage_dealt", "api": "DealDamage"}) is None
