"""``outcome_event`` on real gen-2 effect halves."""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import EventType
from effects.domain.records import ResolutionPayload
from effects.domain.rule_catalogue import Polarity
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
