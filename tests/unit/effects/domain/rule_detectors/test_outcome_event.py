"""``outcome_event`` on real gen-2 effect halves."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.event_schema import EventType
from effects.domain.records import ResolutionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.outcome_event import FAMILY, acting_api


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
