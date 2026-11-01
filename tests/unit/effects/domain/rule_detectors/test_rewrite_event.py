"""``rewrite_event`` on real gen-2 rewrite records (T029)."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.records import RewriteResult
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.rewrite_event import FAMILY


def _entering(by_kind):
    return next(r for r in by_kind("rewrite")
                if r.payload.incoming.type.value == "zone_change"
                and r.payload.result is RewriteResult.UPDATED)


def test_an_updated_zone_change_fires_an_enters_replacement_rule(by_kind):
    params = {"event": "zone_change", "result": "updated", "to_zone": "battlefield"}
    found = FAMILY.match(_entering(by_kind), None, params)
    assert found.polarity is Polarity.FIRES and found.observed


def test_the_same_record_edited_to_not_replaced_is_the_near_miss(by_kind):
    record = _entering(by_kind)
    edited = replace(record, payload=replace(record.payload, result=RewriteResult.NOT_REPLACED,
                                             outgoing=None, replaced_by=()))
    found = FAMILY.match(edited, None, {"event": "zone_change", "result": "updated"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_another_event_or_destination_shows_nothing(by_kind):
    record = _entering(by_kind)
    assert FAMILY.match(record, None, {"event": "damage_dealt"}) is None
    assert FAMILY.match(record, None, {"event": "zone_change", "to_zone": "exile"}) is None
