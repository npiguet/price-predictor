"""``continuous_contribution`` on real gen-2 continuous records (T029)."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.records import ContinuousPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.continuous_contribution import FAMILY


def _trample_grant(by_kind):
    return next(r for r in by_kind("continuous")
                if any("trample" in c.keywords for c in r.payload.contributions))


def test_a_static_granting_the_keyword_fires(by_kind):
    found = FAMILY.match(_trample_grant(by_kind), None, {"field": "keywords", "value": "trample"})
    assert found.polarity is Polarity.FIRES and found.entities


def test_the_same_static_edited_to_grant_nothing_is_the_near_miss(by_kind):
    record = _trample_grant(by_kind)
    stripped = ContinuousPayload(contributions=tuple(
        replace(c, keywords=()) for c in record.payload.contributions))
    found = FAMILY.match(replace(record, payload=stripped), None,
                         {"field": "keywords", "value": "trample"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_pt_boost_fires_the_pt_field(by_kind):
    record = next(r for r in by_kind("continuous")
                  if any(c.pt_boost != (0, 0) for c in r.payload.contributions))
    assert FAMILY.match(record, None, {"field": "pt"}).polarity is Polarity.FIRES


def test_another_keyword_is_the_near_miss(by_kind):
    found = FAMILY.match(_trample_grant(by_kind), None, {"field": "keywords", "value": "flying"})
    assert found.polarity is Polarity.DOES_NOT_FIRE
