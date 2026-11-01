"""``verdict_bit`` on real gen-2 decision records (T028)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from effects.domain.records import PlayabilityDecisionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors.verdict_bit import ACT, FAMILY


def _unaffordable(by_kind):
    return next(r for r in by_kind("playability", "decision")
                if any(not c.affordable for c in r.payload.candidates))


def test_an_unaffordable_candidate_fires_the_cost_rule(by_kind):
    found = FAMILY.match(_unaffordable(by_kind), None, {"bit": "affordable"})
    assert found.polarity is Polarity.FIRES and found.observed and found.entities == (ACT,)


def test_the_same_candidate_edited_affordable_is_the_near_miss(by_kind):
    record = _unaffordable(by_kind)
    paid = PlayabilityDecisionPayload(candidates=tuple(
        replace(c, affordable=True) for c in record.payload.candidates))
    found = FAMILY.match(replace(record, payload=paid), None, {"bit": "affordable"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_responsible_static_counts_only_restricted_candidates(by_kind):
    assert FAMILY.match(_unaffordable(by_kind), None,
                        {"bit": "affordable", "responsible_static": True}) is None


def test_the_gen3_bits_need_their_gen3_fields():
    assert FAMILY.needs({"bit": "timing"}) == {"timing"}
    assert FAMILY.needs({"bit": "failed_check", "check": "casting_speed"}) == {"failed_checks"}
    assert FAMILY.needs({"bit": "can_play"}) == frozenset()


def test_a_failed_check_reads_the_named_identifier(by_kind):
    """Gen-3 candidates carry ``failed_checks``; until the reader parses them
    (T074) the candidate is a stand-in carrying the one field read."""
    record = _unaffordable(by_kind)
    failing = SimpleNamespace(responsible_static=(), failed_checks=("casting_speed",))
    passing = SimpleNamespace(responsible_static=(), failed_checks=())
    params = {"bit": "failed_check", "check": "casting_speed"}
    for candidate, polarity in ((failing, Polarity.FIRES), (passing, Polarity.DOES_NOT_FIRE)):
        edited = SimpleNamespace(kind=record.kind, subkind=record.subkind,
                                 payload=SimpleNamespace(candidates=(candidate,)))
        assert FAMILY.match(edited, None, params).polarity is polarity


def test_other_subkinds_show_nothing_and_a_bad_bit_is_refused(by_kind):
    blockers = by_kind("playability", "blockers")[0]
    assert FAMILY.match(blockers, None, {"bit": "affordable"}) is None
    with pytest.raises(ValueError, match="unknown bit"):
        FAMILY.match(blockers, None, {"bit": "mana_value"})
