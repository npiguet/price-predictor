"""``sba_zone_move`` on real gen-2 records (T025, story 1 acceptance 1).

Every case is a record cut from the gen-2 shards; the near misses are real
``fires`` records edited by one fact, as the spec allows.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import EventType
from effects.domain.records import ResolutionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import DetectorIndex, tag_record
from effects.domain.rule_detectors.sba_zone_move import FAMILY
from tests.unit.effects.domain.rule_detectors.builders import catalogue, declaration, scenario

_DECREE = "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.0-mv5fnkmsv6yb.1019"  # Decree of Pain, -2/-2
_SAPROLINGS = "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.0-mv5ek090so4u.7864"  # 0/0 tokens
_GRIND_DUST = "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.1-mv5eiqndmfap.8695"  # left at toughness 1


def _by_id(records, record_id):
    return next(r for r in records if r.record_id == record_id)


def _toughness_zero(record):
    return FAMILY.match(record, None, {"check": "toughness_zero"})


def test_a_creature_at_toughness_below_zero_that_dies_fires_and_is_observed(gen2_records):
    found = _toughness_zero(_by_id(gen2_records, _DECREE))
    assert found.polarity is Polarity.FIRES and found.observed
    assert found.entities == ("E63",)  # Mischievous Lookout


def test_zero_zero_tokens_dying_fire(gen2_records):
    found = _toughness_zero(_by_id(gen2_records, _SAPROLINGS))
    assert found.polarity is Polarity.FIRES and found.observed
    assert len(found.entities) == 3


def test_a_creature_left_at_toughness_one_is_the_near_miss(gen2_records):
    found = _toughness_zero(_by_id(gen2_records, _GRIND_DUST))
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_fires_record_edited_to_a_tougher_creature_is_rejected(gen2_records):
    """The near miss edited from a real record, the 704.5f scenario's one fact:
    Decree of Pain's -2/-2 meets a 2/3 instead of a 2/1, which keeps toughness 1."""
    record = _by_id(gen2_records, _DECREE)
    lookout = record.state.entity("E63")
    tougher = replace(lookout, pt=replace(lookout.pt, base=(2, 3)))
    state = replace(record.state, entities=tuple(
        tougher if e.id == "E63" else e for e in record.state.entities))
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE and "E63" in e.subjects))
    edited = replace(record, state=state, payload=ResolutionPayload(events=events))
    found = _toughness_zero(edited)
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_death_the_rule_says_should_not_happen_goes_on_the_review_list(gen2_records):
    record = _by_id(gen2_records, _GRIND_DUST)
    index = DetectorIndex.of(catalogue(scenario(
        "704.5f-1", ("704.5f",), "resolution",
        declaration("sba_zone_move", "resolution", check="toughness_zero"))))
    assert not tag_record(record, index).review
    target = next(e for e in record.payload.events if e.type is EventType.PT_CHANGE)
    death = replace(target, type=EventType.ZONE_CHANGE, attributed_to="unresolved",
                    params={"from_zone": "battlefield", "to_zone": "graveyard"})
    disagreeing = replace(record, payload=ResolutionPayload(
        events=(*record.payload.events, death)))
    (tag,) = tag_record(disagreeing, index).tags
    assert tag.polarity is Polarity.DOES_NOT_FIRE and tag.observed and tag.needs_review


def test_records_of_other_kinds_and_moments_show_nothing(gen2_records):
    for record in gen2_records:
        if record.kind.value != "resolution" or record.moment.value != "resolution":
            assert _toughness_zero(record) is None


def test_lethal_damage_tags_a_real_damage_death(gen2_records):
    matches = [FAMILY.match(r, None, {"check": "lethal_damage"}) for r in gen2_records]
    fired = [m for m in matches if m is not None and m.polarity is Polarity.FIRES]
    assert fired and all(m.observed for m in fired)


def test_an_unknown_check_is_refused():
    with pytest.raises(ValueError, match="unknown check"):
        FAMILY.match(None, None, {"check": "legend_rule_typo"})


def test_the_family_runs_on_gen2_records():
    assert FAMILY.needs({"check": "toughness_zero"}) == frozenset()
