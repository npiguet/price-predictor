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


def _fires(records, check):
    from effects.domain.rule_catalogue import Polarity as P

    return next(r for r in records
                if (m := FAMILY.match(r, None, {"check": check})) and m.polarity is P.FIRES)


def test_a_second_legend_entering_fires_the_legend_rule(gen2_records):
    record = _fires(gen2_records, "legend_rule")
    found = FAMILY.match(record, None, {"check": "legend_rule"})
    assert found.observed and len(found.entities) == 2
    names = {record.state.entity(e).name for e in found.entities}
    assert len(names) == 1


def test_the_same_legend_entering_under_another_controller_is_the_near_miss(gen2_records):
    """The near miss edited from the real record: the entering copy changes controller."""
    record = _fires(gen2_records, "legend_rule")
    entering = next(e.subjects[0] for e in record.payload.events
                    if e.type is EventType.ZONE_CHANGE and e.params.get("to_zone") == "battlefield")
    other = next(p.id for p in record.state.players
                 if p.id != record.state.entity(entering).controller)
    state = replace(record.state, entities=tuple(
        replace(e, controller=other) if e.id == entering else e for e in record.state.entities))
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE and e.attributed_to == "unresolved"))
    found = FAMILY.match(replace(record, state=state, payload=ResolutionPayload(events=events)),
                         None, {"check": "legend_rule"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_an_aura_whose_host_left_fires(gen2_records):
    found = FAMILY.match(_fires(gen2_records, "aura_illegal"), None, {"check": "aura_illegal"})
    assert found.observed


def test_an_aura_whose_host_stays_is_the_near_miss(gen2_records):
    record = _fires(gen2_records, "aura_illegal")
    found = FAMILY.match(record, None, {"check": "aura_illegal"})
    hosts = {record.state.entity(a).attached_to for a in found.entities}
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE
                           and e.params.get("from_zone") == "battlefield"
                           and (set(e.subjects) & (hosts | set(found.entities)))))
    stayed = FAMILY.match(replace(record, payload=ResolutionPayload(events=events)), None,
                          {"check": "aura_illegal"})
    assert stayed.polarity is Polarity.DOES_NOT_FIRE and not stayed.observed


def test_a_planeswalker_losing_its_last_loyalty_fires(gen2_records):
    record = next(r for r in gen2_records if any(
        e.type is EventType.COUNTER_CHANGE and e.params.get("counter_type") == "LOYALTY"
        for e in r.payload.listed_events()))
    found = FAMILY.match(record, None, {"check": "loyalty_zero"})
    assert found.polarity is Polarity.FIRES and found.observed


def test_a_planeswalker_left_with_loyalty_is_the_near_miss(gen2_records):
    """Sivitri loses 3 of her 4 loyalty instead of all of it, and stays."""
    record = next(r for r in gen2_records if any(
        e.type is EventType.COUNTER_CHANGE and e.params.get("counter_type") == "LOYALTY"
        for e in r.payload.listed_events()))
    events = []
    for e in record.payload.events:
        if e.type is EventType.COUNTER_CHANGE and e.params.get("counter_type") == "LOYALTY":
            e = replace(e, params={**e.params, "delta": e.params["delta"] + 1})
        elif e.type is EventType.ZONE_CHANGE and e.attributed_to == "unresolved":
            continue
        events.append(e)
    found = FAMILY.match(replace(record, payload=ResolutionPayload(events=tuple(events))), None,
                         {"check": "loyalty_zero"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def _small_hit(gen2_records):
    """A real 704.5g kill edited to a 1-damage hit: non-lethal, its source named."""
    record = _fires(gen2_records, "lethal_damage")
    hit = next(e for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
               and (e.params.get("source") or e.params.get("cause")))
    events = tuple(
        replace(e, params={**e.params, "amount": 1}) if e is hit else e
        for e in record.payload.events
        if not (e.type is EventType.ZONE_CHANGE and e.attributed_to == "unresolved"))
    return replace(record, payload=ResolutionPayload(events=events)), hit


def test_a_small_hit_from_a_source_without_deathtouch_is_the_near_miss(gen2_records):
    record, _ = _small_hit(gen2_records)
    found = FAMILY.match(record, None, {"check": "deathtouch_damage"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_the_same_hit_from_a_deathtouch_source_fires(gen2_records):
    record, hit = _small_hit(gen2_records)
    source_id = hit.params.get("source") or hit.params.get("cause")
    source = record.state.entity(source_id)
    granted = replace(source, granted_temporary=replace(
        source.granted_temporary, keywords=(*source.granted_temporary.keywords, "deathtouch")))
    state = replace(record.state, entities=tuple(
        granted if e.id == source_id else e for e in record.state.entities))
    found = FAMILY.match(replace(record, state=state), None, {"check": "deathtouch_damage"})
    assert found.polarity is Polarity.FIRES and not found.observed  # the edit left no death
