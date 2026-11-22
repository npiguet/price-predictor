"""``sba_zone_move`` on real gen-3 records (T025, story 1 acceptance 1).

Every case is a record cut from the gen-3 fixture; the near misses are real
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

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
_QUICKSAND = f"{_RUN}.0-mwqvqdyrb9c0.315"  # -1/-2 kills a 2/1
_MIASMA_SURVIVED = f"{_RUN}.4-mwqw395jccfe.1855"  # -2/-2 on a 3/3


def _by_id(records, record_id):
    return next(r for r in records if r.record_id == record_id)


def _toughness_zero(record):
    return FAMILY.match(record, None, {"check": "toughness_zero"})


def test_a_creature_at_toughness_below_zero_that_dies_fires_and_is_observed(gen3_records):
    found = _toughness_zero(_by_id(gen3_records, _QUICKSAND))
    assert found.polarity is Polarity.FIRES and found.observed
    assert found.entities == ("E80",)  # Slavering Nulls, a 2/1


def test_a_creature_already_at_toughness_zero_that_dies_fires(gen3_records):
    """The fixture holds no 0/0 creature dying, so the real record is edited:
    the Nulls is printed 0/0 on the board and no P/T change lowers it."""
    record = _by_id(gen3_records, _QUICKSAND)
    nulls = record.state.entity("E80")
    state = replace(record.state, entities=tuple(
        replace(nulls, pt=replace(nulls.pt, base=(0, 0))) if e.id == "E80" else e
        for e in record.state.entities))
    events = tuple(e for e in record.payload.events if e.type is not EventType.PT_CHANGE)
    found = _toughness_zero(replace(record, state=state, payload=ResolutionPayload(events=events)))
    assert found.polarity is Polarity.FIRES and found.observed
    assert found.entities == ("E80",)


def test_a_creature_left_at_toughness_one_is_the_near_miss(gen3_records):
    found = _toughness_zero(_by_id(gen3_records, _MIASMA_SURVIVED))
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_fires_record_edited_to_a_tougher_creature_is_rejected(gen3_records):
    """The near miss edited from a real record, the 704.5f scenario's one fact:
    Quicksand's -1/-2 meets a 1/3 instead of a 2/1, which keeps toughness 1."""
    record = _by_id(gen3_records, _QUICKSAND)
    nulls = record.state.entity("E80")
    tougher = replace(nulls, pt=replace(nulls.pt, base=(1, 3)))
    state = replace(record.state, entities=tuple(
        tougher if e.id == "E80" else e for e in record.state.entities))
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE and "E80" in e.subjects))
    edited = replace(record, state=state, payload=ResolutionPayload(events=events))
    found = _toughness_zero(edited)
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_a_death_the_rule_says_should_not_happen_goes_on_the_review_list(gen3_records):
    record = _by_id(gen3_records, _MIASMA_SURVIVED)
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


def _outside_resolution_and_combat(record):
    if record.kind.value not in ("resolution", "combat"):
        return True
    return record.kind.value == "resolution" and record.moment.value != "resolution"


def test_records_of_other_kinds_and_moments_show_nothing(gen3_records):
    for record in filter(_outside_resolution_and_combat, gen3_records):
        assert _toughness_zero(record) is None


def test_lethal_damage_tags_a_real_damage_death(gen3_records):
    matches = [FAMILY.match(r, None, {"check": "lethal_damage"}) for r in gen3_records]
    fired = [m for m in matches if m is not None and m.polarity is Polarity.FIRES]
    assert fired and all(m.observed for m in fired)


def test_an_unknown_check_is_refused():
    with pytest.raises(ValueError, match="unknown check"):
        FAMILY.validate({"check": "legend_rule_typo"})


def _fires(records, check):
    from effects.domain.rule_catalogue import Polarity as P

    return next(r for r in records if r.kind.value == "resolution"
                and (m := FAMILY.match(r, None, {"check": check})) and m.polarity is P.FIRES)


def test_a_second_legend_entering_fires_the_legend_rule(gen3_records):
    record = _fires(gen3_records, "legend_rule")
    found = FAMILY.match(record, None, {"check": "legend_rule"})
    assert found.observed and len(found.entities) == 2
    names = {record.state.entity(e).name for e in found.entities}
    assert len(names) == 1


def test_the_same_legend_entering_under_another_controller_is_the_near_miss(gen3_records):
    """The near miss edited from the real record: the entering copy changes controller."""
    record = _fires(gen3_records, "legend_rule")
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


def test_an_aura_whose_host_left_fires(gen3_records):
    record = _fires(gen3_records, "aura_illegal")
    found = FAMILY.match(record, None, {"check": "aura_illegal"})
    assert found.observed
    # The Aura is judged and goes to the graveyard; its host is read too.
    hosts = {record.state.entity(a).attached_to for a in found.outcome}
    assert set(found.condition) == set(found.outcome) | hosts


def test_an_aura_whose_host_stays_is_the_near_miss(gen3_records):
    record = _fires(gen3_records, "aura_illegal")
    found = FAMILY.match(record, None, {"check": "aura_illegal"})
    hosts = {record.state.entity(a).attached_to for a in found.outcome}
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE
                           and e.params.get("from_zone") == "battlefield"
                           and (set(e.subjects) & (hosts | set(found.outcome)))))
    stayed = FAMILY.match(replace(record, payload=ResolutionPayload(events=events)), None,
                          {"check": "aura_illegal"})
    assert stayed.polarity is Polarity.DOES_NOT_FIRE and not stayed.observed


def test_a_planeswalker_losing_its_last_loyalty_fires(gen3_records):
    record = next(r for r in gen3_records if any(
        e.type is EventType.COUNTER_CHANGE and e.params.get("counter_type") == "LOYALTY"
        for e in r.payload.listed_events()))
    found = FAMILY.match(record, None, {"check": "loyalty_zero"})
    assert found.polarity is Polarity.FIRES and found.observed


def _is_loyalty_change(event):
    return event.type is EventType.COUNTER_CHANGE and event.params.get("counter_type") == "LOYALTY"


def _with_one_more_loyalty(state, walker_id):
    walker = state.entity(walker_id)
    stronger = replace(walker, counters={**walker.counters,
                                         "LOYALTY": walker.counters["LOYALTY"] + 1})
    return replace(state, entities=tuple(
        stronger if e.id == walker_id else e for e in state.entities))


def test_a_planeswalker_left_with_loyalty_is_the_near_miss(gen3_records):
    """Chandra, Flamecaller, given a loyalty counter more, loses one of her
    two instead of her last, and stays."""
    record = next(r for r in gen3_records if any(
        _is_loyalty_change(e) for e in r.payload.listed_events()))
    walker_id = next(e.subjects[0] for e in record.payload.events if _is_loyalty_change(e))
    state = _with_one_more_loyalty(record.state, walker_id)
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE and e.attributed_to == "unresolved"))
    found = FAMILY.match(replace(record, state=state, payload=ResolutionPayload(events=events)),
                         None, {"check": "loyalty_zero"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def _small_hit(gen3_records):
    """A real 704.5g kill edited to a 1-damage hit: non-lethal, its source named."""
    record = _fires(gen3_records, "lethal_damage")
    killed = set(FAMILY.match(record, None, {"check": "lethal_damage"}).outcome)
    hit = next(e for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
               and set(e.subjects) & killed
               and (e.params.get("source") or e.params.get("cause")))
    events = tuple(
        replace(e, params={**e.params, "amount": 1}) if e is hit else e
        for e in record.payload.events
        if not (e.type is EventType.ZONE_CHANGE and e.attributed_to == "unresolved"))
    return replace(record, payload=ResolutionPayload(events=events)), hit


def test_a_small_hit_from_a_source_without_deathtouch_is_the_near_miss(gen3_records):
    record, _ = _small_hit(gen3_records)
    found = FAMILY.match(record, None, {"check": "deathtouch_damage"})
    assert found.polarity is Polarity.DOES_NOT_FIRE and not found.observed


def test_the_same_hit_from_a_deathtouch_source_fires(gen3_records):
    record, hit = _small_hit(gen3_records)
    source_id = hit.params.get("source") or hit.params.get("cause")
    source = record.state.entity(source_id)
    granted = replace(source, granted_temporary=replace(
        source.granted_temporary, keywords=(*source.granted_temporary.keywords, "deathtouch")))
    state = replace(record.state, entities=tuple(
        granted if e.id == source_id else e for e in record.state.entities))
    found = FAMILY.match(replace(record, state=state), None, {"check": "deathtouch_damage"})
    assert found.polarity is Polarity.FIRES and not found.observed  # the edit left no death


def _granted(record, entity_id, keyword):
    entity = record.state.entity(entity_id)
    granted = replace(entity, granted_temporary=replace(
        entity.granted_temporary, keywords=(*entity.granted_temporary.keywords, keyword)))
    return replace(record, state=replace(record.state, entities=tuple(
        granted if e.id == entity_id else e for e in record.state.entities)))


@pytest.mark.parametrize("keyword", ["wither", "infect"])
def test_a_wither_or_infect_kill_is_no_lethal_damage(gen3_records, keyword):
    """V5 #24: damage from a source with wither or infect is dealt as -1/-1
    counters (702.80a, 702.90b) and marks none, so its kill is 704.5f's, not
    704.5g's: neither side of ``lethal_damage``."""
    record = _fires(gen3_records, "lethal_damage")
    found = FAMILY.match(record, None, {"check": "lethal_damage"})
    hits = [e for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
            and set(e.subjects) & set(found.outcome)]
    sources = {e.params.get("source") or e.params.get("cause") for e in hits}
    assert len(sources) == 1 and None not in sources
    withered = _granted(record, sources.pop(), keyword)
    assert FAMILY.match(withered, None, {"check": "lethal_damage"}) is None


def test_two_legends_the_record_touches_neither_of_show_no_legend_rule(gen3_records):
    """V5 #24: a pair already sharing the battlefield (Mirror Gallery, or a
    check that already ran) is no situation the record's changes made."""
    record = _fires(gen3_records, "legend_rule")
    found = FAMILY.match(record, None, {"check": "legend_rule"})
    state = replace(record.state, entities=tuple(
        replace(e, zone="battlefield") if e.id in found.entities else e
        for e in record.state.entities))
    untouched = replace(record, state=state, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if not set(e.subjects) & set(found.entities))))
    assert FAMILY.match(untouched, None, {"check": "legend_rule"}) is None


#: Real gen-3 records read by hand, and what the family says of them:
#: ``(check, expected)``, the expected ``(polarity, observed)`` or None for a
#: record the check cannot read.
_REVIEWED = [
    # Ghostfire's 3 damage kills a 3/3: 704.5g's move, observed.
    ("2-mwqvqdxrusnv.3944", "lethal_damage", (Polarity.FIRES, True)),
    # A combat whose surviving creatures took less than their toughness, and
    # no deathtouch source dealt any of it.
    ("2-mwqvqdxrusnv.2623", "deathtouch_damage", (Polarity.DOES_NOT_FIRE, False)),
    # The game-end rescue: its board is read after a whole turn of events.
    ("2-mwqvqdxrusnv.4639", "toughness_zero", None),
    # Molten Disaster's damage to each creature kills an enchanted host: the
    # Aura falls with it.
    ("2-mwqvqdxrusnv.5022", "aura_illegal", (Polarity.FIRES, True)),
    # The same record's creatures that took less than their toughness.
    ("2-mwqvqdxrusnv.5022", "deathtouch_damage", (Polarity.DOES_NOT_FIRE, False)),
    # The effect touches Imperiosaur, enchanted by Bound in Silence, and it
    # stays: the Aura stays legal.
    ("2-mwqvqdxrusnv.3912", "aura_illegal", (Polarity.DOES_NOT_FIRE, False)),
    # Searing Blaze's 3 damage kills an Elephant token, and Claws of Valakut
    # goes with it.
    ("0-mwqvqdyrb9c0.613", "aura_illegal", (Polarity.FIRES, True)),
    # Two legendary permanents of one name under one controller.
    ("3-mwqvu8zeq2sj.8902", "legend_rule", (Polarity.FIRES, True)),
]


@pytest.mark.parametrize("case", _REVIEWED, ids=[f"{r[0]}-{r[1]}" for r in _REVIEWED])
def test_a_reviewed_record_reads_as_judged(gen3_records, gen3_sidecars, case):
    suffix, check, expected = case
    found = FAMILY.match(_by_id(gen3_records, f"{_RUN}.{suffix}"), gen3_sidecars,
                         {"check": check})
    if expected is None:
        assert found is None
    else:
        assert (found.polarity, found.observed) == expected


def test_a_saga_sacrificed_in_the_check_is_no_creature_check_s(gen3_records, gen3_sidecars):
    """Summon: Shiva's last chapter: the Saga creature leaves with no P/T
    change. The fixture holds no Saga, so a real kill is edited to have none:
    its P/T change gone and its victim a Saga creature."""
    record = _by_id(gen3_records, _QUICKSAND)
    victim = record.state.entity("E80")
    saga = replace(victim, name="Summon: Shiva", subtypes=("saga",))
    state = replace(record.state, entities=tuple(
        saga if e.id == "E80" else e for e in record.state.entities))
    events = tuple(e for e in record.payload.events if e.type is not EventType.PT_CHANGE)
    edited = replace(record, state=state, payload=ResolutionPayload(events=events))
    assert FAMILY.match(edited, gen3_sidecars, {"check": "toughness_zero"}) is None


def _combat_death(keywords=(), amount=2, granted=()):
    from effects.domain.event_schema import Event
    from effects.domain.records import CombatPayload, RecordKind
    from effects.domain.state_snapshot import GrantedTemporary, PowerToughness
    from tests.unit.effects.builders import make_entity, make_record, make_snapshot

    attacker = make_entity("E1", "giant", types=("Creature",), pt=PowerToughness((3, 3)),
                           granted_temporary=GrantedTemporary(keywords=tuple(keywords)))
    blocker = make_entity("E2", "bears", controller="P1", types=("Creature",),
                          pt=PowerToughness((2, 2)))
    events = [Event(type=EventType.DAMAGE_DEALT, subjects=("E2",),
                    params={"amount": amount, "combat": True, "source": "E1"},
                    attributed_to="unresolved")]
    events += [Event(type=EventType.KEYWORD_CHANGE, subjects=("E2",),
                     params={"keywords": list(granted)}, attributed_to="unresolved")
               for _ in granted[:1]]
    events.append(Event(type=EventType.ZONE_CHANGE, subjects=("E2",),
                        params={"from_zone": "battlefield", "to_zone": "graveyard"},
                        attributed_to="unresolved"))
    return make_record(RecordKind.COMBAT, state=make_snapshot(entities=(attacker, blocker)),
                       payload=CombatPayload(attackers=("E1",), events=tuple(events)))


def test_a_combat_death_is_the_check_s_own():
    """The commonest lethal-damage death is combat's; the family read
    resolutions alone."""
    found = FAMILY.match(_combat_death(), None, {"check": "lethal_damage"})
    assert (found.polarity, found.observed, found.outcome) == (Polarity.FIRES, True, ("E2",))
    deathtouch = FAMILY.match(_combat_death(("deathtouch",), amount=1), None,
                              {"check": "deathtouch_damage"})
    assert deathtouch.polarity is Polarity.FIRES


def test_indestructible_the_record_grants_spares_the_creature_from_the_check():
    assert FAMILY.match(_combat_death(granted=("indestructible",)), None,
                        {"check": "lethal_damage"}) is None
