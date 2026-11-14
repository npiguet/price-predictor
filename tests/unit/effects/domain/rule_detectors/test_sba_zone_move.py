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
        if record.kind.value not in ("resolution", "combat") or (
                record.kind.value == "resolution" and record.moment.value != "resolution"):
            assert _toughness_zero(record) is None


def test_lethal_damage_tags_a_real_damage_death(gen2_records):
    matches = [FAMILY.match(r, None, {"check": "lethal_damage"}) for r in gen2_records]
    fired = [m for m in matches if m is not None and m.polarity is Polarity.FIRES]
    assert fired and all(m.observed for m in fired)


def test_an_unknown_check_is_refused():
    with pytest.raises(ValueError, match="unknown check"):
        FAMILY.validate({"check": "legend_rule_typo"})


def _fires(records, check):
    from effects.domain.rule_catalogue import Polarity as P

    return next(r for r in records if r.kind.value == "resolution"
                and (m := FAMILY.match(r, None, {"check": check})) and m.polarity is P.FIRES)


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
    record = _fires(gen2_records, "aura_illegal")
    found = FAMILY.match(record, None, {"check": "aura_illegal"})
    assert found.observed
    # The Aura is judged and goes to the graveyard; its host is read too.
    hosts = {record.state.entity(a).attached_to for a in found.outcome}
    assert set(found.condition) == set(found.outcome) | hosts


def test_an_aura_whose_host_stays_is_the_near_miss(gen2_records):
    record = _fires(gen2_records, "aura_illegal")
    found = FAMILY.match(record, None, {"check": "aura_illegal"})
    hosts = {record.state.entity(a).attached_to for a in found.outcome}
    events = tuple(e for e in record.payload.events
                   if not (e.type is EventType.ZONE_CHANGE
                           and e.params.get("from_zone") == "battlefield"
                           and (set(e.subjects) & (hosts | set(found.outcome)))))
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


def _granted(record, entity_id, keyword):
    entity = record.state.entity(entity_id)
    granted = replace(entity, granted_temporary=replace(
        entity.granted_temporary, keywords=(*entity.granted_temporary.keywords, keyword)))
    return replace(record, state=replace(record.state, entities=tuple(
        granted if e.id == entity_id else e for e in record.state.entities)))


@pytest.mark.parametrize("keyword", ["wither", "infect"])
def test_a_wither_or_infect_kill_is_no_lethal_damage(gen2_records, keyword):
    """V5 #24: damage from a source with wither or infect is dealt as -1/-1
    counters (702.80a, 702.90b) and marks none, so its kill is 704.5f's, not
    704.5g's: neither side of ``lethal_damage``."""
    record = _fires(gen2_records, "lethal_damage")
    found = FAMILY.match(record, None, {"check": "lethal_damage"})
    hits = [e for e in record.payload.events if e.type is EventType.DAMAGE_DEALT
            and set(e.subjects) & set(found.outcome)]
    sources = {e.params.get("source") or e.params.get("cause") for e in hits}
    assert len(sources) == 1 and None not in sources
    withered = _granted(record, sources.pop(), keyword)
    assert FAMILY.match(withered, None, {"check": "lethal_damage"}) is None


def test_two_legends_the_record_touches_neither_of_show_no_legend_rule(gen2_records):
    """V5 #24: a pair already sharing the battlefield (Mirror Gallery, or a
    check that already ran) is no situation the record's changes made."""
    record = _fires(gen2_records, "legend_rule")
    found = FAMILY.match(record, None, {"check": "legend_rule"})
    state = replace(record.state, entities=tuple(
        replace(e, zone="battlefield") if e.id in found.entities else e
        for e in record.state.entities))
    untouched = replace(record, state=state, payload=ResolutionPayload(events=tuple(
        e for e in record.payload.events if not set(e.subjects) & set(found.entities))))
    assert FAMILY.match(untouched, None, {"check": "legend_rule"}) is None


#: Real gen-2 records the review list held before T036, each read by hand
#: (``scripts/make_fixture_records.py``'s ``_GEN2_PINNED``), and what the
#: family now says of them: ``(check, polarity, observed)``, or None for a
#: record the check cannot read.
_REVIEWED = {
    # A death alone names no cause: Nyx Infusion's -2/-2 static kills with no P/T change.
    "71253f20-5472-4ddd-8c58-27c78651878b.8-mv3ts9u2nlt0.11931": ("toughness_zero", None),
    # The game-end rescue: its board is read after a whole turn of events.
    "71253f20-5472-4ddd-8c58-27c78651878b.8-mv3ts9u2nlt0.12970": ("toughness_zero", None),
    # Disintegrate's exile replaces the death: still the check's move.
    "71253f20-5472-4ddd-8c58-27c78651878b.7-mv4cjj66gz7g.13429":
        ("lethal_damage", (Polarity.FIRES, True)),
    # Stuffy Doll is indestructible: 704.5g destroys, so it is no side.
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv42599vpk5m.16775": ("lethal_damage", None),
    # Pestilence takes a player to 0: the check ends the game, no creature dies.
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv42599vpk5m.1744": ("lethal_damage", None),
    # One damage from a deathtouch creature: 704.5h fires, 704.5g has no near miss.
    "e4713c96-1647-4d93-ac89-039a435f0363.11-mv4w45ojam2q.11464": ("lethal_damage", None),
    # Deathtouch granted by the resolution itself, then the fight.
    "e4713c96-1647-4d93-ac89-039a435f0363.0-mv4yz1t23qrs.5464":
        ("deathtouch_damage", (Polarity.FIRES, True)),
    # Protection from white makes the white Aura fall.
    "e4713c96-1647-4d93-ac89-039a435f0363.11-mv4w45ojam2q.15348":
        ("aura_illegal", (Polarity.FIRES, True)),
    # A bestowed Aura becomes a creature when its host leaves (702.103e).
    "e4713c96-1647-4d93-ac89-039a435f0363.2-mv541ovjbkr3.2538": ("aura_illegal", None),
    # Ordeal of Thassa sacrifices itself; gen-2 attributes it as it does the check.
    "71253f20-5472-4ddd-8c58-27c78651878b.2-mv465aywnc5h.14696": ("aura_illegal", None),
    # The Aura is the destroy spell's target.
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv4cne1d8rhq.2995": ("aura_illegal", None),
    # The host destroyed, its Aura exiled in the check.
    "71253f20-5472-4ddd-8c58-27c78651878b.9-mv4cne1d8rhq.2092":
        ("aura_illegal", (Polarity.FIRES, True)),
    # The host stops being a creature: what the Aura enchants is on no record.
    "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.11-mv5epq5qc9mg.3703": ("aura_illegal", None),
    # A lone legend entering at 0/0 dies by 704.5f, not the legend rule.
    "50e3aa10-aa8d-4153-a614-e786a1dc6a6d.11-mv5g0s6bka1j.14623": ("legend_rule", None),
    # A dies trigger's board shows its dead host at toughness 0.
    "71253f20-5472-4ddd-8c58-27c78651878b.10-mv4aecvvg3eo.3650": ("toughness_zero", None),
    # Losing an Aura's +0/+2 leaves marked damage lethal: 704.5g's.
    "e4713c96-1647-4d93-ac89-039a435f0363.7-mv4yq1tr68fr.2370": ("toughness_zero", None),
}


@pytest.mark.parametrize("record_id", sorted(_REVIEWED))
def test_a_reviewed_record_reads_as_judged(gen2_records, gen2_sidecars, record_id):
    check, expected = _REVIEWED[record_id]
    found = FAMILY.match(_by_id(gen2_records, record_id), gen2_sidecars, {"check": check})
    if expected is None:
        assert found is None
    else:
        assert (found.polarity, found.observed) == expected


def test_a_saga_sacrificed_in_the_check_is_no_creature_check_s(gen2_records, gen2_sidecars):
    """Summon: Shiva's last chapter: the Saga creature leaves with no P/T change."""
    record = _by_id(gen2_records, "2ed74263-ee28-4948-9621-55dba1160c44.1-mv5im7i940u2.36687")
    assert FAMILY.match(record, gen2_sidecars, {"check": "toughness_zero"}) is None


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
