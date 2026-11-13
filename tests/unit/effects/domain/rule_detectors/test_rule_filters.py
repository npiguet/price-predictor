"""The parameters D4 added to tell one rule's records from another's.

``verdict_bit``'s candidate, source type, source designation and sorcery
speed; ``outcome_event``'s source type, acting face and target face state;
``trigger_mode``'s source subtype and subject qualifiers; ``sba_zone_move``'s
``marked_before`` and ``aura_cause``. Hand-built boards in the records'
own spelling.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    TriggerPayload,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import families
from effects.domain.state_snapshot import PowerToughness, Refs
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

VERDICT = families()["verdict_bit"]
OUTCOME = families()["outcome_event"]
TRIGGER = families()["trigger_mode"]
SBA = families()["sba_zone_move"]
_CAST = ProvenanceKey("cardsfolder/g/grizzly_bears.txt", 0, "spell", 0)


def _decision(source, timing, key=_CAST):
    candidate = Candidate(ability=(key,), can_play=timing, affordable=True,
                          has_legal_target=True, timing=timing, failed_checks=())
    state = make_snapshot(entities=(source,), refs=Refs(source=source.id))
    return make_record(RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION, state=state,
                       payload=PlayabilityDecisionPayload(candidates=(candidate,)), generation=3)


def test_a_cast_is_judged_by_its_card_s_type():
    """D4: 301.1-310.1 each read the timing of their own card type's casts."""
    bears = make_entity("E1", "grizzly bears", zone="hand", types=("Creature",))
    record = _decision(bears, True)
    timing = {"bit": "timing", "fires_when": True, "candidate": "cast"}
    assert VERDICT.match(record, None, {**timing, "source_type": "creature"}).polarity is (
        Polarity.FIRES)
    assert VERDICT.match(record, None, {**timing, "source_type": "sorcery"}) is None
    # A dropped first spell trait is the permanent's implicit cast; no line, no activation.
    assert VERDICT.match(record, None, {**timing, "candidate": "activated"}) is None


def test_sorcery_speed_reads_the_card_s_type():
    bears = make_entity("E1", "grizzly bears", zone="hand", types=("Creature",))
    bolt = make_entity("E1", "lightning bolt", zone="hand", types=("Instant",))
    params = {"bit": "timing", "fires_when": True, "candidate": "cast"}
    assert VERDICT.match(_decision(bears, False), None, {**params, "sorcery_speed": True}
                         ).polarity is Polarity.DOES_NOT_FIRE
    assert VERDICT.match(_decision(bears, False), None, {**params, "sorcery_speed": False}) is None
    assert VERDICT.match(_decision(bolt, True), None, {**params, "sorcery_speed": False}
                         ).polarity is Polarity.FIRES


def test_a_source_designation_narrows_the_candidates():
    plotted = make_entity("E1", "x", zone="exile", types=("Creature",), designations=("plotted",),
                          lines=())
    plain = replace(plotted, designations=())
    params = {"bit": "timing", "fires_when": True, "source_designation": "plotted"}
    assert VERDICT.match(_decision(plotted, True), None, params).polarity is Polarity.FIRES
    assert VERDICT.match(_decision(plain, True), None, params) is None


@pytest.mark.parametrize("params", [
    {"bit": "timing", "candidate": "spell"},
    {"bit": "can_play", "check": "phase"},
])
def test_verdict_bit_refuses_values_it_cannot_read(params):
    with pytest.raises(ValueError):
        VERDICT.validate(params)


def _resolution(source, events, key, targets=()):
    state = make_snapshot(entities=(source, *targets),
                          refs=Refs(source=source.id, targets=tuple(t.id for t in targets)))
    return make_record(state=state, ability=(key,), payload=ResolutionPayload(events=events),
                       generation=3)


def test_outcome_event_reads_the_source_s_type_and_its_acting_face():
    giant = make_entity("E1", "bonecrusher giant", zone="stack", types=("Creature",))
    exiled = Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                   params={"from_zone": "stack", "to_zone": "exile"})
    stomp = ProvenanceKey("cardsfolder/b/bonecrusher_giant.txt", 1, "spell", 0)
    record = _resolution(giant, (exiled,), stomp)
    params = {"event": "zone_change", "from_zone": "stack", "to_zone": "exile"}
    assert OUTCOME.match(record, None, {**params, "acting_face": "alternative"}).polarity is (
        Polarity.FIRES)
    assert OUTCOME.match(record, None, {**params, "acting_face": "front"}) is None
    assert OUTCOME.match(record, None, {**params, "source_type": "creature"}) is not None
    assert OUTCOME.match(record, None, {**params, "source_type": "sorcery"}) is None


def test_outcome_event_reads_a_target_s_face_state():
    """712.15a: what a transform instruction does to a face-down permanent."""
    source = make_entity("E1", "moonmist", zone="stack", types=("Instant",))
    down = make_entity("E2", "delver", types=("Creature",), face_state="face_down", lines=())
    turned = Event(type=EventType.FACE_CHANGE, subjects=("E2",), params={"to_state": "back"})
    key = ProvenanceKey("cardsfolder/m/moonmist.txt", 0, "spell", 0)
    record = _resolution(source, (turned,), key, targets=(down,))
    params = {"event": "face_change", "target_face_state": "face_down"}
    assert OUTCOME.match(record, None, params).polarity is Polarity.FIRES
    assert OUTCOME.needs(params) == {"face_state"}
    up = replace(record, state=replace(record.state, entities=(
        source, replace(down, face_state="front"))))
    assert OUTCOME.match(up, None, params) is None


def _trigger(source, subject, fired=True):
    event = Event(type=EventType.ZONE_CHANGE, subjects=(subject.id,),
                  params={"to_zone": "battlefield"})
    state = make_snapshot(entities=(source, subject), refs=Refs(source=source.id))
    key = ProvenanceKey("cardsfolder/d/druid_class.txt", 0, "trigger", 0)
    return make_record(RecordKind.TRIGGER, state=state, ability=(key,),
                       payload=TriggerPayload(event=event, fired=fired), generation=3)


def test_trigger_mode_reads_its_source_s_subtype_and_its_event_s_subject():
    druid = make_entity("E1", "druid class", types=("Enchantment",), subtypes=("Class",))
    forest = make_entity("E2", "forest", types=("Land",), face_state="front", lines=())
    found = TRIGGER.match(_trigger(druid, forest), None, {"step": "main1",
                                                          "source_subtype": "Class"})
    assert found.polarity is Polarity.FIRES
    assert TRIGGER.match(_trigger(druid, forest), None,
                         {"step": "main1", "source_subtype": "Case"}) is None
    down = replace(forest, face_state="face_down")
    params = {"step": "main1", "subject_face_state": "face_down"}
    assert TRIGGER.match(_trigger(druid, down), None, params).polarity is Polarity.FIRES
    assert TRIGGER.match(_trigger(druid, forest), None, params) is None


def test_sba_zone_move_marked_before_and_aura_cause_are_validated():
    with pytest.raises(ValueError):
        SBA.validate({"check": "toughness_zero", "marked_before": True})
    with pytest.raises(ValueError):
        SBA.validate({"check": "aura_illegal", "aura_cause": "tapped"})
    SBA.validate({"check": "aura_illegal", "aura_cause": "protection"})


def test_marked_before_keeps_only_creatures_with_damage_already_marked():
    """120.6: damage accumulates over the turn."""
    fresh = make_entity("E2", "bears", types=("Creature",), pt=PowerToughness((2, 2)))
    worn = replace(fresh, damage=1)
    hit = Event(type=EventType.DAMAGE_DEALT, subjects=("E2",), params={"amount": 1},
                attributed_to="root")
    key = ProvenanceKey("cardsfolder/s/shock.txt", 0, "spell", 0)
    shock = make_entity("E1", "shock", zone="stack", types=("Instant",))
    params = {"check": "lethal_damage", "marked_before": True}
    on_fresh = _resolution(shock, (hit,), key, targets=(fresh,))
    on_worn = _resolution(shock, (hit,), key, targets=(worn,))
    assert SBA.match(on_fresh, None, params) is None
    assert SBA.match(on_worn, None, params).polarity is Polarity.FIRES
    assert SBA.match(on_fresh, None, {"check": "lethal_damage"}).polarity is (
        Polarity.DOES_NOT_FIRE)
