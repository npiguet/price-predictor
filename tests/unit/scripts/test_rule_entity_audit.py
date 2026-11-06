"""``scripts/rule_entity_audit.py`` on hand-built records."""

from __future__ import annotations

from types import SimpleNamespace

import rule_entity_audit as audit_script

from effects.domain.event_schema import Event, EventType
from effects.domain.records import CombatPayload, RecordKind, ResolutionPayload
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import DetectorIndex, RuleTag
from effects.domain.state_snapshot import PlayerState
from tests.unit.effects.builders import make_entity, make_record, make_snapshot

PLAYERS = (PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0),
           PlayerState(id="P1", life=17, hand=3, library=30, graveyard=0))


def _deathtouch_combat():
    """E1 (the deathtouch carrier) blocks E2 and kills it; E1 takes no damage."""
    board = (make_entity("E1", "carrier"), make_entity("E2", "attacker", controller="P1"),
             make_entity("E3", "bystander"))
    return make_record(
        RecordKind.COMBAT, generation=3, state=make_snapshot(entities=board, players=PLAYERS),
        payload=CombatPayload(attackers=("E2",), blocks={"E2": ("E1",)}, events=(
            Event(EventType.DAMAGE_DEALT, ("E2",), {"amount": 1, "source": "E1"}),
            Event(EventType.ZONE_CHANGE, ("E2",),
                  {"from_zone": "battlefield", "to_zone": "graveyard"}),
        )))


def _tag(rule, outcome, polarity=Polarity.FIRES, condition=()):
    return RuleTag(rule, polarity, True, tuple(condition), tuple(outcome), f"{rule}-1")


def test_an_outcome_on_the_carrier_misses_the_outputs_its_victim_carries():
    audit = audit_script.Audit({"702.2b": ("zone_outcome", "damage_taken", "gate")})
    audit.add(_deathtouch_combat(), [("keyword_combat", _tag("702.2b", ["E1"]))])

    tally = audit.tallies[("702.2b", "keyword_combat", "fires")]
    assert tally.records == 1 and tally.empty == 0
    assert tally.kinds == {"card:battlefield": 1}
    zone = tally.outputs["zone_outcome"]
    assert (zone.sets["outcome"].pairs, zone.sets["outcome"].supervised) == (1, 0)
    # The carrier is unaffected, so a rule test reads the gated field's zero there.
    assert zone.sets["outcome"].gate_zero == 1
    # Other entities (in neither role): E2, E3, P0, P1; only E2 moved.
    assert (zone.sets["other"].pairs, zone.sets["other"].changed) == (4, 1)
    assert (zone.sets["affected"].pairs, zone.sets["affected"].changed) == (1, 1)
    assert (zone.changed_anywhere, zone.changed_at_tagged, zone.changed_at_other) == (1, 0, 1)
    gate = tally.outputs["gate"]
    assert (gate.sets["outcome"].supervised, gate.sets["outcome"].changed) == (1, 0)

    [miss] = audit_script.misses(audit)
    assert miss.hit == 0.0 and miss.tagged == 0.0 and miss.affected == 1.0
    assert "zone_outcome[t0/0 z100 c- a100/100 h0 o100 n1]" in miss.outputs


def test_a_condition_entity_is_measured_apart_from_the_outcome():
    audit = audit_script.Audit({"702.2b": ("zone_outcome",)}, effective={"702.2b": ()})
    audit.add(_deathtouch_combat(),
              [("keyword_combat", _tag("702.2b", ["E2"], condition=["E1"]))])
    zone = audit.tallies[("702.2b", "keyword_combat", "fires")].outputs["zone_outcome"]
    assert (zone.sets["outcome"].supervised, zone.sets["outcome"].changed) == (1, 1)
    assert (zone.sets["condition"].pairs, zone.sets["condition"].supervised) == (1, 0)
    # Neither role: E3, P0, P1.
    assert zone.sets["other"].pairs == 3
    [miss] = audit_script.misses(audit)
    assert miss.outputs.startswith("~zone_outcome[t100/100")


def test_an_outcome_tag_hits_and_record_level_outputs_are_read_per_record():
    record = _deathtouch_combat()
    audit = audit_script.Audit({"704.5g": ("zone_outcome",), "111.10a": ("created_objects",)})
    audit.add(record, [("sba_zone_move", _tag("704.5g", ["E2"])),
                       ("outcome_event", _tag("111.10a", []))])

    zone = audit.tallies[("704.5g", "sba_zone_move", "fires")].outputs["zone_outcome"]
    assert (zone.sets["outcome"].supervised, zone.sets["outcome"].changed) == (1, 1)
    assert (zone.changed_anywhere, zone.changed_at_tagged) == (1, 1)
    created = audit.tallies[("111.10a", "outcome_event", "fires")]
    assert created.empty == 1
    # A combat record supervises no created-objects head.
    assert created.outputs["created_objects"].record_supervised == 0

    resolution = make_record(generation=3, state=make_snapshot(players=PLAYERS),
                             payload=ResolutionPayload(events=(Event(
                                 EventType.TOKEN_CREATED, ("P0",), {"count": 1}),)))
    audit.add(resolution, [("outcome_event", _tag("111.10a", []))])
    assert created.outputs["created_objects"].record_supervised == 1
    assert created.outputs["created_objects"].record_changed == 1


def test_entity_kinds_name_act_players_zones_and_missing_refs():
    record = _deathtouch_combat()
    assert audit_script.entity_kind(record, "[ACT]") == "act"
    assert audit_script.entity_kind(record, "P1") == "player"
    assert audit_script.entity_kind(record, "E3") == "card:battlefield"
    assert audit_script.entity_kind(record, "E99") == "off-snapshot"


def test_the_index_is_cut_per_family():
    def entry(family):
        return (SimpleNamespace(rules=("1.1",)), SimpleNamespace(family=family), None)

    index = DetectorIndex({"combat": (entry("a"), entry("b")), "trigger": (entry("a"),)},
                          frozenset({"1.1"}))
    cut = audit_script.family_indexes(index)
    assert sorted(cut) == ["a", "b"]
    assert {kind: len(v) for kind, v in cut["a"].by_kind.items()} == {"combat": 1, "trigger": 1}
    assert set(cut["b"].by_kind) == {"combat"}


def test_the_unsupported_list_names_each_map_output_no_record_carries():
    audit = audit_script.Audit(
        {"508.4": ("trigger_fired", "life_delta"), "704.5g": ("zone_outcome",)},
        effective={"508.4": ("trigger_fired",), "704.5g": ("zone_outcome",)})
    audit.add(_deathtouch_combat(), [("trigger_mode", _tag("508.4", [])),
                                     ("sba_zone_move", _tag("704.5g", ["E2"]))])
    lines = audit_script.unsupported_lines(audit, "corpus")
    rows = [line for line in lines if line.startswith("| 5") or line.startswith("| 7")]
    assert len(rows) == 1
    assert rows[0].startswith("| 508.4 | life_delta | trigger_fired | trigger_mode |")
