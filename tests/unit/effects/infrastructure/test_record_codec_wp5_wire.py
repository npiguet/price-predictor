"""The gen-3 wire as the connector writes it with this batch of members (contract
``record-schema-delta.md``): a fork record's ``copy_dropped``, and a forced
resolution holding the events the engine hooks report.

Read off the Java golden ``record-intervention-hooked.jsonl``, a forced
Raise the Alarm through the hooks on a copy that left one remembered object
out.
"""

from __future__ import annotations

import json

import pytest

from effects.domain.event_schema import EventType
from effects.domain.fork_operation import ForkOperation
from effects.domain.records import DirectedTag, EffectRecord, ResolutionOutcome
from effects.domain.rule_detectors.emittable_events import absence_readable
from effects.domain.wire_closed_lists import off_list_values
from effects.infrastructure.record_codec import format_record_line, record_from_dict
from tests.unit.effects.builders import JAVA_GOLDENS, java_golden_records, make_record


def _lossy_golden() -> dict:
    return next(raw for raw in java_golden_records()
                if raw.get("copy_dropped") and raw.get("interventional"))


class TestCopyDropped:
    def test_a_lossy_fork_record_reads_what_its_copy_left_out(self):
        record = record_from_dict(_lossy_golden())
        assert record.fork
        assert record.copy_dropped == (("remembered", 1),)

    def test_it_writes_back_after_directed_as_the_collector_wrote_it(self):
        raw = _lossy_golden()
        written = json.loads(format_record_line(record_from_dict(raw)))
        assert written["copy_dropped"] == raw["copy_dropped"]
        keys = list(written)
        assert keys.index("copy_dropped") < keys.index("actor_player")

    def test_a_record_whose_copy_left_nothing_out_carries_none(self):
        for raw in java_golden_records():
            if "copy_dropped" not in raw:
                assert record_from_dict(raw).copy_dropped is None
                assert "copy_dropped" not in json.loads(
                    format_record_line(record_from_dict(raw)))
                return
        pytest.fail("every golden carries copy_dropped")

    def test_only_a_fork_carries_it(self):
        with pytest.raises(ValueError, match="copy_dropped"):
            make_record(copy_dropped=(("command", 1),))

    @pytest.mark.parametrize("dropped", [{}, {"command": 0}, {"remembered": 1, "command": -1}])
    def test_an_empty_member_or_a_count_below_one_is_refused(self, dropped):
        """The connector writes the member only for a lossy copy, and each
        kind only with at least one drop."""
        raw = dict(_lossy_golden(), copy_dropped=dropped)
        with pytest.raises(ValueError, match="copy_dropped"):
            record_from_dict(raw)

    def test_the_golden_s_kinds_are_on_the_closed_list(self):
        assert list(off_list_values(_lossy_golden())) == []

    def test_a_kind_off_the_list_is_named(self):
        raw = dict(_lossy_golden(), copy_dropped={"remembered": 1, "spellbook": 2})
        assert ("copy_dropped", "spellbook") in list(off_list_values(raw))


class TestAForcedResolutionHearsTheHooks:
    def test_the_golden_s_forced_half_holds_the_token_the_outcome_hook_reported(self):
        record: EffectRecord = record_from_dict(_lossy_golden())
        types = [event.type for event in record.payload.events]
        assert EventType.TOKEN_CREATED in types

    def test_a_missing_hook_derived_event_on_it_reads_as_not_happened(self):
        record = record_from_dict(_lossy_golden())
        assert absence_readable(record, EventType.COIN_FLIPPED)


def _golden(name: str) -> list[EffectRecord]:
    """One Java golden's records, decoded."""
    path = JAVA_GOLDENS / name
    return [record_from_dict(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestTheForkShapesPythonReads:
    """Each fork shape the connector writes, read off its Java golden
    (``RecordCharacterizationTest``)."""

    def test_a_directed_fork_names_its_rule_variant_and_operation(self):
        records = _golden("record-directed-fork.jsonl")
        assert records
        for record in records:
            assert record.fork and not record.interventional
            assert record.aimed_at == ("704.5f",)
            assert record.directed == DirectedTag("tougher", ForkOperation.PT)
            assert record.mirror_of is None

    def test_a_protection_fork_is_interventional_and_undirected(self):
        records = _golden("record-protection-fork.jsonl")
        assert records
        for record in records:
            assert record.fork and record.interventional
            assert record.directed is None

    def test_a_forced_spell_that_fizzled_holds_its_move_off_the_stack_and_an_ability_nothing(self):
        spell, ability = _golden("record-intervention-fizzled.jsonl")
        for record in (spell, ability):
            assert record.fork and record.interventional and record.link_id is None
            assert record.payload.outcome is ResolutionOutcome.FIZZLED
        assert [event.type for event in spell.payload.events] == [EventType.ZONE_CHANGE]
        assert ability.payload.events == ()

    def test_a_forced_you_may_declined_reads_declined(self):
        (record,) = _golden("record-intervention-declined.jsonl")
        assert record.payload.outcome is ResolutionOutcome.DECLINED
        assert [event.type for event in record.payload.events] == [EventType.ZONE_CHANGE]

    def test_a_probe_s_combat_record_holds_the_prevented_damage(self):
        (record,) = _golden("record-combat-probe-prevented.jsonl")
        assert record.fork and not record.interventional
        assert record.payload.probed_keyword == "trample"
        assert EventType.DAMAGE_PREVENTED in [event.type for event in record.payload.events]


class TestTheBoardExtrasPythonReads:
    """What a board holds beside front faces and permanents, read off the Java
    golden ``record-back-face-and-delayed-trigger.jsonl``: Graceful Reprieve's
    pending delayed trigger and a transformed Delver of Secrets, around a
    Lightning Bolt."""

    def test_a_pending_delayed_trigger_reads_with_its_line_and_remembered_card(self):
        from effects.domain.records import ProvenanceKey

        records = _golden("record-back-face-and-delayed-trigger.jsonl")
        pending = [record for record in records if record.state.global_.delayed_triggers]
        # Both of the bolt's halves; the reprieve's own were read before it.
        assert len(pending) == 2
        for record in pending:
            (trigger,) = record.state.global_.delayed_triggers
            assert trigger.id.startswith("D")
            assert trigger.source_lines == (
                ProvenanceKey("cardsfolder/g/graceful_reprieve.txt", 0, "spell", 0),)
            bears = next(e for e in record.state.entities if e.name == "Grizzly Bears")
            assert trigger.remembered == (bears.id,)
            assert trigger.duration == "end_of_turn"

    def test_a_permanent_on_its_back_face_reads_as_that_face(self):
        from effects.domain.state_snapshot import FaceState

        for record in _golden("record-back-face-and-delayed-trigger.jsonl"):
            (delver,) = [e for e in record.state.entities if e.name == "Insectile Aberration"]
            assert (delver.zone, delver.face_state) == ("battlefield", FaceState.BACK)
            assert delver.face == 1


class TestTheTurnActionsPythonReads:
    """A turn wrap's six turn-based actions, read off the Java golden
    ``record-turn-actions.jsonl``: each its own record, no acting line, the
    events it is made of."""

    def test_each_turn_action_reads_its_subkind_and_its_events(self):
        from effects.domain.records import RecordKind, TurnActionSubkind

        records = _golden("record-turn-actions.jsonl")
        assert [record.kind for record in records] == [RecordKind.TURN_ACTION] * 6
        assert [(record.subkind, tuple(event.type for event in record.payload.events))
                for record in records] == [
            (TurnActionSubkind.CLEANUP_DISCARD, (EventType.ZONE_CHANGE, EventType.CARD_DISCARDED)),
            (TurnActionSubkind.PHASING, (EventType.PHASED,)),
            (TurnActionSubkind.DAY_NIGHT, (EventType.DAY_NIGHT_CHANGED,)),
            (TurnActionSubkind.UNTAP, (EventType.UNTAPPED,)),
            (TurnActionSubkind.DRAW, (EventType.ZONE_CHANGE, EventType.CARD_DRAWN)),
            (TurnActionSubkind.LORE_COUNTER, (EventType.COUNTER_CHANGE,)),
        ]
        for record in records:
            assert record.ability is None and record.moment is None
            assert record.payload.records_withheld is None

    def test_the_draw_and_the_discard_name_who_acted(self):
        from effects.domain.records import TurnActionSubkind

        records = {record.subkind: record for record in _golden("record-turn-actions.jsonl")}
        draw = records[TurnActionSubkind.DRAW]
        discard = records[TurnActionSubkind.CLEANUP_DISCARD]
        assert draw.actor_player != discard.actor_player
        assert draw.payload.events[1].subjects == (draw.actor_player,)
        assert draw.payload.events[0].params["from_zone"] == "library"
        assert discard.payload.events[0].params["from_zone"] == "hand"
