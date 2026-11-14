"""Per-entity targets of what the WP7 wire adds (contract ``record-schema-delta.md``):
a rewrite's set-aside ability's run, a mana record's own effects, and the
game-end marker. Read off the Java goldens, edited where the goldens hold no
such case.
"""

from __future__ import annotations

from dataclasses import replace

from effects.domain.aux_head_targets import global_targets, supervises_created_objects
from effects.domain.effect_head_layout import HeadField
from effects.domain.effect_targets import derive_targets
from effects.domain.records import EffectRecord, RewritePayload
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import java_golden_records


def _golden(predicate) -> EffectRecord:
    return record_from_dict(next(raw for raw in java_golden_records() if predicate(raw)))


def _buffered() -> EffectRecord:
    return _golden(lambda raw: raw["kind"] == "rewrite" and "buffered" in raw["payload"])


class TestASetAsideAbilitysRun:
    def test_its_events_are_the_rewrite_record_s_outcome(self):
        """Swans of Bryn Argoll's draws: no resolution half holds them."""
        targets = derive_targets(_buffered())
        assert targets["P0"].fields[HeadField.CARDS_DRAWN] == 2
        assert targets["P0"].affected and targets["E1"].affected and targets["E2"].affected

    def test_a_run_that_never_came_trains_nobody_affected(self):
        """The set-aside ability never ran: the replacement's outcome is
        known, and it touched nobody."""
        record = _buffered()
        never = replace(record, payload=replace(
            record.payload, buffered_ran=False, buffered_events=()))
        targets = derive_targets(never)
        assert dict(targets) == {} and targets.gated

    def test_a_record_whose_run_another_record_took_trains_no_gate(self):
        """The run happened and its events are another record's: who it
        touched is not this record's to say, so no entity is trained as
        unaffected (the holder of the same run says P0 drew two)."""
        record = _buffered()
        other = replace(record, payload=replace(record.payload, buffered_events=()))
        targets = derive_targets(other)
        assert other.payload.buffered_ran and dict(targets) == {}
        assert not targets.gated
        assert derive_targets(record).gated

    def test_a_non_holder_with_its_own_outgoing_event_supervises_that_alone(self):
        record = _buffered()
        event = record.payload.incoming
        other = replace(record, payload=replace(
            record.payload, buffered_events=(), outgoing=event))
        targets = derive_targets(other)
        assert not targets.gated
        assert set(targets) == set(event.subjects)
        assert all(targets[subject].affected for subject in event.subjects)

    def test_the_record_holding_the_run_supervises_the_record_heads(self):
        record = _buffered()
        assert supervises_created_objects(record) and global_targets(record) is not None
        other = replace(record, payload=replace(record.payload, buffered_events=()))
        assert not supervises_created_objects(other) and global_targets(other) is None


class TestAnUnknownIncomingEvent:
    def test_no_in_place_edit_is_supervised(self):
        record = _golden(lambda raw: raw["kind"] == "rewrite"
                         and raw["payload"].get("incoming_unknown"))
        assert isinstance(record.payload, RewritePayload)
        targets = derive_targets(record)
        assert dict(targets) == {}
        assert supervises_created_objects(record) is False

    def test_what_changed_is_unknown_so_no_gate_trains(self):
        """Its known-incoming twin, the same replacement, trains the gate: an
        unknown incoming event must not teach "nobody affected" instead."""
        record = _golden(lambda raw: raw["kind"] == "rewrite"
                         and raw["payload"].get("incoming_unknown"))
        assert not derive_targets(record).gated
        known = replace(record, payload=replace(record.payload, incoming_unknown=False))
        assert derive_targets(known).gated


class TestAManaRecordsOwnEffects:
    def test_a_pain_land_s_damage_to_its_controller_is_its_outcome(self):
        record = _golden(lambda raw: raw.get("moment") == "resolution" and any(
            e["type"] == "mana_produced" for e in raw["payload"]["events"])
            and any(e["type"] == "damage_dealt" for e in raw["payload"]["events"]))
        targets = derive_targets(record)
        assert targets["P0"].fields[HeadField.DAMAGE_TAKEN] == 1
        assert targets["P0"].fields[HeadField.LIFE_DELTA] == -1
        assert targets["E1"].fields[HeadField.TAP_STATE] == 1


class TestTheGameEndMarker:
    def test_it_names_nobody_and_supervises_nothing(self):
        marker = _golden(lambda raw: raw.get("subkind") == "game_end")
        targets = derive_targets(marker)
        assert dict(targets) == {} and not targets.unsupervised
        assert not supervises_created_objects(marker) and global_targets(marker) is None
