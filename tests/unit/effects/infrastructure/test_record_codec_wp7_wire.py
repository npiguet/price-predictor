"""The gen-3 wire as the connector writes it after WP7 (contract
``record-schema-delta.md``): the game-end marker, a rewrite whose incoming
event is unknown or whose ability was set aside, and a blocker a lure
requires elsewhere. Each member decodes into its field and writes back where
the collector wrote it.

Read off the Java goldens where they carry the value, hand-built where none
does.
"""

from __future__ import annotations

import json

import pytest

from effects.domain.event_schema import EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    PlayabilityBlockersPayload,
    RecordKind,
    RewritePayload,
    RewriteResult,
    TurnActionPayload,
    TurnActionSubkind,
    is_game_end,
)
from effects.infrastructure.record_codec import format_record_line, record_from_dict
from tests.unit.effects.builders import java_golden_records

_SWANS = ProvenanceKey("cardsfolder/s/swans_of_bryn_argoll.txt", 0, "replacement", 0)
_PACIFISM = ProvenanceKey("cardsfolder/p/pacifism.txt", 0, "static", 0)


def _golden(predicate) -> dict:
    return next(raw for raw in java_golden_records() if predicate(raw))


def _round_trips(raw: dict) -> None:
    """The payload writes back member for member, in the collector's order."""
    written = json.loads(format_record_line(record_from_dict(raw)))
    assert (written["subkind"], written["payload"]) == (raw["subkind"], raw["payload"])
    assert json.dumps(written["payload"]) == json.dumps(raw["payload"])


class TestTheGameEndMarker:
    @staticmethod
    def _marker() -> dict:
        return _golden(lambda raw: raw.get("subkind") == "game_end")

    def test_the_marker_reads_with_the_records_the_cap_withheld(self):
        record = record_from_dict(self._marker())
        assert (record.kind, record.subkind) == (RecordKind.TURN_ACTION, TurnActionSubkind.GAME_END)
        assert record.payload == TurnActionPayload(events=(), records_withheld=0)
        assert is_game_end(record)

    def test_the_marker_writes_back_as_the_collector_wrote_it(self):
        _round_trips(self._marker())

    def test_another_turn_action_writes_no_withheld_count(self):
        raw = dict(self._marker(), subkind="draw", payload={"events": []})
        record = record_from_dict(raw)
        assert record.payload.records_withheld is None and not is_game_end(record)
        assert json.loads(format_record_line(record))["payload"] == {"events": []}

    def test_a_withheld_count_reads_as_written(self):
        raw = dict(self._marker(), payload={"events": [], "records_withheld": 37})
        assert record_from_dict(raw).payload.records_withheld == 37
        _round_trips(raw)


class TestAnUnknownIncomingEvent:
    @staticmethod
    def _raw() -> dict:
        return _golden(lambda raw: raw["kind"] == "rewrite"
                       and raw["payload"].get("incoming_unknown"))

    def test_it_reads_as_unknown_and_says_nothing_of_an_in_place_edit(self):
        payload = record_from_dict(self._raw()).payload
        assert isinstance(payload, RewritePayload)
        assert payload.incoming_unknown and payload.outgoing is None
        assert payload.arrived is None
        assert payload.departed is not None and payload.departed.params["amount"] == 3
        assert payload.rewritten is None
        assert payload.happened_events() == ()

    def test_a_known_incoming_event_is_what_arrived(self):
        raw = _golden(lambda raw: raw["kind"] == "rewrite" and raw["payload"]["outgoing"])
        payload = record_from_dict(raw).payload
        assert payload.arrived is payload.incoming and payload.departed is payload.outgoing
        assert payload.rewritten is True

    def test_it_writes_back_as_the_collector_wrote_it(self):
        _round_trips(self._raw())


class TestABufferedAbility:
    @staticmethod
    def _raw() -> dict:
        return _golden(lambda raw: raw["kind"] == "rewrite" and "buffered" in raw["payload"])

    def test_the_set_aside_ability_and_its_run_read_into_their_fields(self):
        payload = record_from_dict(self._raw()).payload
        assert payload.replaced_by == () and payload.buffered == (_SWANS,)
        assert payload.buffered_ran is True
        assert [e.type for e in payload.buffered_events] == [
            EventType.ZONE_CHANGE, EventType.CARD_DRAWN] * 2
        assert {e.attributed_to for e in payload.buffered_events} == {"root"}
        assert payload.result is RewriteResult.REPLACED

    def test_its_run_is_what_the_record_saw_happen_and_its_keys_are_named(self):
        payload = record_from_dict(self._raw()).payload
        assert payload.happened_events() == payload.buffered_events
        assert payload.listed_events() == ()
        assert _SWANS in tuple(payload.provenance_keys())

    def test_a_record_with_no_ability_set_aside_writes_none_of_the_members(self):
        raw = _golden(lambda raw: raw["kind"] == "rewrite" and "buffered" not in raw["payload"])
        payload = record_from_dict(raw).payload
        assert (payload.buffered, payload.buffered_ran, payload.buffered_events) == (None, None, ())
        assert "buffered" not in json.loads(format_record_line(record_from_dict(raw)))["payload"]

    def test_a_run_that_never_came_and_a_run_another_record_took_write_back(self):
        raw = self._raw()
        for ran in (False, True):
            changed = dict(raw, payload=dict(raw["payload"], buffered_ran=ran, buffered_events=[]))
            payload = record_from_dict(changed).payload
            assert (payload.buffered_ran, payload.happened_events()) == (ran, ())
            _round_trips(changed)

    def test_it_writes_back_as_the_collector_wrote_it(self):
        _round_trips(self._raw())


class TestAForbiddenBlocker:
    @staticmethod
    def _raw() -> dict:
        return _golden(lambda raw: raw.get("subkind") == "blockers" and any(
            entry.get("required_elsewhere") for entry in raw["payload"]["forbidden"]))

    def test_a_lure_bound_blocker_and_a_pacifism_blocker_read_apart(self):
        payload = record_from_dict(self._raw()).payload
        assert isinstance(payload, PlayabilityBlockersPayload)
        by_entity = {entry.entity: entry for entry in payload.forbidden}
        assert (by_entity["E8"].responsible_static, by_entity["E8"].required_elsewhere) == (
            (_PACIFISM,), False)
        assert (by_entity["E11"].responsible_static, by_entity["E11"].required_elsewhere) == (
            (), True)

    def test_its_own_keyword_line_and_an_unkeyed_static_read_apart(self):
        """Grakk's own "can't attack or block" line, an engine effect's
        unkeyable static, and Pacifism, another card's static."""
        by_entity = {e.entity: e for e in record_from_dict(self._raw()).payload.forbidden}
        grakk, stopped, pacified = by_entity["E12"], by_entity["E13"], by_entity["E8"]
        assert grakk.responsible_static[0].script_file == "cardsfolder/g/grakk_the_pacifist.txt"
        assert (grakk.responsible_own, grakk.responsible_unresolved) == (True, None)
        assert (stopped.responsible_static, stopped.responsible_own,
                stopped.responsible_unresolved) == ((), False, "engine_effect")
        assert (pacified.responsible_own, pacified.responsible_unresolved) == (False, None)
        assert all(e.explained for e in (grakk, stopped, pacified, by_entity["E11"]))

    def test_an_unresolved_reason_outside_the_vocabulary_is_refused(self):
        raw = self._raw()
        entries = [dict(e, responsible_unresolved="made_up") if e["entity"] == "E13" else e
                   for e in raw["payload"]["forbidden"]]
        with pytest.raises(ValueError, match="responsible_unresolved"):
            record_from_dict(dict(raw, payload=dict(raw["payload"], forbidden=entries)))

    def test_it_writes_back_as_the_collector_wrote_it(self):
        _round_trips(self._raw())
