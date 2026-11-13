"""The gen-3 wire as the connector writes it after WP3 and WP4 (contract
``record-schema-delta.md``): every new member decodes into its field and
writes back where the collector wrote it.

Read off the Java goldens where they carry the value, hand-built where none
does.
"""

from __future__ import annotations

import json

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.records import ContinuousPayload, ResolutionPayload
from effects.infrastructure.record_codec import format_record_line, record_from_dict
from tests.unit.effects.builders import java_golden_records


def _golden(predicate) -> dict:
    return next(raw for raw in java_golden_records() if predicate(raw))


def _effect_half_with(member: str, value) -> dict:
    return _golden(lambda raw: raw.get("moment") == "resolution"
                   and raw["state"]["refs"].get(member) == value)


class TestTheRefs:
    def test_targets_at_cast_reads_beside_targets(self):
        raw = _effect_half_with("targets_at_cast", ["L5"])
        refs = record_from_dict(raw).state.refs
        assert (refs.targets, refs.targets_at_cast, refs.targets_on_stack) == ((), ("L5",), ())

    def test_an_empty_list_and_null_stay_apart(self):
        """``[]``: no targets as cast; null: unknown (no hook, another block)."""
        empty = record_from_dict(_effect_half_with("targets_at_cast", [])).state.refs
        cost = record_from_dict(_golden(lambda raw: raw.get("moment") == "activation"
                                        and "targets_on_stack" in raw["state"]["refs"]))
        assert empty.targets_at_cast == ()
        assert cost.state.refs.targets_at_cast is None

    def test_the_refs_write_back_in_the_collector_s_order(self):
        raw = _effect_half_with("targets_at_cast", ["L7", "P1"])
        line = format_record_line(record_from_dict(raw))
        assert list(json.loads(line)["state"]["refs"]) == list(raw["state"]["refs"])

    def test_a_cast_time_target_gone_since_is_a_last_known_entity_with_no_zone(self):
        raw = _effect_half_with("targets_at_cast", ["L5"])
        entity = record_from_dict(raw).state.entity("L5")
        assert (entity.role, entity.zone, entity.order_rank) == ("last_known", None, None)


class TestTheHiddenKeywords:
    def test_a_card_s_hidden_keywords_read_apart_from_its_keywords(self):
        raw = _golden(lambda raw: any(
            (e.get("granted_temporary") or {}).get("hidden_keywords")
            for e in raw["state"]["entities"]))
        granted = next(e.granted_temporary for e in record_from_dict(raw).state.entities
                       if e.granted_temporary.hidden_keywords)
        assert granted.hidden_keywords[0].startswith("CARDNAME ")
        assert not set(granted.hidden_keywords) & set(granted.keywords)

    def test_a_hidden_keyword_change_decodes(self):
        raw = _golden(lambda raw: any(event.get("params", {}).get("hidden")
                                      for event in raw["payload"].get("events") or ()))
        payload = record_from_dict(raw).payload
        assert isinstance(payload, ResolutionPayload)
        (change,) = [e for e in payload.events if e.type is EventType.KEYWORD_CHANGE]
        assert change.params["hidden"] is True

    def test_a_contribution_s_abilities_and_hidden_keywords_read(self):
        raws = [raw for raw in java_golden_records()
                if raw["kind"] == "continuous"
                and "abilities" in raw["payload"]["contributions"][0]]
        contributions = [c for raw in raws
                         for c in record_from_dict(raw).payload.contributions]
        assert any(c.hidden_keywords == ("CARDNAME can only attack alone.",)
                   for c in contributions)
        removed = next(c for c in contributions if c.keywords[:1] == ("=",))
        assert removed.abilities[0].startswith("-(cardsfolder/")
        assert removed.keyword_changes.sets_outright

    def test_a_contribution_writes_back_as_the_collector_wrote_it(self):
        for raw in java_golden_records():
            if raw["kind"] == "continuous":
                assert isinstance(record_from_dict(raw).payload, ContinuousPayload)
                written = json.loads(format_record_line(record_from_dict(raw)))
                assert written["payload"] == raw["payload"]


class TestTheNewEvents:
    @pytest.mark.parametrize(("event_type", "params"), [
        (EventType.NAME_CHANGE, {"name": "Serra Angel"}),
        (EventType.MANA_COST_CHANGE, {"mana_cost": {"generic": 3, "W": 2}, "mana_value": 5}),
        (EventType.MANA_COST_CHANGE, {"mana_value": 0}),
        (EventType.KEYWORD_CHANGE, {"keywords": ["CARDNAME can't block."], "removed": False,
                                    "hidden": True}),
        (EventType.DAMAGE_DEALT, {"amount": 2, "combat": False, "damage_type": "m1m1_counters"}),
    ])
    def test_each_decodes_and_round_trips(self, event_type, params):
        data = {"type": event_type.value, "subjects": ["E1"], "params": params,
                "duration": None, "attributed_to": "root"}
        event = Event.from_dict(data)
        assert event.as_dict() == data
