"""``scan_output_vocabularies.py``, ``closed_lists`` and the module the scan writes
(T119, T121; root spec section 11.2, section 11.3)."""

from __future__ import annotations

import json

import scan_output_vocabularies as sov

from effects.domain.closed_lists import (
    keywords_named,
    ordered_keywords,
    subtypes_named,
    subtypes_of,
    type_tokens,
)
from effects.domain.effect_head_input import OVERLAY_KEYWORDS
from effects.domain.event_schema import Event, EventType
from effects.domain.forge_vocabularies import SUBTYPES
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import ContinuousPayload, Contribution, RecordKind, ResolutionPayload
from tests.unit.effects.builders import FIXTURE_KEYWORD_DEFINITIONS, make_record

_KNOWN = frozenset({"flying", "toxic", "ward", "enchant", "protection"})


def _resolution(*events):
    return make_record(payload=ResolutionPayload(events=tuple(events)))


class TestReadingOneRecord:
    def test_a_type_line_splits_into_tokens(self):
        assert type_tokens("Legendary Creature — Elf Warrior") == (
            "legendary", "creature", "elf", "warrior")
        assert subtypes_of(type_tokens("Legendary Creature - Elf Warrior")) == ("elf", "warrior")

    def test_subtypes_come_from_type_changes_tokens_and_contributions(self):
        resolution = _resolution(
            Event(type=EventType.TYPE_CHANGE, subjects=("E1",),
                  params={"types_added": ["creature", "golem", "legendary"],
                          "types_removed": ["human"]}),
            Event(type=EventType.TOKEN_CREATED, params={
                "count": 1, "characteristics": {"types": "Artifact - Treasure"}}),
        )
        continuous = make_record(
            RecordKind.CONTINUOUS,
            payload=ContinuousPayload(contributions=(
                Contribution(entity="E1", types=("=", "zombie", "-all-creature-types")),
            )),
        )
        assert subtypes_named(resolution) == {"golem", "human", "treasure"}
        assert subtypes_named(continuous) == {"zombie"}

    def test_keywords_are_forge_keywords_in_the_heads_spelling(self):
        record = _resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["ward:2", "protection from black", "toxic:1",
                                       "prevent all combat damage that would be dealt"]}),
            Event(type=EventType.TOKEN_CREATED, params={
                "count": 1, "characteristics": {"keywords": ["flying"]}}),
        )
        assert keywords_named(record, _KNOWN) == {"ward", "protection", "toxic", "flying"}

    def test_the_overlay_keywords_lead_in_their_own_order(self):
        assert ordered_keywords({"toxic", "flying", "bushido"}) == (
            *OVERLAY_KEYWORDS, "bushido", "toxic")


class TestTheScan:
    def test_a_survey_orders_subtypes_as_forge_does(self):
        records = [_resolution(Event(
            type=EventType.TYPE_CHANGE, subjects=("E1",),
            params={"types_added": ["zombie", "plains"], "types_removed": []},
        ))]
        lists = sov.survey(records, _KNOWN)
        assert lists["SUBTYPES_CHANGED"] == tuple(
            sorted(("zombie", "plains"), key=SUBTYPES.index))
        assert lists["KEYWORDS_CHANGED"] == OVERLAY_KEYWORDS

    def test_the_written_module_reads_back(self, tmp_path):
        lists = {"SUBTYPES_CHANGED": ("goblin",), "KEYWORDS_CHANGED": ("flying",)}
        target = tmp_path / "vocabularies.py"
        target.write_text(sov.render(lists, ["somewhere"]), encoding="utf-8")
        namespace: dict = {}
        exec(target.read_text(encoding="utf-8"), namespace)
        assert {name: namespace[name] for name in lists} == lists

    def test_forge_keywords_are_read_in_the_normalized_spelling(self, tmp_path):
        definitions = tmp_path / "keywords.json"
        definitions.write_text(json.dumps({"First Strike": {}, "Ward": {}}), encoding="utf-8")
        assert sov.forge_keywords(definitions) == {"first_strike", "ward"}


class TestTheCheckedInLists:
    def test_they_hold_every_name_the_gen3_fixture_sets(self):
        """The fixture is cut from the pilots the module was surveyed from."""
        from effects.infrastructure.shard_io import read_shard
        from tests.unit.effects.builders import GEN3_RECORDS

        known = sov.forge_keywords(FIXTURE_KEYWORD_DEFINITIONS)
        for record in read_shard(GEN3_RECORDS):
            assert subtypes_named(record) <= set(SUBTYPES_CHANGED), record.record_id
            assert keywords_named(record, known) <= set(KEYWORDS_CHANGED), record.record_id

    def test_they_are_ordered_as_the_scan_writes_them(self):
        assert KEYWORDS_CHANGED == ordered_keywords(KEYWORDS_CHANGED)
        assert list(SUBTYPES_CHANGED) == sorted(SUBTYPES_CHANGED, key=SUBTYPES.index)
