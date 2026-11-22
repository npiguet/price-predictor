"""The smaller label rules of the per-entity and global targets (finding 10.7).

A tutor is the collector's ``mode: SearchedLibrary`` and nothing inferred from a
shuffle; a ``pt_change`` duration is unknown where one entity's events mix
durations; a turn event counts once per player it names; an enchant keyword an
Aura both gains and loses (Animate Dead) is neither.

Records are fixture records where the fixture holds the case. The fixture holds
no Animate Dead and no multi-player turn event (none of the five review pilots
holds one either), so those two are the events the collector writes for the
pilot's single-player case, with the shape widened by hand.
"""

from __future__ import annotations

import pytest

from effects.domain.aux_head_targets import global_targets
from effects.domain.effect_head_layout import (
    CREATED_KINDS,
    DURATIONS,
    GLOBAL_LAYOUT,
    LIBRARY_EVENTS,
    ZONE_OUTCOMES,
    GlobalField,
    HeadField,
)
from effects.domain.effect_targets import derive_targets, keyword_index
from effects.domain.event_schema import Event, EventType
from effects.domain.records import ResolutionPayload
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records, make_record

_RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"
#: Elfhame Sanctuary's search: its ``card_looked_at`` names ``SearchedLibrary``.
_ELFHAME_SANCTUARY = f"{_RUN}.0-mwqvqdyrb9c0.1"
#: Thada Adel's search of an opponent's library: reveal, exile, shuffle, and no
#: ``card_looked_at`` (a collector gap, reported with finding 10.7).
_THADA_ADEL = f"{_RUN}.0-mwqvqdyrb9c0.681"
#: Choking Miasma: E35 gets a permanent +1/+1 and an end-of-turn -2/-2; its
#: other five victims one end-of-turn -2/-2 each.
_CHOKING_MIASMA_MIXED = f"{_RUN}.4-mwqw395jccfe.1925"


@pytest.fixture(scope="module")
def records():
    wanted = {_ELFHAME_SANCTUARY, _THADA_ADEL, _CHOKING_MIASMA_MIXED}
    return {
        record.record_id: record
        for record in map(record_from_dict, gen3_fixture_records())
        if record.record_id in wanted
    }


def _library_flags(targets, subject: str) -> dict[str, float]:
    row = targets[subject].fields["library_events"]
    return dict(zip(LIBRARY_EVENTS, row))


class TestTheTutorFlag:
    def test_a_named_search_is_a_tutor(self, records):
        flags = _library_flags(derive_targets(records[_ELFHAME_SANCTUARY]), "P1")
        assert flags["tutor"] == 1.0

    def test_a_shuffle_after_cards_left_the_library_is_no_tutor(self, records):
        flags = _library_flags(derive_targets(records[_THADA_ADEL]), "P1")
        assert flags["tutor"] == 0.0
        assert flags["shuffle"] == 1.0


class TestThePtDuration:
    def test_an_entity_whose_changes_mix_durations_has_none(self, records):
        entry = derive_targets(records[_CHOKING_MIASMA_MIXED])["E35"]
        assert HeadField.PT_DURATION not in entry.fields
        assert HeadField.PT_DURATION in entry.unknown
        assert entry.fields[HeadField.POWER_DELTA] == -1.0

    def test_an_entity_with_one_duration_keeps_it(self, records):
        entry = derive_targets(records[_CHOKING_MIASMA_MIXED])["E29"]
        assert entry.fields[HeadField.PT_DURATION] == DURATIONS.index("end_of_turn")


def _resolution(*events) -> object:
    return make_record(payload=ResolutionPayload(events=tuple(events)))


class TestTheEnchantKeyword:
    """Animate Dead's trigger, as the collector writes it (pilot record
    ``329a67b0-...0-mwqgfmpvi1xa.10644``): the Aura gains one enchant
    restriction and loses another, and both spellings fold to ``enchant``."""

    def _targets(self, *extra):
        return derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E91",), duration="permanent",
                  params={"keywords": ["enchant:creature.isremembered:creature put onto the "
                                       "battlefield with cardname"],
                          "removed": False, "cause": "E91"}),
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E91",), duration="permanent",
                  params={"keywords": ["enchant:creature.inzonegraveyard:creature card in a "
                                       "graveyard"],
                          "removed": True, "cause": "E91"}),
            *extra,
        ))

    def test_gained_and_lost_together_is_neither(self):
        fields = self._targets()["E91"].fields
        column = keyword_index("enchant")
        assert fields[HeadField.KEYWORDS_GAINED][column] == 0.0
        assert fields[HeadField.KEYWORDS_LOST][column] == 0.0

    def test_another_keyword_gained_is_still_gained(self):
        flying = Event(type=EventType.KEYWORD_CHANGE, subjects=("E91",), duration="permanent",
                       params={"keywords": ["Flying"], "removed": False, "cause": "E91"})
        fields = self._targets(flying)["E91"].fields
        assert fields[HeadField.KEYWORDS_GAINED][keyword_index("flying")] == 1.0

    def test_one_enchant_gained_alone_is_gained(self):
        fields = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E91",), duration="permanent",
                  params={"keywords": ["enchant:creature"], "removed": False}),
        ))["E91"].fields
        assert fields[HeadField.KEYWORDS_GAINED][keyword_index("enchant")] == 1.0


class TestTheTurnStructureCount:
    def _counts(self, event_type, subjects, count):
        row = global_targets(_resolution(
            Event(type=event_type, subjects=subjects, params={"count": count}),
        ))
        values, _ = row
        start, end = GLOBAL_LAYOUT.target_slices[GlobalField.TURN_STRUCTURE_CHANGE]
        return dict(zip(
            ("extra_turns", "turns_skipped", "extra_phases", "phases_skipped"),
            values[start:end],
        ))

    def test_a_turn_added_for_two_players_counts_both(self):
        assert self._counts(EventType.TURN_ADDED, ("P0", "P1"), 1)["extra_turns"] == 2.0

    def test_a_turn_added_for_one_player_counts_its_count(self):
        assert self._counts(EventType.TURN_ADDED, ("P0",), 2)["extra_turns"] == 2.0

    def test_an_event_naming_no_player_counts_once(self):
        assert self._counts(EventType.PHASE_ADDED, (), 1)["extra_phases"] == 1.0


class TestNothingNeverWrittenIsAnOutput:
    def test_no_emblem_group_kind(self):
        assert "emblem" not in CREATED_KINDS

    def test_no_transformed_zone_outcome(self):
        assert "transformed" not in ZONE_OUTCOMES
