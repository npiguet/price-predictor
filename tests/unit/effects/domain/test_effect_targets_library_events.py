"""``library_events``' scry, surveil, tutor and shuffle bits, by the
collector's named mode or the events around a look."""

from __future__ import annotations

import pytest

from effects.domain.effect_head_layout import LIBRARY_EVENTS
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from tests.unit.effects.domain.effect_targets_support import (
    _resolution,
)


class TestLibraryEventExtensions:
    def _looked(self, *extra):
        return derive_targets(_resolution(
            Event(type=EventType.CARD_LOOKED_AT, subjects=("P0",), params={"count": 2}),
            Event(type=EventType.LIBRARY_REORDERED, subjects=("P0",), params={"count": 2}),
            *extra,
        ))["P0"].fields["library_events"]

    def test_a_scry_sets_scry_alone(self):
        flags = self._looked()
        assert flags[LIBRARY_EVENTS.index("scry")] == 1.0
        assert flags[LIBRARY_EVENTS.index("surveil")] == 0.0

    def test_a_surveil_that_mills_sets_surveil_not_scry(self):
        flags = self._looked(
            Event(type=EventType.CARD_MILLED, subjects=("P0",), params={"count": 1}),
        )
        assert flags[LIBRARY_EVENTS.index("surveil")] == 1.0
        assert flags[LIBRARY_EVENTS.index("scry")] == 0.0

    def test_a_search_then_shuffle_is_a_tutor_only_by_its_named_mode(self):
        """Elfhame Sanctuary's events, as the collector writes them."""
        flags = derive_targets(_resolution(
            Event(type=EventType.CARD_LOOKED_AT, subjects=("P0",),
                  params={"from_zone": "library", "mode": "SearchedLibrary", "cause": "E138"}),
            Event(type=EventType.ZONE_CHANGE, subjects=("E9",),
                  params={"from_zone": "library", "to_zone": "hand", "cause": "E138"}),
            Event(type=EventType.LIBRARY_SHUFFLED, subjects=("P0",), params={"cause": "E138"}),
        ))["P0"].fields["library_events"]
        assert flags[LIBRARY_EVENTS.index("tutor")] == 1.0
        assert flags[LIBRARY_EVENTS.index("shuffle")] == 1.0

    def test_a_shuffle_after_a_card_left_the_library_unnamed_is_no_tutor(self):
        flags = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E9",),
                  params={"from_zone": "library", "to_zone": "battlefield"}),
            Event(type=EventType.LIBRARY_SHUFFLED, subjects=("P0",)),
        ))["P0"].fields["library_events"]
        assert flags[LIBRARY_EVENTS.index("tutor")] == 0.0
        assert flags[LIBRARY_EVENTS.index("shuffle")] == 1.0

    def test_a_shuffle_alone_is_no_tutor(self):
        flags = derive_targets(_resolution(
            Event(type=EventType.LIBRARY_SHUFFLED, subjects=("P0",)),
        ))["P0"].fields["library_events"]
        assert flags[LIBRARY_EVENTS.index("tutor")] == 0.0

    @pytest.mark.parametrize(("mode", "kind"), [
        ("Scry", "scry"), ("Surveil", "surveil"), ("SearchedLibrary", "tutor"),
    ])
    def test_a_named_mode_decides_the_library_event(self, mode, kind):
        flags = derive_targets(_resolution(
            Event(type=EventType.CARD_LOOKED_AT, subjects=("P0",), params={"mode": mode}),
        ))["P0"].fields["library_events"]
        assert [name for i, name in enumerate(LIBRARY_EVENTS) if flags[i] == 1.0] == [kind]

    def test_a_surveil_keeping_every_card_on_top_is_still_a_surveil(self):
        flags = self._looked()
        named = derive_targets(_resolution(
            Event(type=EventType.CARD_LOOKED_AT, subjects=("P0",),
                  params={"count": 2, "mode": "Surveil"}),
        ))["P0"].fields["library_events"]
        assert flags[LIBRARY_EVENTS.index("scry")] == 1.0
        assert named[LIBRARY_EVENTS.index("surveil")] == 1.0
        assert named[LIBRARY_EVENTS.index("scry")] == 0.0
