"""Where each event leaves an entity: ``zone_outcome`` and ``face_after``,
from a destroy, a sacrifice, a zone change, a counter, a phase-out, a blink
and a move into a library."""

from __future__ import annotations

import pytest

from effects.domain.effect_head_layout import LIBRARY_EVENTS, ZONE_OUTCOMES
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    ResolutionPayload,
)
from tests.unit.effects.domain.effect_targets_support import (
    _entity,
    _on_board,
    _resolution,
    _spec,
    _zone,
)


class TestZoneOutcomes:
    def test_a_destroy_reads_as_died(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1",)),
        ))
        assert targets["E1"].fields["zone_outcome"] == ZONE_OUTCOMES.index("died")

    def test_a_sacrifice_also_reads_as_died(self):
        """Different game actions, the same destination."""
        targets = derive_targets(_resolution(
            Event(type=EventType.SACRIFICED, subjects=("E1",)),
        ))
        assert targets["E1"].fields["zone_outcome"] == ZONE_OUTCOMES.index("died")

    def test_a_zone_change_reads_its_destination(self):
        for destination, outcome in (
            ("exile", "exiled"), ("hand", "to_hand"), ("graveyard", "died"),
        ):
            targets = derive_targets(_resolution(
                Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                      params={"to_zone": destination}),
            ))
            assert targets["E1"].fields["zone_outcome"] == (
                ZONE_OUTCOMES.index(outcome)
            )

    def test_a_countered_spell_reads_as_countered(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.SPELL_COUNTERED, subjects=("S1",)),
        ))
        assert targets["S1"].fields["zone_outcome"] == (
            ZONE_OUTCOMES.index("countered")
        )

    def test_a_phase_out_reads_as_phased_out(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.PHASED, subjects=("E1",), params={"out": True}),
        ))
        assert targets["E1"].fields["zone_outcome"] == (
            ZONE_OUTCOMES.index("phased_out")
        )


class TestZoneOutcomeExtensions:
    def test_a_permanent_spell_resolving_entered_with_the_face_it_was_cast_with(self):
        from effects.domain.input_vocabularies import FACE_STATES

        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "stack", "to_zone": "battlefield"}),
        )), _entity("E1", zone="stack", face_state="face_down")))
        assert targets["E1"].fields["zone_outcome"] == _zone("entered")
        assert targets["E1"].fields["face_after"] == FACE_STATES.index("face_down")

    def test_leaving_and_returning_in_one_record_is_a_blink(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "battlefield", "to_zone": "exile"}),
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "exile", "to_zone": "battlefield"}),
        ))
        assert targets["E1"].fields["zone_outcome"] == _zone("blinked")

    def test_a_card_put_into_a_library_then_shuffled_was_shuffled_in(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "battlefield", "to_zone": "library"}),
            Event(type=EventType.LIBRARY_SHUFFLED, subjects=("P0",)),
        )), _entity("E1", owner="P0")))
        assert targets["E1"].fields["zone_outcome"] == _zone("shuffled_into_library")
        assert targets["P0"].fields["library_events"][LIBRARY_EVENTS.index("shuffle")] == 1.0

    def test_a_library_position_says_top_or_bottom(self):
        for position, outcome in ((0, "library_top"), (30, "library_bottom"),
                                  (-1, "library_bottom")):
            targets = derive_targets(_on_board(ResolutionPayload(events=(
                Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                      params={"from_zone": "battlefield", "to_zone": "library",
                              "library_position": position}),
            )), _entity("E1", owner="P0")))
            assert targets["E1"].fields["zone_outcome"] == _zone(outcome), position

    def test_a_library_move_the_record_cannot_place_is_unknown(self):
        entry = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "battlefield", "to_zone": "library"}),
        )), _entity("E1", owner="P0")))["E1"]
        assert entry.affected
        assert entry.supervised(_spec("zone_outcome")) is None

    @pytest.mark.parametrize("destination", ["stack", "sideboard", "merged", "ante", "nowhere"])
    def test_a_move_to_a_zone_no_outcome_names_leaves_the_outcome_unknown(self, destination):
        """The card moved, so it did not stay; no outcome names where it went
        (a card cast during a resolution goes to the stack), so the field is
        unknown rather than "stayed"."""
        entry = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "hand", "to_zone": destination}),
        ))["E1"]
        assert entry.affected
        assert "zone_outcome" not in entry.fields
        assert entry.supervised(_spec("zone_outcome")) is None

    def test_a_later_event_still_names_the_outcome_after_an_unnamed_move(self):
        entry = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",), params={"to_zone": "stack"}),
            Event(type=EventType.SPELL_COUNTERED, subjects=("E1",)),
        ))["E1"]
        assert entry.supervised(_spec("zone_outcome")) == _zone("countered")

    def test_a_move_to_no_zone_ceased_to_exist(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",), params={"to_zone": "none"}),
        ))
        assert targets["E1"].fields["zone_outcome"] == _zone("ceased_to_exist")

    def test_phasing_in_and_out_are_distinct(self):
        from effects.domain.input_vocabularies import DESIGNATIONS

        out = derive_targets(_resolution(
            Event(type=EventType.PHASED, subjects=("E1",), params={"out": True}),
        ))["E1"].fields
        back = derive_targets(_resolution(
            Event(type=EventType.PHASED, subjects=("E1",), params={"out": False}),
        ))["E1"].fields
        assert out["zone_outcome"] == _zone("phased_out")
        assert back["zone_outcome"] == _zone("phased_in")
        phased = DESIGNATIONS.index("phased_out")
        assert out["designations_gained"][phased] == 1.0
        assert back["designations_lost"][phased] == 1.0
