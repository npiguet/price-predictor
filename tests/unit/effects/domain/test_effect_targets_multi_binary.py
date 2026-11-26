"""The multi-binary fields: one bit per keyword, card type or library event
an event gains or loses."""

from __future__ import annotations

from effects.domain.effect_head_layout import LIBRARY_EVENTS, OVERLAY_KEYWORDS
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.input_vocabularies import CARD_TYPES
from effects.domain.records import (
    ContinuousPayload,
    Contribution,
    RecordKind,
)
from tests.unit.effects.domain.effect_targets_support import (
    _record,
    _resolution,
)


class TestMultiBinaryFields:
    def test_a_gained_keyword_sets_its_bit(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["flying"], "removed": False}),
        ))
        vector = targets["E1"].fields["keywords_gained"]
        assert vector[OVERLAY_KEYWORDS.index("flying")] == 1.0
        assert sum(vector) == 1.0

    def test_a_removed_keyword_goes_to_the_other_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["flying"], "removed": True}),
        ))
        assert "keywords_lost" in targets["E1"].fields
        assert "keywords_gained" not in targets["E1"].fields

    def test_a_multi_word_keyword_matches_its_token_form(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["first strike"]}),
        ))
        vector = targets["E1"].fields["keywords_gained"]
        assert vector[OVERLAY_KEYWORDS.index("first_strike")] == 1.0

    def test_a_keyword_with_a_value_sets_its_bit(self):
        """``StatDiffer`` writes the keyword as printed, value included."""
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["ward:2", "annihilator:2"]}),
        ))
        vector = targets["E1"].fields["keywords_gained"]
        assert vector[OVERLAY_KEYWORDS.index("ward")] == 1.0
        assert vector[OVERLAY_KEYWORDS.index("annihilator")] == 1.0

    def test_protection_from_a_quality_sets_the_protection_bit(self):
        """The overlay has one protection slot; the input layout folds
        ``protection_from_red`` onto it, and the target has to agree."""
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["protection from blue"]}),
        ))
        vector = targets["E1"].fields["keywords_gained"]
        assert vector[OVERLAY_KEYWORDS.index("protection")] == 1.0

    def test_a_continuous_grant_with_a_value_sets_its_bit(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", keywords=("ward:2", "-protection_from_red")),
            )),
            kind=RecordKind.CONTINUOUS,
        )
        fields = derive_targets(record)["E1"].fields
        assert fields["keywords_gained"][OVERLAY_KEYWORDS.index("ward")] == 1.0
        assert fields["keywords_lost"][OVERLAY_KEYWORDS.index("protection")] == 1.0

    def test_an_untracked_keyword_sets_no_bit(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["cascade"]}),
        ))
        assert sum(targets["E1"].fields.get("keywords_gained", [0])) == 0

    def test_a_type_change_sets_the_right_bits(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.TYPE_CHANGE, subjects=("E1",),
                  params={"types_added": ["Creature"], "types_removed": ["Land"]}),
        ))
        gained = targets["E1"].fields["types_gained"]
        lost = targets["E1"].fields["types_lost"]
        assert gained[CARD_TYPES.index("creature")] == 1.0
        assert lost[CARD_TYPES.index("land")] == 1.0

    def test_library_manipulations_set_their_bits(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.CARD_LOOKED_AT, subjects=("P0",),
                  params={"count": 2, "from_zone": "library"}),
        ))
        vector = targets["P0"].fields["library_events"]
        assert vector[LIBRARY_EVENTS.index("scry")] == 1.0
