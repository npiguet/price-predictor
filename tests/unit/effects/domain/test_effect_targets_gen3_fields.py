"""Fields gen 3 adds: base power and toughness, the state flags, copies and
retargets, and the gen-3 event types."""

from __future__ import annotations

import pytest

from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.input_vocabularies import CARD_TYPES
from effects.domain.records import (
    ContinuousPayload,
    Contribution,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import COLORS
from tests.unit.effects.domain.effect_targets_support import (
    Board,
    _entity,
    _on_board,
    _resolution,
    _spec,
)


class TestBasePowerAndToughness:
    """Story 5 acceptance 5: Ensoul Artifact sets base 5/5; the target is the
    change against the previous base."""

    def test_a_set_base_is_its_change_against_the_snapshot_base(self):
        from effects.domain.state_snapshot import PowerToughness

        artifact = _entity("E1", types=("artifact",))
        creature = _entity("E2", types=("artifact", "creature"),
                           pt=PowerToughness(base=(1, 3), boosts=(2, 0)))
        record = _on_board(ResolutionPayload(events=(
            Event(type=EventType.PT_CHANGE, subjects=("E1", "E2"),
                  params={"power_delta": 5, "toughness_delta": 5, "set_pt": [5, 5]}),
        )), Board.of(artifact, creature))
        targets = derive_targets(record)
        assert targets["E1"].fields["base_power_delta"] == 5.0
        assert targets["E1"].fields["base_toughness_delta"] == 5.0
        assert targets["E2"].fields["base_power_delta"] == 4.0
        assert targets["E2"].fields["base_toughness_delta"] == 2.0

    def test_a_continuous_set_base_is_its_change_against_the_board_without_it(self):
        """A continuous record's board is the board without its static, so
        Frogify's 1/1 on a 4/4 is a base change of -3/-3, and Ensoul
        Artifact's 5/5 on a noncreature artifact one of +5/+5."""
        from effects.domain.state_snapshot import PowerToughness

        angel = _entity("E1", types=("creature",), pt=PowerToughness(base=(4, 4)))
        artifact = _entity("E2", types=("artifact",))
        record = _on_board(ContinuousPayload(contributions=(
            Contribution(entity="E1", set_pt=(1, 1)),
            Contribution(entity="E2", set_pt=(5, 5)),
        )), Board.of(angel, artifact), kind=RecordKind.CONTINUOUS, moment=None)
        targets = derive_targets(record)
        assert targets["E1"].fields["base_power_delta"] == -3.0
        assert targets["E1"].fields["base_toughness_delta"] == -3.0
        assert targets["E2"].fields["base_power_delta"] == 5.0
        assert "power_delta" not in targets["E1"].fields

    def test_a_continuous_set_half_says_nothing_of_the_other_half(self):
        from effects.domain.state_snapshot import PowerToughness

        bears = _entity("E1", types=("creature",), pt=PowerToughness(base=(2, 2)))
        record = _on_board(ContinuousPayload(contributions=(
            Contribution(entity="E1", set_pt=(0, None)),
        )), Board.of(bears), kind=RecordKind.CONTINUOUS, moment=None)
        entry = derive_targets(record)["E1"]
        assert entry.fields["base_power_delta"] == -2.0
        assert entry.supervised(_spec("base_toughness_delta")) is None

    def test_a_gen3_static_that_sets_a_line_loses_what_the_board_without_it_had(self):
        """Imprisoned in the Moon makes a green creature a colourless land. A
        gen-3 continuous record's board is the board without the static
        (FR-042, ``StaticSuppression``), so what the ``=`` displaced is the
        entity's own types and colours there: creature and green are lost."""
        bears = _entity("E1", types=("creature",), colors=("G",))
        record = _on_board(ContinuousPayload(contributions=(
            Contribution(entity="E1", colors=("=", "C"), types=("=", "land")),
        )), Board.of(bears), kind=RecordKind.CONTINUOUS, moment=None, generation=3)
        entry = derive_targets(record)["E1"]
        assert entry.fields["types_lost"] == [
            1.0 if name == "creature" else 0.0 for name in CARD_TYPES
        ]
        assert entry.fields["colors_lost"] == [
            1.0 if name == "G" else 0.0 for name in COLORS
        ]
        assert "types_lost" not in entry.unknown and "colors_lost" not in entry.unknown

    def test_a_gen3_set_line_the_board_already_had_loses_nothing(self):
        land = _entity("E1", types=("land",))
        record = _on_board(ContinuousPayload(contributions=(
            Contribution(entity="E1", types=("=", "land")),
        )), Board.of(land), kind=RecordKind.CONTINUOUS, moment=None, generation=3)
        entry = derive_targets(record)["E1"]
        assert entry.fields["types_lost"] == [0.0] * len(CARD_TYPES)

    def test_a_continuous_control_change_is_a_control_change(self):
        record = _on_board(ContinuousPayload(contributions=(
            Contribution(entity="E1", controller="P0"),
        )), Board.of(_entity("E1", controller="P1")), kind=RecordKind.CONTINUOUS, moment=None)
        assert derive_targets(record)["E1"].fields["control_change"] == 1

    def test_a_pump_says_nothing_of_the_base(self):
        entry = derive_targets(_resolution(
            Event(type=EventType.PT_CHANGE, subjects=("E1",),
                  params={"power_delta": 3, "toughness_delta": 3}),
        ))["E1"]
        assert entry.supervised(_spec("base_power_delta")) is None


class TestStateFlagOutputs:
    """T123g's ``state_flag_change`` flags reach their own fields."""

    def _fields(self, subject, flag, value=True):
        return derive_targets(_resolution(
            Event(type=EventType.STATE_FLAG_CHANGE, subjects=(subject,),
                  params={"flag": flag, "value": value}),
        ))[subject].fields

    @pytest.mark.parametrize(("flag", "field"), [
        ("door_unlocked", "door_unlocked"),
        ("paired", "paired"),
        ("merged", "merged"),
        ("entered_attacking", "entered_in_combat"),
        ("entered_blocking", "entered_in_combat"),
    ])
    def test_a_permanent_flag_sets_its_field(self, flag, field):
        assert self._fields("E1", flag)[field] == 1.0

    @pytest.mark.parametrize(("flag", "field"), [
        ("city_blessing", "gained_city_blessing"),
        ("dungeon_completed", "completed_dungeon"),
    ])
    def test_a_player_flag_sets_its_field(self, flag, field):
        assert self._fields("P0", flag)[field] == 1.0

    def test_each_class_level_event_counts_one_level(self):
        targets = derive_targets(_resolution(*(
            Event(type=EventType.STATE_FLAG_CHANGE, subjects=("E1",),
                  params={"flag": "class_level", "value": level})
            for level in (2, 3)
        )))
        assert targets["E1"].fields["class_levels_gained"] == 2.0


class TestCopiesAndRetargets:
    def test_a_clone_entering_as_a_copy_is_placed_on_the_clone(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "stack", "to_zone": "battlefield"}),
            Event(type=EventType.PERMANENT_COPIED,
                  params={"copy_source": "Grizzly Bears", "count": 1, "cause": "E1"}),
        )), Board.of(_entity("E1", zone="stack"))))
        assert targets["E1"].fields["copied"] == 1

    def test_token_copies_place_nothing_and_leave_copied_unknown(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.PERMANENT_COPIED,
                  params={"copy_source": "Mountain", "count": 1, "cause": "E1"}),
            Event(type=EventType.TAPPED, subjects=("E1",)),
        )), Board.of(_entity("E1"))))
        assert targets["E1"].supervised(_spec("copied")) is None

    def test_a_spell_copy_lands_on_the_targeted_stack_object(self):
        from effects.domain.state_snapshot import Refs

        targets = derive_targets(_on_board(
            ResolutionPayload(events=(
                Event(type=EventType.SPELL_COPIED, params={"count": 2, "cause": "E1"}),
            )),
            Board.of(_entity("E1", zone="stack"), _entity("E2", zone="stack"),
                     refs=Refs(targets=("E2",))),
        ))
        assert targets["E2"].fields["spell_copies"] == 2.0
        assert "E1" not in targets

    def test_a_retarget_lands_on_the_retargeted_spell(self):
        from effects.domain.state_snapshot import Refs

        targets = derive_targets(_on_board(
            ResolutionPayload(events=(
                Event(type=EventType.TARGETS_CHANGED, params={"targets": ["P0"]}),
            )),
            Board.of(_entity("E2", zone="stack"), refs=Refs(targets=("E2",))),
        ))
        assert targets["E2"].fields["targets_changed"] == 1


class TestGen3EventFields:
    @pytest.mark.parametrize(("event_type", "subject", "params", "expected"), [
        (EventType.REGENERATED, "E1", {}, {"regenerated": 1}),
        (EventType.DAMAGE_PREVENTED, "E1", {"amount": 2}, {"damage_prevented": 2.0}),
        (EventType.DAMAGE_PREVENTED, "P0", {"amount": 3}, {"damage_prevented": 3.0}),
        (EventType.ENERGY_CHANGE, "P0", {"delta": -2}, {"energy_delta": -2.0}),
        (EventType.RADIATION_CHANGE, "P0", {"delta": 3}, {"rad_delta": 3.0}),
        (EventType.SPEED_CHANGED, "P0", {"delta": 1}, {"speed_gained": 1.0}),
        (EventType.MONARCH_CHANGED, "P0", {}, {"became_monarch": 1}),
        (EventType.INITIATIVE_TAKEN, "P0", {}, {"took_initiative": 1}),
        (EventType.REMOVED_FROM_COMBAT, "E1", {}, {"removed_from_combat": 1}),
        (EventType.TEXT_CHANGE, "E1", {}, {"text_changed": 1}),
        (EventType.COUNTER_CHANGE, "E1", {"counter_type": "PAGE", "delta": 1},
         {"counters_delta_page": 1.0}),
        (EventType.ABILITY_CHANGE, "E1", {"abilities": "ABPump,ABTap", "removed": True},
         {"abilities_gained": 2.0, "abilities_lost": 1}),
    ])
    def test_its_fields(self, event_type, subject, params, expected):
        targets = derive_targets(_resolution(
            Event(type=event_type, subjects=(subject,), params=params),
        ))
        assert targets[subject].fields == expected

    def test_damage_prevented_is_silent_zero_on_a_card_and_a_player(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1", "P0"), params={"amount": 1}),
        ))
        assert targets["E1"].supervised(_spec("damage_prevented")) == 0.0
        assert targets["P0"].supervised(_spec("damage_prevented")) == 0.0

    def test_a_venture_is_its_causes_controllers(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.DUNGEON_VENTURED,
                  params={"dungeon": "Undercity", "room": "Secret Entrance", "cause": "E1"}),
        )), Board.of(_entity("E1", controller="P1"))))
        assert targets["P1"].fields == {"ventured": 1}

    def test_a_goad_is_a_designation_gained(self):
        from effects.domain.input_vocabularies import DESIGNATIONS

        fields = derive_targets(_resolution(
            Event(type=EventType.RESTRICTION_CHANGE, subjects=("E1",),
                  params={"restriction": "goad", "value": True}),
        ))["E1"].fields
        assert fields["designations_gained"][DESIGNATIONS.index("goaded")] == 1.0

    def test_a_type_change_reads_subtypes_and_supertypes_too(self):
        from effects.domain.input_vocabularies import SUPERTYPES
        from effects.domain.output_vocabularies import SUBTYPES_CHANGED

        fields = derive_targets(_resolution(
            Event(type=EventType.TYPE_CHANGE, subjects=("E1",),
                  params={"types_added": ["creature", "golem", "legendary"],
                          "types_removed": ["human"]}),
        ))["E1"].fields
        assert fields["types_gained"][CARD_TYPES.index("creature")] == 1.0
        assert fields["subtypes_gained"][SUBTYPES_CHANGED.index("golem")] == 1.0
        assert fields["subtypes_lost"][SUBTYPES_CHANGED.index("human")] == 1.0
        assert fields["supertypes_gained"][SUPERTYPES.index("legendary")] == 1.0

    def test_a_keyword_beyond_the_overlay_list_has_its_own_column(self):
        from effects.domain.output_vocabularies import KEYWORDS_CHANGED

        fields = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["toxic:2"], "removed": False}),
        ))["E1"].fields
        assert fields["keywords_gained"][KEYWORDS_CHANGED.index("toxic")] == 1.0
