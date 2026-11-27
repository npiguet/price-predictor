"""Which fields an affected entity is scored on (FR-077, FR-081).

A record that says nothing about a field must not be read as having said zero:
an untapped creature is not one no event tapped, and a creature that was
destroyed carries no tap state at all. These go through the public pipeline —
targets, collation, loss — so they hold whatever the targets look like inside.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from effects.domain.effect_head_layout import (
    DISPERSED_FIELDS,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    PER_ENTITY_WIDTH,
    FieldScope,
)
from effects.domain.effect_losses import EntityLossScoring, per_entity_loss
from effects.domain.effect_targets import derive_targets
from effects.domain.event_schema import Event, EventType
from effects.domain.input_layout import Slot, SlotKind
from effects.domain.records import (
    ContinuousPayload,
    Contribution,
    ForbiddenEntity,
    PlayabilityAttackersPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.sampling import SamplingClass
from effects.domain.state_snapshot import PlayerState
from effects.domain.surface_collation import entity_target_tensors
from tests.unit.effects.builders import (
    curated_fixture_records,
    make_entity,
    make_entity_targets,
    make_record,
    make_snapshot,
)


@dataclass
class _Surface:
    slots: tuple
    entity_slots: tuple


_SURFACE = _Surface(
    slots=(
        Slot(SlotKind.GLOBAL, 0), Slot(SlotKind.ACT, 1),
        Slot(SlotKind.PLAYER, 2, player_id="P0"),
        Slot(SlotKind.CARD, 0, entity_id="E1"),
    ),
    entity_slots=(2, 3),
)
_SNAPSHOT = make_snapshot(
    players=(PlayerState(id="P0", life=20, hand=7, library=40, graveyard=0),),
    entities=(make_entity("E1", "Bear"),),
)


def _resolution(*events):
    return make_record(state=_SNAPSHOT, payload=ResolutionPayload(events=events))


def _losses(record, names, outputs):
    fields = tuple(FIELDS_BY_NAME[name] for name in names)
    targets = entity_target_tensors([_SURFACE], [derive_targets(record)], fields)
    return per_entity_loss(
        outputs, targets,
        EntityLossScoring(
            fields=fields,
            log_dispersion=torch.zeros(len(DISPERSED_FIELDS)),
            report_parts=True,
        ),
    )


def _moves(record, scored: str, perturbed: str | None = None) -> bool:
    """Whether pushing ``perturbed``'s outputs moves the loss of ``scored``."""
    perturbed = perturbed or scored
    torch.manual_seed(0)
    outputs = torch.randn(1, 2, PER_ENTITY_WIDTH)
    names = sorted({scored, perturbed, "zone_outcome"})
    before = _losses(record, names, outputs)[0]
    start, end = FIELD_SLICES[perturbed]
    pushed = outputs.clone()
    # A ramp rather than a constant: a constant shift leaves a softmax as it was.
    pushed[..., start:end] += 3.0 * torch.arange(1, end - start + 1)
    after = _losses(record, names, pushed)[0]
    return abs(float(after) - float(before)) > 1e-4


_DESTROYED = Event(type=EventType.DESTROYED, subjects=("E1",))
_LIFE_LOST = Event(type=EventType.LIFE_CHANGE, subjects=("P0",), params={"delta": -3})


class TestAbsentFieldsAreNotScored:
    def test_an_absent_field_on_an_affected_entity_contributes_no_loss(self):
        """A destroyed creature had no tap event: its tap state is unknown."""
        assert not _moves(_resolution(_DESTROYED), "tap_state")

    def test_a_tap_event_is_scored(self):
        tapped = Event(type=EventType.TAPPED, subjects=("E1",))
        assert _moves(_resolution(tapped), "tap_state")

    def test_an_untap_event_is_scored_as_untapped(self):
        untapped = Event(type=EventType.UNTAPPED, subjects=("E1",))
        assert _moves(_resolution(untapped), "tap_state")

    def test_a_duration_no_event_states_is_not_scored(self):
        pumped = Event(
            type=EventType.PT_CHANGE, subjects=("E1",),
            params={"power_delta": 2, "toughness_delta": 2},
        )
        assert not _moves(_resolution(pumped), "pt_duration")

    def test_a_player_is_not_scored_on_a_permanents_fields(self):
        assert not _moves(_resolution(_LIFE_LOST), "zone_outcome")
        assert not _moves(_resolution(_LIFE_LOST), "damage_taken")

    def test_a_permanent_is_not_scored_on_a_players_fields(self):
        assert not _moves(_resolution(_DESTROYED), "life_delta")
        assert not _moves(_resolution(_DESTROYED), "mana_delta_r")


class TestSilentFieldsReadTheirKnownValue:
    def test_an_affected_permanent_that_took_no_damage_is_scored_on_zero_damage(self):
        assert _moves(_resolution(_DESTROYED), "damage_taken")

    def test_a_damaged_permanent_that_did_not_move_is_scored_as_stayed(self):
        hit = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 1})
        assert _moves(_resolution(hit), "zone_outcome")

    def test_an_affected_player_who_drew_nothing_is_scored_on_zero_cards(self):
        assert _moves(_resolution(_LIFE_LOST), "cards_drawn")


class TestRecordKindsSuperviseTheirOwnFields:
    def test_a_legality_record_says_nothing_about_zones(self):
        record = make_record(
            RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
            state=_SNAPSHOT,
            payload=PlayabilityAttackersPayload(
                forbidden=(ForbiddenEntity(entity="E1"),),
            ),
        )
        assert _moves(record, "attacker_legal")
        assert not _moves(record, "damage_taken")
        assert not _moves(record, "zone_outcome")

    def test_a_static_that_only_grants_a_keyword_is_scored_on_no_pt_change(self):
        record = make_record(
            RecordKind.CONTINUOUS, state=_SNAPSHOT,
            payload=ContinuousPayload(contributions=(
                Contribution(entity="E1", keywords=("flying",)),
            )),
        )
        assert _moves(record, "power_delta")


class TestTheFloorReadsPresenceToo:
    def test_an_absent_target_does_not_move_the_constant_predictor(self):
        """The floor is scored on the entities the loss is, absent ones excluded."""
        from effects.domain.entity_field_targets import EntityFieldTargets
        from effects.domain.loss_floor import constant_predictor_floor

        spec = FIELDS_BY_NAME["tap_state"]
        gate = torch.ones(1, 4)

        def floor(present):
            batch = make_entity_targets(
                gate=gate,
                fields=EntityFieldTargets(
                    {"tap_state": torch.tensor([[1.0, 0.0, 0.0, 0.0]])},
                    {"tap_state": torch.tensor([present])},
                ),
                mask=gate,
            )
            return constant_predictor_floor([batch], fields=(spec,))["tap_state"]

        # Two present entities, one tapped: a coin flip over two entities.
        assert floor([True, True, False, False]) == pytest.approx(2 * math.log(2))


#: The fields whose event the collector writes on every resolution that
#: does it, so a resolution's effect half that names an entity and no such
#: event says 0: soulbond, mutate and a Class level are clause
#: rules on the only APIs that do them, the blessing and a completed dungeon
#: bus readings of the command zone.
_SILENT_ON_EFFECT_HALVES = (
    "paired", "merged", "class_levels_gained", "gained_city_blessing", "completed_dungeon",
)
#: Of those, the ones no fixture record says 1 or more for: their 1s are rare
#: mechanics the fixture's 73 games never played.
_NO_POSITIVE_IN_THE_FIXTURE = frozenset(_SILENT_ON_EFFECT_HALVES)
#: The one-valued fields left without a silent value, and why: a 0 on them
#: would contradict what Forge did on some resolution.
_LEFT_WITHOUT_A_SILENT_VALUE = {
    "door_unlocked": "a Room spell's entry unlocks the door cast with no UnlockDoor clause",
    "entered_in_combat": "Dig, DigUntil, Meld, Block and a sneak cast add combatants unread",
    "became_blocked": "a creature put onto the battlefield blocking blocks its attacker unread",
}
#: The one-valued fields whose silent 0 is a continuous record's alone: a
#: contribution names every ability its static grants or takes,
#: and an Animate of its own source names no subject on a resolution.
_SILENT_ON_CONTINUOUS = ("abilities_gained", "abilities_lost")


def _scored_values(records, name: str) -> dict[SamplingClass, set[str]]:
    """``sampling class -> {"0", "+"}``: the values ``name`` is scored on
    over ``records``, on the entities the gate trains."""
    spec = FIELDS_BY_NAME[name]
    seen: dict[SamplingClass, set[str]] = {}
    for record in records:
        targets = derive_targets(record)
        for entry in targets.values():
            value = entry.supervised(spec) if targets.gated and entry.affected else None
            if value is not None:
                seen.setdefault(entry.record_class, set()).add("+" if float(value) > 0 else "0")
    return seen


@pytest.fixture(scope="module")
def fixture_records():
    return curated_fixture_records()


class TestOneValuedFieldsTakeTheirSilentZero:
    @pytest.mark.parametrize("name", _SILENT_ON_EFFECT_HALVES)
    def test_the_fixture_scores_both_values_or_names_the_missing_positive(
        self, fixture_records, name,
    ):
        values = set().union(*_scored_values(fixture_records, name).values())
        expected = {"0"} if name in _NO_POSITIVE_IN_THE_FIXTURE else {"0", "+"}
        assert values == expected, f"{name} is scored on {sorted(values)} over the fixture"

    @pytest.mark.parametrize("name", _SILENT_ON_EFFECT_HALVES)
    def test_only_a_resolutions_effect_half_says_nothing_happened(self, fixture_records, name):
        """A combat or continuous record's affected permanent is no
        resolution the clause rule watched."""
        classes = set(_scored_values(fixture_records, name))
        assert classes <= {SamplingClass.RESOLUTION_EFFECT}

    @pytest.mark.parametrize("name", sorted(_LEFT_WITHOUT_A_SILENT_VALUE))
    def test_a_field_whose_rule_misses_a_case_takes_no_silent_value(self, name):
        assert FIELDS_BY_NAME[name].silent is None, _LEFT_WITHOUT_A_SILENT_VALUE[name]


class TestAbilitiesTakeTheirSilentZeroOnAContinuousRecord:
    @pytest.mark.parametrize("name", _SILENT_ON_CONTINUOUS)
    def test_the_fixture_s_continuous_records_score_both_values(self, fixture_records, name):
        values = _scored_values(fixture_records, name).get(SamplingClass.CONTINUOUS, set())
        assert values == {"0", "+"}, f"{name} is scored on {sorted(values)} there"

    @pytest.mark.parametrize("name", _SILENT_ON_CONTINUOUS)
    def test_no_other_class_reads_silence_as_none_changed(self, name):
        assert FIELDS_BY_NAME[name].supervised_by == {SamplingClass.CONTINUOUS}


class TestAWrittenValueOutsideTheScopeIsRefused:
    """``damage_taken`` is a permanent's field: a player's damage is its
    ``life_delta`` and ``poison_delta``."""

    _RUN = "3550376f-3c30-4231-90cd-cc0f1b308e87"

    @pytest.mark.parametrize("record_id", [
        f"{_RUN}.0-mwqvqdyrb9c0.8",  # a combat record: an attacker hits P0
        f"{_RUN}.0-mwqvqdyrb9c0.613",  # a resolution's effect half: 3 damage to P0
    ])
    def test_a_damage_dealt_naming_a_player_sets_no_damage_taken_on_the_player(self, record_id):
        record = next(r for r in curated_fixture_records() if r.record_id == record_id)
        players = [t for t in derive_targets(record).values() if t.scope.value == "player"]
        assert players
        for player in players:
            assert player.supervised(FIELDS_BY_NAME["damage_taken"]) is None, record_id

    def test_a_legality_mark_is_kept_on_an_entity_of_either_scope(self):
        spec = FIELDS_BY_NAME["target_legal"]
        for scope in (FieldScope.PERMANENT, FieldScope.PLAYER):
            assert spec.supervised_value(
                {"target_legal": 1.0}, set(), record_class=SamplingClass.PLAYABILITY_LEGALITY,
                scope=scope,
            ) == 1.0
