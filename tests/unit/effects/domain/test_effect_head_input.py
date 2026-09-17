"""The effect head's token geometry (T043).

Four properties the head depends on: position ids restart at each ``[CARD]``,
``[ACT]`` is empty where no single line acts, controller tags derive from the
record's actor rather than being stored, and scalars enter unbinned as raw plus
a log1p copy.
"""

from __future__ import annotations

import math
import random

import pytest

from effects.domain.effect_head_input import (
    OVERLAY_KEYWORDS,
    SlotKind,
    _scalar,
    act_features,
    anchored_attacker,
    build_effect_head_input,
    card_features,
    continuous_masked_keywords,
    global_features,
    player_features,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ActivationPayload,
    Candidate,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    GrantedTemporary,
    PlayerState,
    PowerToughness,
    Refs,
    StateSnapshot,
)

E_DIM = 4
_ZERO = (0.0,) * E_DIM

_BOLT = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_ANTHEM = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
_WARD = ProvenanceKey("cardsfolder/w/ward.txt", 0, "static", 1)

_VECTORS = {
    _BOLT: (1.0, 0.0, 0.0, 0.0),
    _ANTHEM: (0.0, 1.0, 0.0, 0.0),
    _WARD: (0.0, 0.0, 1.0, 0.0),
}


def _e_for(key: ProvenanceKey):
    return _VECTORS.get(key)


def _entity(entity_id: str, **overrides) -> EntityState:
    defaults = {
        "id": entity_id, "name": "Serra Angel", "zone": "battlefield",
        "controller": "P0", "types": ("Creature",), "colors": ("W",),
        "mana_value": 5, "pt": PowerToughness(base=(4, 4)),
    }
    defaults.update(overrides)
    return EntityState(**defaults)


def _snapshot(**overrides) -> StateSnapshot:
    defaults = {
        "global_": GlobalState(
            turn=7, phase="main1", active="P0", priority="P0", stack_size=1,
        ),
        "players": (
            PlayerState(id="P0", life=14, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        ),
        "entities": (
            _entity("E1", printed=(_ANTHEM,)),
            _entity("E2", controller="P1", printed=(_WARD,)),
        ),
    }
    defaults.update(overrides)
    return StateSnapshot(**defaults)


def _record(**overrides) -> EffectRecord:
    kind = overrides.pop("kind", RecordKind.RESOLUTION)
    defaults: dict = {
        "record_id": "run.0.1", "run_id": "run", "timestamp": "t",
        "game_id": "run.0.1", "kind": kind, "actor_player": "P0",
        "state": _snapshot(),
    }
    if kind is RecordKind.RESOLUTION:
        defaults["moment"] = Moment.RESOLUTION
        defaults["ability"] = (_BOLT,)
        defaults["payload"] = ResolutionPayload()
    elif kind is RecordKind.COMBAT:
        defaults["payload"] = CombatPayload()
    defaults.update(overrides)
    return EffectRecord(**defaults)


def _build(record, **kwargs):
    return build_effect_head_input(record, e_for=_e_for, e_dim=E_DIM, **kwargs)


class TestSurfaceShape:
    def test_the_surface_opens_with_global_then_act(self):
        surface = _build(_record())
        assert surface.slots[0].kind is SlotKind.GLOBAL
        assert surface.slots[1].kind is SlotKind.ACT
        assert surface.act_index == 1

    def test_players_come_before_cards(self):
        kinds = [slot.kind for slot in _build(_record()).slots]
        assert kinds.index(SlotKind.PLAYER) < kinds.index(SlotKind.CARD)

    def test_every_player_and_entity_gets_a_slot(self):
        surface = _build(_record())
        assert len(surface.of_kind(SlotKind.PLAYER)) == 2
        assert len(surface.of_kind(SlotKind.CARD)) == 2

    def test_the_per_entity_head_targets_every_card_and_player(self):
        """FR-081: the gate trains on every entity, players included."""
        surface = _build(_record())
        targeted = {surface.slots[i].kind for i in surface.entity_slots}
        assert targeted == {SlotKind.PLAYER, SlotKind.CARD}
        assert len(surface.entity_slots) == 4

    def test_an_entitys_abilities_follow_its_card_token(self):
        surface = _build(_record())
        kinds = [slot.kind for slot in surface.slots]
        first_card = kinds.index(SlotKind.CARD)
        assert kinds[first_card + 1] is SlotKind.ABILITY
        assert surface.slots[first_card + 1].entity_id == "E1"


class TestPositionReset:
    def test_positions_run_up_to_the_first_card(self):
        surface = _build(_record())
        leading = [
            slot.position for slot in surface.slots
            if slot.kind in (SlotKind.GLOBAL, SlotKind.ACT, SlotKind.PLAYER)
        ]
        assert leading == [0, 1, 2, 3]

    def test_each_card_restarts_at_zero(self):
        surface = _build(_record())
        for slot in surface.of_kind(SlotKind.CARD):
            assert slot.position == 0

    def test_an_entitys_abilities_number_upward_from_its_card(self):
        surface = _build(_record())
        positions = [
            (slot.kind, slot.position) for slot in surface.slots
            if slot.kind in (SlotKind.CARD, SlotKind.ABILITY)
        ]
        assert positions == [
            (SlotKind.CARD, 0), (SlotKind.ABILITY, 1),
            (SlotKind.CARD, 0), (SlotKind.ABILITY, 1),
        ]

    def test_a_board_position_does_not_change_an_entitys_geometry(self):
        """The reset is what makes twelve permanents one geometry, not twelve."""
        one = _snapshot(entities=(_entity("E1", printed=(_ANTHEM,)),))
        many = _snapshot(entities=(
            _entity("E0"), _entity("E1", printed=(_ANTHEM,)),
        ))
        first = _build(_record(state=one))
        second = _build(_record(state=many))
        target_first = [s for s in first.slots if s.entity_id == "E1"]
        target_second = [s for s in second.slots if s.entity_id == "E1"]
        assert [s.position for s in target_first] == [
            s.position for s in target_second
        ]


class TestActSlot:
    def test_a_resolution_carries_its_acting_lines_vector(self):
        assert _build(_record()).slots[1].e == _VECTORS[_BOLT]

    def test_act_is_empty_for_a_combat_record(self):
        surface = _build(_record(kind=RecordKind.COMBAT, ability=None))
        assert surface.slots[1].e == _ZERO
        assert set(surface.slots[1].features) == {0.0}

    def test_act_is_empty_for_the_attackers_subkind(self):
        from effects.domain.records import PlayabilityAttackersPayload

        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.ATTACKERS,
            payload=PlayabilityAttackersPayload(),
        )
        assert set(_build(record).slots[1].features) == {0.0}

    def test_act_is_empty_for_the_blockers_subkind(self):
        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.BLOCKERS,
            payload=PlayabilityBlockersPayload(anchor_attacker="E1"),
        )
        assert set(_build(record).slots[1].features) == {0.0}

    def test_a_populated_act_slot_says_so_in_its_first_feature(self):
        assert act_features(_record())[0] == 1.0
        assert act_features(_record(kind=RecordKind.COMBAT, ability=None))[0] == 0.0

    def test_a_decision_record_puts_the_candidates_vector_in_act(self):
        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.DECISION,
            payload=PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(_BOLT,), can_play=True, affordable=True,
                          has_legal_target=True),
                Candidate(ability=(_ANTHEM,), can_play=False, affordable=False,
                          has_legal_target=False),
            )),
        )
        assert _build(record, candidate_index=0).slots[1].e == _VECTORS[_BOLT]
        assert _build(record, candidate_index=1).slots[1].e == _VECTORS[_ANTHEM]

    def test_an_unresolvable_key_contributes_zeros_rather_than_dropping(self):
        missing = ProvenanceKey("cardsfolder/x/absent.txt", 0, "spell", 0)
        assert _build(_record(ability=(missing,))).slots[1].e == _ZERO

    def test_the_cost_halfs_outcome_reaches_the_act_slot(self):
        resolved = _record(
            moment=Moment.ACTIVATION,
            payload=ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
            ),
        )
        countered = _record(
            moment=Moment.ACTIVATION,
            payload=ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.COUNTERED,
            ),
        )
        assert act_features(resolved) != act_features(countered)


class TestAbilityTokens:
    def test_printed_and_attachment_granted_lines_become_tokens(self):
        entity = _entity("E1", printed=(_ANTHEM,), granted_attached=(_WARD,))
        surface = _build(_record(state=_snapshot(entities=(entity,))))
        vectors = [s.e for s in surface.of_kind(SlotKind.ABILITY)]
        assert vectors == [_VECTORS[_ANTHEM], _VECTORS[_WARD]]

    def test_temporary_grants_do_not_become_tokens(self):
        """FR-073: they ride the overlay instead."""
        entity = _entity(
            "E1", printed=(_ANTHEM,),
            granted_temporary=GrantedTemporary(abilities=(_WARD,)),
        )
        surface = _build(_record(state=_snapshot(entities=(entity,))))
        assert len(surface.of_kind(SlotKind.ABILITY)) == 1

    def test_context_dropout_can_drop_a_context_ability(self):
        record = _record()
        kept = _build(record, context_dropout=0.0)
        dropped = _build(record, context_dropout=1.0, rng=random.Random(1))
        assert len(dropped.of_kind(SlotKind.ABILITY)) < len(
            kept.of_kind(SlotKind.ABILITY)
        )

    def test_the_acting_entitys_abilities_are_never_dropped(self):
        """Dropping the ability under study would make the record unanswerable."""
        state = _snapshot(refs=Refs(source="E1"))
        surface = _build(
            _record(state=state), context_dropout=1.0, rng=random.Random(1),
        )
        acting = [s for s in surface.of_kind(SlotKind.ABILITY) if s.entity_id == "E1"]
        assert len(acting) == 1

    def test_dropout_is_reproducible_under_a_seeded_rng(self):
        record = _record()
        first = _build(record, context_dropout=0.5, rng=random.Random(3))
        second = _build(record, context_dropout=0.5, rng=random.Random(3))
        assert len(first.slots) == len(second.slots)


class TestControllerTags:
    def test_mine_and_opponent_derive_from_the_records_actor(self):
        state = _snapshot()
        from_p0 = card_features(state.entities[0], _record(state=state))
        from_p1 = card_features(
            state.entities[0], _record(state=state, actor_player="P1"),
        )
        assert from_p0 != from_p1

    def test_the_player_slot_flags_the_actor(self):
        record = _record()
        mine = player_features(record.state.players[0], record)
        theirs = player_features(record.state.players[1], record)
        assert mine[-3:] == (1.0, 0.0, 0.0)
        assert theirs[-3:] == (0.0, 1.0, 0.0)

    def test_a_targeted_player_is_flagged(self):
        state = _snapshot(refs=Refs(targets=("P1",)))
        record = _record(state=state)
        assert player_features(state.players[1], record)[-1] == 1.0

    def test_the_global_slot_flags_whose_turn_it_is(self):
        mine = global_features(_record())
        theirs = global_features(_record(actor_player="P1"))
        assert mine != theirs


class TestNumericEncoding:
    def test_a_scalar_enters_raw_plus_log1p(self):
        assert _scalar(3) == pytest.approx([3.0, math.log1p(3)])

    def test_the_log_copy_is_signed_so_it_stays_monotone_across_zero(self):
        assert _scalar(-3)[1] == pytest.approx(-math.log1p(3))
        assert _scalar(-3)[1] < _scalar(0)[1] < _scalar(3)[1]

    def test_nothing_is_binned(self):
        """Two nearby values give two different feature vectors."""
        assert _scalar(3) != _scalar(4)

    def test_a_players_life_reaches_the_features_unbinned(self):
        record = _record()
        low = player_features(
            PlayerState(id="P0", life=1, hand=0, library=0, graveyard=0), record,
        )
        high = player_features(
            PlayerState(id="P0", life=2, hand=0, library=0, graveyard=0), record,
        )
        assert low != high


class TestPerKindVariations:
    def test_a_continuous_record_masks_its_own_static_contribution(self):
        entity = _entity(
            "E1", granted_temporary=GrantedTemporary(keywords=("flying",)),
        )
        record = _record(
            kind=RecordKind.CONTINUOUS, moment=None, ability=(_ANTHEM,),
            state=_snapshot(entities=(entity,)),
            payload=ContinuousPayload(contributions=(
                Contribution(entity="E1", keywords=("flying",)),
            )),
        )
        masked = continuous_masked_keywords(record)
        assert masked == {"E1": frozenset({"flying"})}
        with_mask = card_features(entity, record, masked_keywords=masked["E1"])
        without = card_features(entity, record)
        assert with_mask != without

    def test_masking_leaves_the_structured_features_alone(self):
        """The static changed keywords, not the card's printed type line."""
        entity = _entity(
            "E1", granted_temporary=GrantedTemporary(keywords=("flying",)),
        )
        record = _record(kind=RecordKind.COMBAT, ability=None,
                         state=_snapshot(entities=(entity,)))
        masked = card_features(entity, record, masked_keywords=frozenset({"flying"}))
        without = card_features(entity, record)
        # Only the overlay's flying bit differs.
        differing = [i for i, (a, b) in enumerate(zip(masked, without)) if a != b]
        assert len(differing) == 1

    def test_a_granted_keyword_is_addressable_for_the_gate_two_perturbation(self):
        for keyword in ("first_strike", "double_strike", "deathtouch", "lifelink",
                        "trample", "indestructible", "wither", "infect"):
            assert keyword in OVERLAY_KEYWORDS

    def test_a_blockers_record_names_its_anchored_attacker(self):
        record = _record(
            kind=RecordKind.PLAYABILITY, moment=None, ability=None,
            subkind=PlayabilitySubkind.BLOCKERS,
            payload=PlayabilityBlockersPayload(anchor_attacker="E2"),
        )
        assert anchored_attacker(record) == "E2"

    def test_a_non_blockers_record_has_no_anchor(self):
        assert anchored_attacker(_record()) is None

    def test_a_combat_record_carries_its_damage_assignment_choices(self):
        entity = _entity("E1", combat=CombatStatus(attacking="P1"))
        plain = _record(
            kind=RecordKind.COMBAT, ability=None,
            state=_snapshot(entities=(entity,)), payload=CombatPayload(),
        )
        assigned = _record(
            kind=RecordKind.COMBAT, ability=None,
            state=_snapshot(entities=(entity,)),
            payload=CombatPayload(
                attackers=("E1",), assignment_choices={"E1": {"E2": 4}},
            ),
        )
        assert card_features(entity, plain) != card_features(entity, assigned)

    def test_a_trigger_record_overlays_its_pending_event(self):
        entity = _entity("E1")
        pending = Event(type=EventType.LIFE_CHANGE, subjects=("E1",),
                        params={"delta": 3})
        record = _record(
            kind=RecordKind.TRIGGER, moment=None,
            state=_snapshot(entities=(entity,), pending_event=pending),
            payload=TriggerPayload(event=pending, fired=True),
        )
        without = _record(
            kind=RecordKind.TRIGGER, moment=None,
            state=_snapshot(entities=(entity,)),
            payload=TriggerPayload(event=pending, fired=True),
        )
        assert card_features(entity, record) != card_features(entity, without)


class TestFeatureWidths:
    def test_every_card_slot_has_the_same_width(self):
        surface = _build(_record())
        widths = {len(slot.features) for slot in surface.of_kind(SlotKind.CARD)}
        assert len(widths) == 1

    def test_every_player_slot_has_the_same_width(self):
        surface = _build(_record())
        widths = {len(slot.features) for slot in surface.of_kind(SlotKind.PLAYER)}
        assert len(widths) == 1

    def test_act_width_is_the_same_whether_populated_or_empty(self):
        populated = len(act_features(_record()))
        empty = len(act_features(_record(kind=RecordKind.COMBAT, ability=None)))
        assert populated == empty

    def test_an_entity_without_power_toughness_keeps_the_width(self):
        creature = _entity("E1")
        land = _entity("E2", pt=None, types=("Land",))
        record = _record(state=_snapshot(entities=(creature, land)))
        assert len(card_features(creature, record)) == len(
            card_features(land, record)
        )


class TestKeysWithNoLine:
    """A key the sidecar maps to no line is not an ability (FR-073).

    Forge attaches an implicit cast-this-permanent object to every permanent,
    and the converter renders no line for it. Without the predicate the
    surface carried a zero-vector token for it on almost every entity.
    """

    _PHANTOM = ProvenanceKey("cardsfolder/s/serra_angel.txt", 0, "spell", 0)

    def test_a_key_with_no_line_gets_no_ability_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM, _ANTHEM)),
        )))
        surface = _build(record, has_line=lambda key: key in _VECTORS)
        abilities = surface.of_kind(SlotKind.ABILITY)
        assert [slot.e for slot in abilities] == [_VECTORS[_ANTHEM]]

    def test_positions_stay_contiguous_after_a_skip(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM, _ANTHEM, _WARD)),
        )))
        surface = _build(record, has_line=lambda key: key in _VECTORS)
        positions = [
            slot.position for slot in surface.slots
            if slot.kind in (SlotKind.CARD, SlotKind.ABILITY)
        ]
        assert positions == [0, 1, 2]

    def test_a_masked_line_keeps_its_zero_token(self):
        """The state-only control zeroes ``e`` but keeps the geometry."""
        record = _record(state=_snapshot(entities=(_entity("E1", printed=(_ANTHEM,)),)))
        surface = build_effect_head_input(
            record, e_for=lambda key: None, e_dim=E_DIM, has_line=lambda key: True,
        )
        assert [slot.e for slot in surface.of_kind(SlotKind.ABILITY)] == [_ZERO]

    def test_without_the_predicate_every_key_keeps_its_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM,)),
        )))
        assert [slot.e for slot in _build(record).of_kind(SlotKind.ABILITY)] == [_ZERO]
