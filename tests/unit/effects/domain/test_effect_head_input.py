"""The effect head's token geometry (T043).

Four properties the head depends on: position ids restart at each ``[CARD]``,
``[ACT]`` is empty where no single line acts, controller tags derive from the
record's actor rather than being stored, and scalars enter unbinned as raw plus
a log1p copy.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, replace

import pytest

from effects.domain.effect_head_input import (
    ACT_SLOT,
    NO_STRIP,
    OVERLAY_KEYWORDS,
    PHASES,
    THERMOMETER_STEPS,
    SlotKind,
    SurfaceBuilder,
    _push_scalar,
    act_features,
    card_features,
    card_segments,
    continuous_masked_keywords,
    counter_rows,
    global_features,
    player_features,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.kept_keyword_flags import KEPT_KEYWORD_FLAGS
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
from tests.unit.effects.builders import (
    make_entity,
    make_record,
    make_snapshot,
    slots_of_kind,
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


@dataclass(frozen=True)
class _Lines:
    """A :class:`LineLookup` from plain callables: every key has a line, is
    no keyword and has no modes unless a test says otherwise."""

    e_for: Callable = _e_for
    has_line: Callable = lambda key: True
    keyword_of: Callable = lambda key: None
    options_for: Callable = lambda key: ()


def _flags(slot) -> dict[str, float]:
    """An ``[ABILITY]`` slot's line flags by name."""
    active, granted, temporary, cast, count, _, chosen = slot.features
    return {"active": active, "granted": granted, "temporary": temporary,
            "cast_with": cast, "count": count, "chosen": chosen}


def _entity(entity_id: str, **overrides) -> EntityState:
    defaults = {
        "name": "Serra Angel", "types": ("Creature",), "colors": ("W",),
        "mana_value": 5, "pt": PowerToughness(base=(4, 4)),
    }
    defaults.update(overrides)
    return make_entity(entity_id, defaults.pop("name"), **defaults)


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
    return make_snapshot(**defaults)


def _record(**overrides) -> EffectRecord:
    kind = overrides.pop("kind", RecordKind.RESOLUTION)
    defaults: dict = {
        "record_id": "run.0.1", "game_id": "run.0.1", "state": _snapshot(),
    }
    if kind is RecordKind.RESOLUTION:
        defaults["ability"] = (_BOLT,)
    defaults.update(overrides)
    return make_record(kind, **defaults)


def _build(
    record, *, rng=None, e_for=_e_for, has_line=None, options_for=None,
    context_dropout=0.0, stripped=NO_STRIP,
):
    lines = _Lines(e_for=e_for)
    if has_line is not None:
        lines = replace(lines, has_line=has_line)
    if options_for is not None:
        lines = replace(lines, options_for=options_for)
    return SurfaceBuilder(
        lines, e_dim=E_DIM, rng=rng or random.Random(0), context_dropout=context_dropout,
    ).build(record, stripped)


class TestSurfaceShape:
    def test_the_surface_opens_with_global_then_act(self):
        surface = _build(_record())
        assert surface.slots[0].kind is SlotKind.GLOBAL
        assert surface.slots[1].kind is SlotKind.ACT
        assert ACT_SLOT == 1

    def test_players_come_before_cards(self):
        kinds = [slot.kind for slot in _build(_record()).slots]
        assert kinds.index(SlotKind.PLAYER) < kinds.index(SlotKind.CARD)

    def test_every_player_and_entity_gets_a_slot(self):
        surface = _build(_record())
        assert len(slots_of_kind(surface, SlotKind.PLAYER)) == 2
        assert len(slots_of_kind(surface, SlotKind.CARD)) == 2

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
        for slot in slots_of_kind(surface, SlotKind.CARD):
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

    def test_a_decision_record_puts_its_candidates_vector_in_act(self):
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
        assert _build(record).slots[1].e == _VECTORS[_BOLT]

    def test_an_unresolvable_key_contributes_zeros_rather_than_dropping(self):
        missing = ProvenanceKey("cardsfolder/x/absent.txt", 0, "spell", 0)
        assert _build(_record(ability=(missing,))).slots[1].e == _ZERO

    def test_the_cost_halfs_outcome_is_no_input(self):
        """Whether the spell resolved is what a cost half predicts (T115)."""
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
        assert act_features(resolved) == act_features(countered)


class TestAbilityTokens:
    def test_printed_and_attachment_granted_lines_become_tokens(self):
        entity = _entity("E1", printed=(_ANTHEM,), granted_attached=(_WARD,))
        surface = _build(_record(state=_snapshot(entities=(entity,))))
        vectors = [s.e for s in slots_of_kind(surface, SlotKind.ABILITY)]
        assert vectors == [_VECTORS[_ANTHEM], _VECTORS[_WARD]]

    def test_temporary_grants_become_tokens_flagged_temporary(self):
        """Root spec § 9.2: a temporary grant is a line, in place of a count."""
        entity = _entity(
            "E1", printed=(_ANTHEM,),
            granted_temporary=GrantedTemporary(abilities=(_WARD,)),
        )
        surface = _build(_record(state=_snapshot(entities=(entity,))))
        abilities = slots_of_kind(surface, SlotKind.ABILITY)
        assert [slot.e for slot in abilities] == [_VECTORS[_ANTHEM], _VECTORS[_WARD]]
        assert [_flags(slot)["temporary"] for slot in abilities] == [0.0, 1.0]

    def test_a_gen3_entity_builds_the_surface_its_gen2_twin_builds(self):
        """The lines in force read as the gen-2 twin's tokens, printed or
        attached; an unmet static is one more token, flagged inactive (T074)."""
        from effects.domain.state_snapshot import CardLine

        lines = (
            CardLine(keys=(_ANTHEM,), kind="static", granted=False, active=True),
            CardLine(keys=(_WARD,), kind="static", granted=True, active=True),
            CardLine(keys=(_BOLT,), kind="static", granted=False, active=False),
        )
        gen3 = _build(_record(state=_snapshot(entities=(_entity("E1", lines=lines),))))
        gen2 = _build(_record(state=_snapshot(entities=(
            _entity("E1", printed=(_ANTHEM,), granted_attached=(_WARD,)),))))
        assert gen3.slots[:-1] == gen2.slots
        inactive = gen3.slots[-1]
        assert inactive.e == _VECTORS[_BOLT] and _flags(inactive)["active"] == 0.0

    def test_context_dropout_can_drop_a_context_ability(self):
        record = _record()
        kept = _build(record, context_dropout=0.0)
        dropped = _build(record, context_dropout=1.0, rng=random.Random(1))
        assert len(slots_of_kind(dropped, SlotKind.ABILITY)) < len(
            slots_of_kind(kept, SlotKind.ABILITY)
        )

    def test_the_acting_entitys_abilities_are_never_dropped(self):
        """Dropping the ability under study would make the record unanswerable."""
        state = _snapshot(refs=Refs(source="E1"))
        surface = _build(
            _record(state=state), context_dropout=1.0, rng=random.Random(1),
        )
        acting = [s for s in slots_of_kind(surface, SlotKind.ABILITY) if s.entity_id == "E1"]
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
        # Acting, not acting, active, holding priority and targeted sit just
        # before the life thermometer; P0 is active and holds priority.
        flags = slice(-5 - THERMOMETER_STEPS, -THERMOMETER_STEPS)
        assert mine[flags] == (1.0, 0.0, 1.0, 1.0, 0.0)
        assert theirs[flags] == (0.0, 1.0, 0.0, 0.0, 0.0)

    def test_a_targeted_player_is_flagged(self):
        state = _snapshot(refs=Refs(targets=("P1",)))
        record = _record(state=state)
        assert player_features(state.players[1], record)[-1 - THERMOMETER_STEPS] == 1.0

    def test_the_global_slot_flags_whose_turn_it_is(self):
        mine = global_features(_record())
        theirs = global_features(_record(actor_player="P1"))
        assert mine != theirs


class TestNumericEncoding:
    @staticmethod
    def _scalar(value: float) -> list[float]:
        out: list[float] = []
        _push_scalar(out, value)
        return out

    def test_a_scalar_enters_raw_plus_log1p(self):
        assert self._scalar(3) == pytest.approx([3.0, math.log1p(3)])

    def test_the_log_copy_is_signed_so_it_stays_monotone_across_zero(self):
        assert self._scalar(-3)[1] == pytest.approx(-math.log1p(3))
        assert self._scalar(-3)[1] < self._scalar(0)[1] < self._scalar(3)[1]

    def test_nothing_is_binned(self):
        """Two nearby values give two different feature vectors."""
        assert self._scalar(3) != self._scalar(4)

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
        # Only the flying flag differs.
        differing = [i for i, (a, b) in enumerate(zip(masked, without)) if a != b]
        assert len(differing) == 1

    def test_a_granted_keyword_is_addressable_for_the_gate_two_perturbation(self):
        for keyword in ("first_strike", "double_strike", "deathtouch", "lifelink",
                        "trample", "indestructible", "wither", "infect"):
            assert keyword in OVERLAY_KEYWORDS

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
        widths = {len(slot.features) for slot in slots_of_kind(surface, SlotKind.CARD)}
        assert len(widths) == 1

    def test_every_player_slot_has_the_same_width(self):
        surface = _build(_record())
        widths = {len(slot.features) for slot in slots_of_kind(surface, SlotKind.PLAYER)}
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
        abilities = slots_of_kind(surface, SlotKind.ABILITY)
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
        surface = _build(record, e_for=lambda key: None, has_line=lambda key: True)
        assert [slot.e for slot in slots_of_kind(surface, SlotKind.ABILITY)] == [_ZERO]

    def test_without_the_predicate_every_key_keeps_its_token(self):
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._PHANTOM,)),
        )))
        assert [slot.e for slot in slots_of_kind(_build(record), SlotKind.ABILITY)] == [_ZERO]


class TestOptionRows:
    """FR-063a: a charm's mode rows follow its root row inside the card's block."""

    _CHARM = ProvenanceKey("cardsfolder/c/cryptic_command.txt", 0, "spell", 0)

    def _modes(self):
        return tuple(
            ProvenanceKey(self._CHARM.script_file, 0, "spell", 0, option)
            for option in range(3)
        )

    def _surface(self, options_for):
        vectors = {self._CHARM: (0.5,) * E_DIM}
        vectors |= {mode: (float(mode.option),) * E_DIM for mode in self._modes()}
        record = _record(state=_snapshot(entities=(
            _entity("E1", printed=(self._CHARM, _ANTHEM)),
        )))
        return _build(
            record, e_for=lambda key: vectors.get(key, _VECTORS.get(key)),
            options_for=options_for,
        )

    def test_a_charm_gets_its_root_then_its_modes_in_choices_order(self):
        modes = self._modes()
        surface = self._surface(
            lambda key: modes if key == self._CHARM else (),
        )
        abilities = slots_of_kind(surface, SlotKind.ABILITY)
        assert [slot.option for slot in abilities] == [False, True, True, True, False]
        assert [slot.e[0] for slot in abilities[1:4]] == [0.0, 1.0, 2.0]
        assert [slot.position for slot in abilities] == [1, 2, 3, 4, 5]

    def test_no_options_callback_adds_no_rows(self):
        abilities = slots_of_kind(self._surface(None), SlotKind.ABILITY)
        assert [slot.option for slot in abilities] == [False, False]

    def test_the_batcher_reads_the_modes_off_the_sidecar(self):
        import torch

        from effects.domain.baselines import BaselineMasks
        from effects.domain.provenance import ProvenanceSidecar, SidecarLine
        from tests.unit.effects.builders import make_batcher

        modes = self._modes()
        sidecar = ProvenanceSidecar(
            card="cryptic command", script_file=self._CHARM.script_file,
            lines=(
                SidecarLine(3, "spell", (self._CHARM,), script_text="Choices$ SV1,SV2,SV3"),
                *(SidecarLine(4 + m.option, "option", (m,), script_text=f"SV1: DB$ M{m.option}")
                  for m in modes),
                SidecarLine(7, "option", (), script_text=""),
            ),
        )

        class _Sidecars:
            def get(self, script_file):
                return sidecar

            def line_for(self, key):
                return sidecar.line_for(key)

            def prose_for(self, key):
                return None

        batcher = make_batcher(
            tokenizer=None, sidecars=_Sidecars(), masks=BaselineMasks(),
            surface="script", e_dim=E_DIM, widths={}, device=torch.device("cpu"),
        )
        assert batcher.options_for(self._CHARM) == modes
        assert batcher.options_for(modes[0]) == ()


class TestGen2Layout:
    """The gen-2 input layout reads the collector's names and compares amounts (FR-063c).

    The spellings are the collector's own, as a real gen-2 shard writes them:
    Forge's ``combat_*`` phase names, ``+1/+1`` counters, and keywords already
    normalized at parse time (``protection_from_red``, ``landwalk``).
    """

    @staticmethod
    def _phase_slot(record):
        features = global_features(record)
        # turn and stack size are two scalars each; the phase one-hot follows.
        return features[4:4 + len(PHASES) + 1]

    def test_a_forge_combat_phase_reaches_its_own_slot(self):
        state = _snapshot(global_=GlobalState(
            turn=3, phase="combat_declare_attackers", active="P0",
            priority="P0", stack_size=0,
        ))
        record = _record(state=state)
        assert self._phase_slot(record)[PHASES.index("declare_attackers")] == 1.0

    def test_every_forge_combat_phase_name_is_mapped(self):
        for forge_name in (
            "combat_begin", "combat_declare_attackers", "combat_declare_blockers",
            "combat_first_strike_damage", "combat_end",
        ):
            state = _snapshot(global_=GlobalState(
                turn=3, phase=forge_name, active="P0", priority="P0",
                stack_size=0,
            ))
            assert self._phase_slot(_record(state=state))[-1] == 0.0

    def test_plus_one_counters_read_as_p1p1(self):
        assert counter_rows(_entity("E1", counters={"+1/+1": 2})) == counter_rows(
            _entity("E1", counters={"P1P1": 2}))

    def test_protection_from_a_colour_and_landwalk_set_their_flags(self):
        entity = _entity("E1", granted_temporary=GrantedTemporary(
            keywords=("protection_from_red", "landwalk"),
        ))
        record = _record(state=_snapshot(entities=(entity,)))
        block = card_features(entity, record)[card_segments(entity, record)["keyword_flags"]]
        assert block[KEPT_KEYWORD_FLAGS.index("protection")] == 1.0
        assert block[KEPT_KEYWORD_FLAGS.index("landwalk")] == 1.0
        assert sum(block) == 2

    def test_the_card_ends_with_power_toughness_and_remaining_toughness(self):
        entity = _entity(
            "E1", pt=PowerToughness(base=(2, 3), boosts=(1, 1), counters=(1, 1)),
            damage=2,
        )
        record = _record(state=_snapshot(entities=(entity,)))
        features = card_features(entity, record)
        steps = THERMOMETER_STEPS
        power, toughness, remaining = (
            features[-3 * steps:-2 * steps],
            features[-2 * steps:-steps],
            features[-steps:],
        )
        assert sum(power) == 4 and power[3] == 1.0 and power[4] == 0.0
        assert sum(toughness) == 5
        assert sum(remaining) == 3

    def test_an_entity_without_power_and_toughness_carries_zero_codes(self):
        entity = _entity("E1", types=("Enchantment",), pt=None)
        record = _record(state=_snapshot(entities=(entity,)))
        features = card_features(entity, record)
        assert features[-3 * THERMOMETER_STEPS:] == (0.0,) * (3 * THERMOMETER_STEPS)

    def test_the_player_ends_with_a_life_thermometer(self):
        record = _record()
        features = player_features(record.state.players[1], record)
        assert sum(features[-THERMOMETER_STEPS:]) == 8  # P1 is at 8 life

    def test_life_above_the_starting_twenty_reads_in_full(self):
        players = (
            PlayerState(id="P0", life=33, hand=3, library=21, graveyard=9),
            PlayerState(id="P1", life=8, hand=1, library=30, graveyard=4),
        )
        record = _record(state=_snapshot(players=players))
        features = player_features(players[0], record)
        assert sum(features[-THERMOMETER_STEPS:]) == 33


# ── gen-3 inputs (T111-T113) ──────────────────────────────────────────────

_CHOICE = ProvenanceKey("cardsfolder/c/charm.txt", 0, "spell", 0)
_KICKER = ProvenanceKey("cardsfolder/k/kicker.txt", 0, "keyword", 0)


def _line(key, kind="static", granted=False, **fields):
    from effects.domain.state_snapshot import CardLine

    return CardLine(keys=(key,) if key else (), kind=kind, granted=granted, **fields)


def _card_part(entity, record, name):
    return card_features(entity, record)[card_segments(entity, record)[name]]


class TestLineFlags:
    """Root spec § 9.2: every ability slot carries its line's flags."""

    def _abilities(self, entity, **build):
        return slots_of_kind(
            _build(_record(state=_snapshot(entities=(entity,))), **build), SlotKind.ABILITY)

    def test_every_ability_slot_has_the_flags_width(self):
        from effects.domain.effect_head_input import ABILITY_FEATURE_WIDTH

        entity = _entity("E1", lines=(_line(_ANTHEM), _line(_WARD, active=False)))
        assert {len(slot.features) for slot in self._abilities(entity)} == {
            ABILITY_FEATURE_WIDTH}

    def test_a_face_not_up_is_slots_flagged_inactive(self):
        entity = _entity("E1", lines=(
            _line(_ANTHEM, active=True), _line(_WARD, kind="trigger", active=False)))
        assert [_flags(slot)["active"] for slot in self._abilities(entity)] == [1.0, 0.0]

    def test_a_grant_by_a_static_is_granted_but_not_temporary(self):
        entity = _entity("E1", lines=(_line(_ANTHEM, kind="activated", granted=True),))
        flags = _flags(self._abilities(entity)[0])
        assert (flags["granted"], flags["temporary"]) == (1.0, 0.0)

    def test_a_keyword_an_effect_granted_reads_through_its_reminder_line(self):
        from effects.domain.effect_head_input import keyword_line_fragment

        entity = _entity("E1", lines=(
            _line(None, kind="keyword", granted=True, fragment="Flying"),))
        keys = []
        abilities = self._abilities(entity, e_for=lambda key: keys.append(key) or None)
        flags = _flags(abilities[0])
        assert (flags["granted"], flags["temporary"]) == (1.0, 1.0)
        assert keyword_line_fragment(keys[-1]) == "Flying"

    def test_a_line_with_no_key_and_no_keyword_has_no_slot(self):
        entity = _entity("E1", lines=(_line(None, kind="trigger"), _line(_ANTHEM)))
        assert len(self._abilities(entity)) == 1

    def test_cast_with_carries_the_multikicker_count(self):
        entity = _entity("E1", lines=(_line(_KICKER, kind="keyword", cast_with=3),))
        flags = _flags(self._abilities(entity)[0])
        assert flags["cast_with"] == 1.0 and flags["count"] == 3.0

    def test_the_chosen_flag_sits_on_the_chosen_mode(self):
        modes = tuple(ProvenanceKey(_CHOICE.script_file, 0, "spell", 0, n) for n in range(3))
        entity = _entity("E1", lines=(_line(_CHOICE, kind="spell", chosen=(0, 2)),))
        abilities = self._abilities(
            entity, options_for=lambda key: modes if key == _CHOICE else ())
        assert [_flags(slot)["chosen"] for slot in abilities] == [0.0, 1.0, 0.0, 1.0]

    def test_a_keyword_counter_is_a_granted_keyword_line_not_a_counter(self):
        from effects.domain.effect_head_input import counter_id, keyword_line_fragment, slot_lines

        entity = _entity("E1", counters={"FLYING": 1, "P1P1": 2})
        lines = slot_lines(entity)
        assert [keyword_line_fragment(line.key) for line in lines] == ["FLYING"]
        assert lines[0].granted and not lines[0].temporary
        assert counter_rows(entity) == ((counter_id("P1P1"), 2.0),)

    def test_a_keyword_counter_adds_no_line_where_the_keyword_has_one(self):
        from effects.domain.effect_head_input import slot_lines

        entity = _entity("E1", counters={"FLYING": 1}, lines=(
            _line(None, kind="keyword", granted=True, fragment="Flying"),))
        assert len(slot_lines(entity)) == 1

    def test_stripping_a_keyword_drops_its_granted_line_and_its_flag(self):
        entity = _entity(
            "E1", granted_temporary=GrantedTemporary(keywords=("first_strike",)),
            lines=(_line(None, kind="keyword", granted=True, fragment="First Strike"),
                   _line(_ANTHEM)),
        )
        record = _record(state=_snapshot(entities=(entity,)))
        base = _build(record)
        stripped = _build(record, stripped={"E1": frozenset({"first_strike"})})
        assert len(slots_of_kind(base, SlotKind.ABILITY)) == 2
        assert len(slots_of_kind(stripped, SlotKind.ABILITY)) == 1
        flags = card_segments(entity, record)["keyword_flags"]
        assert sum(slots_of_kind(stripped, SlotKind.CARD)[0].features[flags]) == 0.0
        assert sum(slots_of_kind(base, SlotKind.CARD)[0].features[flags]) == 1.0


class TestKeywordFlags:
    """Root spec § 9.6: only the kept keywords keep a flag."""

    def test_a_keyword_without_a_kept_flag_sets_none(self):
        assert "exalted" not in KEPT_KEYWORD_FLAGS
        entity = _entity("E1", granted_temporary=GrantedTemporary(keywords=("exalted",)))
        record = _record(state=_snapshot(entities=(entity,)))
        assert sum(_card_part(entity, record, "keyword_flags")) == 0.0

    def test_split_second_keeps_its_flag(self):
        entity = _entity("E1", granted_temporary=GrantedTemporary(keywords=("split_second",)))
        record = _record(state=_snapshot(entities=(entity,)))
        flags = _card_part(entity, record, "keyword_flags")
        assert flags[KEPT_KEYWORD_FLAGS.index("split_second")] == 1.0

    def test_an_in_force_keyword_line_sets_its_flag_and_a_face_not_up_none(self):
        up = _entity("E1", lines=(_line(None, kind="keyword", fragment="Flying"),))
        down = _entity("E1", lines=(
            _line(None, kind="keyword", fragment="Flying", active=False),))
        record = _record()
        index = KEPT_KEYWORD_FLAGS.index("flying")
        assert _card_part(up, record, "keyword_flags")[index] == 1.0
        assert _card_part(down, record, "keyword_flags")[index] == 0.0


class TestCardSlotInputs:
    """Root spec § 9.4: the card slot's new inputs."""

    def test_face_state_replaces_the_face_down_flag(self):
        from effects.domain.effect_head_input import FACE_STATES

        record = _record()
        for face in FACE_STATES:
            part = _card_part(_entity("E1", face_state=face), record, "face_state")
            assert part[FACE_STATES.index(face)] == 1.0 and sum(part) == 1.0
        gen2 = _card_part(_entity("E1", face_down=True), record, "face_state")
        assert gen2[FACE_STATES.index("face_down")] == 1.0

    def test_two_face_down_cards_differ_only_in_their_inactive_lines(self):
        from effects.domain.state_snapshot import RealFace

        def face_down(name, key):
            return _entity(
                "E1", name="", face_state="face_down", types=("creature",), colors=(),
                mana_value=0, pt=PowerToughness(base=(2, 2)),
                real_face=RealFace(name=name, types=("creature",), colors=("R",),
                                   mana_value=5, base_pt=(5, 4)),
                lines=(_line(key, kind="trigger", active=False),),
            )

        one = _build(_record(state=_snapshot(entities=(face_down("A", _ANTHEM),))))
        two = _build(_record(state=_snapshot(entities=(face_down("B", _WARD),))))
        assert slots_of_kind(one, SlotKind.CARD) == slots_of_kind(two, SlotKind.CARD)
        one_line, = slots_of_kind(one, SlotKind.ABILITY)
        two_line, = slots_of_kind(two, SlotKind.ABILITY)
        assert one_line.features == two_line.features and one_line.e != two_line.e

    def test_is_a_copy_and_is_a_token(self):
        record = _record()
        plain = _card_part(_entity("E1"), record, "characteristics")
        copy = _card_part(_entity("E1", copy_source="E9"), record, "characteristics")
        token = _card_part(_entity("E1", token_script_id="c_1_1_soldier"), record,
                           "characteristics")
        assert (plain[-2:], copy[-2:], token[-2:]) == ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    def test_mana_cost_reads_each_symbol(self):
        from effects.domain.effect_head_input import MANA_SYMBOLS

        entity = _entity("E1", mana_cost={"generic": 2, "W/U": 1, "G/P": 2})
        part = _card_part(entity, _record(), "mana_cost")
        assert part[0] == 1.0
        assert part[1 + 2 * MANA_SYMBOLS.index("W/U")] == 1.0
        assert part[1 + 2 * MANA_SYMBOLS.index("G/P")] == 2.0
        assert part[1 + 2 * MANA_SYMBOLS.index("generic")] == 2.0
        assert _card_part(_entity("E1"), _record(), "mana_cost")[0] == 0.0

    def test_printed_loyalty_carries_a_thermometer(self):
        part = _card_part(_entity("E1", printed_loyalty=3), _record(),
                          "printed_loyalty_defense")
        loyalty = part[:3 + THERMOMETER_STEPS]
        assert loyalty[:2] == (1.0, 3.0) and sum(loyalty[3:]) == 3

    def test_the_designations_are_the_collector_s_table(self):
        """``CardFacts.designationTable`` writes the closed list the input reads."""
        import re
        from pathlib import Path

        from effects.domain.effect_head_input import DESIGNATIONS

        java = (Path(__file__).resolve().parents[4] / "forge-connector/src/main/java/com/"
                "pricepredictor/connector/effects/CardFacts.java").read_text(encoding="utf-8")
        assert tuple(re.findall(r'table\.put\("([a-z_]+)"', java)) == DESIGNATIONS

    def test_designations_x_paid_history_and_order_rank(self):
        from effects.domain.effect_head_input import DESIGNATIONS
        from effects.domain.state_snapshot import CardHistory

        entity = _entity(
            "E1", designations=("goaded", "monstrous"), x_paid=4, order_rank=2,
            history=CardHistory(entered_this_turn=True, loyalty_activated=True),
        )
        record = _record()
        designations = _card_part(entity, record, "designations")
        assert designations[DESIGNATIONS.index("goaded")] == 1.0
        assert sum(designations) == 2.0
        assert _card_part(entity, record, "x_paid")[:2] == (1.0, 4.0)
        assert _card_part(entity, record, "history") == (1.0, 0.0, 0.0, 1.0)
        assert _card_part(entity, record, "order_rank")[:2] == (1.0, 2.0)

    def test_chosen_values_and_costs_paid(self):
        from effects.domain.state_snapshot import COLORS, CardChoices, CostsPaid

        entity = _entity(
            "E1", chosen=CardChoices(colors=("white",), type="Human", number=3),
            costs_paid=CostsPaid(mana={"B": 2}, cast_from="graveyard", mana_rider=True),
        )
        record = _record()
        chosen = _card_part(entity, record, "chosen")
        assert chosen[COLORS.index("W")] == 1.0
        paid = _card_part(entity, record, "costs_paid")
        assert paid[0] == 1.0 and paid[1 + 2 * COLORS.index("B")] == 2.0 and paid[-1] == 1.0

    def test_subtypes_and_chosen_types_share_the_subtype_rows(self):
        from effects.domain.effect_head_input import SUBTYPE_ROWS, subtype_id
        from effects.domain.state_snapshot import CardChoices

        entity = _entity("E1", subtypes=("human", "soldier"),
                         chosen=CardChoices(type="Human"))
        card, = slots_of_kind(_build(_record(state=_snapshot(entities=(entity,)))),
                              SlotKind.CARD)
        assert card.subtypes == (subtype_id("human"), subtype_id("soldier"))
        assert card.chosen_types == (subtype_id("Human"),) == card.subtypes[:1]
        assert subtype_id("not a subtype") == SUBTYPE_ROWS - 1

    def test_a_chosen_card_type_is_no_subtype(self):
        from effects.domain.effect_head_input import chosen_subtypes
        from effects.domain.state_snapshot import CardChoices

        assert chosen_subtypes(_entity("E1", chosen=CardChoices(type="Artifact"))) == ()

    def test_every_named_counter_has_a_row_of_its_own(self):
        from effects.domain.effect_head_input import COUNTER_ROWS, counter_id

        rows = dict(counter_rows(_entity("E1", counters={"OIL": 2, "+1/+1": 1, "ZZZ": 4})))
        assert rows[counter_id("OIL")] == 2.0 and rows[counter_id("P1P1")] == 1.0
        assert rows[COUNTER_ROWS - 1] == 4.0
        assert len({counter_id("OIL"), counter_id("P1P1"), counter_id("ACORN")}) == 3

    def test_turn_and_control_inputs(self):
        record = _record(state=_snapshot(global_=GlobalState(
            turn=2, phase="main1", active="P1", priority="P0", stack_size=0)))
        stolen = _entity("E1", controller="P0", owner="P1")
        part = _card_part(stolen, record, "relation_to_act")
        # controller active, controller holds priority, owner acting, owner differs
        assert part[3:] == (0.0, 1.0, 0.0, 1.0)


class TestPlayerGlobalAndActInputs:
    """Root spec § 9.4-§ 9.5 and the land-play limit of FR-044."""

    def test_max_land_plays_beside_lands_played_and_unlimited(self):
        from effects.domain.effect_head_input import THIS_TURN_COUNTERS

        record = _record()
        at = 6 * 2 + len(THIS_TURN_COUNTERS) * 2

        def limit(player):
            return player_features(player, record)[at:at + 3]

        two = PlayerState(id="P0", life=20, hand=0, library=0, graveyard=0, max_land_plays=2)
        free = PlayerState(id="P0", life=20, hand=0, library=0, graveyard=0,
                           land_plays_unlimited=True)
        assert limit(two) == pytest.approx((2.0, math.log1p(2), 0.0))
        assert limit(free) == (0.0, 0.0, 1.0)

    def test_player_designations_reach_the_slot(self):
        record = _record()
        plain = PlayerState(id="P0", life=20, hand=0, library=0, graveyard=0)
        for change in ({"speed": 2}, {"rad": 3}, {"monarch": True},
                       {"initiative": True}, {"city_blessing": True}):
            assert player_features(replace(plain, **change), record) != player_features(
                plain, record), change

    def test_day_night_and_the_pending_event_reach_the_global_slot(self):
        from effects.domain.effect_head_input import DAY_NIGHT, EVENT_TYPES

        night = _record(state=_snapshot(global_=GlobalState(
            turn=2, phase="main1", active="P0", priority="P0", stack_size=0,
            day_night="night")))
        # turn, stack size, the phase and substep one-hots, the two actor
        # flags and the emblem count come first.
        from effects.domain.effect_head_input import COMBAT_SUBSTEPS

        start = 4 + len(PHASES) + 1 + len(COMBAT_SUBSTEPS) + 1 + 2 + 2
        assert global_features(night)[start + DAY_NIGHT.index("night")] == 1.0
        pending = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 3})
        record = _record(kind=RecordKind.TRIGGER, moment=None,
                         state=_snapshot(pending_event=pending),
                         payload=TriggerPayload(event=pending, fired=True))
        at = start + len(DAY_NIGHT) + 1
        event = global_features(record)[at:at + len(EVENT_TYPES) + 3]
        assert event[EVENT_TYPES.index("damage_dealt")] == 1.0
        assert event[-2] == 3.0

    def test_dice_and_coins_reach_the_act_slot(self):
        from effects.domain.state_snapshot import RandomResults

        rolled = _record(state=_snapshot(refs=Refs(
            random=RandomResults(dice=(4, 6), coin_flips_won=1))))
        assert act_features(rolled) != act_features(_record())
        assert len(act_features(rolled)) == len(act_features(_record()))


class TestTheGen3Fixture:
    """The real gen-3 records lay out at one width per slot kind."""

    @pytest.fixture(scope="class")
    def records(self):
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import gen3_fixture_records

        return [record_from_dict(raw) for raw in gen3_fixture_records()]

    def test_every_slot_kind_keeps_one_width(self, records):
        from effects.domain.effect_head_input import feature_widths

        widths = feature_widths(records)
        seen: dict = {}
        for record in records:
            for slot in _build(record).slots:
                seen.setdefault(slot.kind, set()).add(len(slot.features))
        assert seen == {kind: {widths[kind]} for kind in seen}

    def test_the_fixture_reaches_the_new_inputs(self, records):
        from effects.domain.effect_head_input import keyword_line_fragment, slot_lines

        entities = [entity for record in records for entity in record.state.entities]
        lines = [line for entity in entities for line in slot_lines(entity)]
        assert any(not line.active for line in lines)
        assert any(line.temporary and keyword_line_fragment(line.key) for line in lines)
        assert any(counter_rows(entity) for entity in entities)
        assert any(entity.chosen and entity.chosen.type for entity in entities)


# ── command-zone slots and relations (T114-T117) ──────────────────────────


@pytest.fixture(scope="module")
def gen3_records():
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import gen3_fixture_records

    return {record.record_id: record
            for record in map(record_from_dict, gen3_fixture_records())}


_RUN = "e8aa2e54-4112-425c-a7fa-96c7bb7340db"


def _relation(family, direction="forward", index=None):
    from effects.domain.relation_types import relation_type

    return relation_type(family, direction, index)


def _subject_slot(surface, subject):
    """The slot of the card or player ``subject`` names."""
    return next(index for index, slot in enumerate(surface.slots)
                if slot.kind in (SlotKind.CARD, SlotKind.PLAYER) and slot.subject_id == subject)


def _linked(surface, a, b, family, index=None) -> bool:
    """Whether the surface relates slots ``a`` and ``b`` by ``family``, both ways."""
    relations = set(surface.relations)
    return ((a, b, _relation(family, "forward", index)) in relations
            and (b, a, _relation(family, "reverse", index)) in relations)


def _gen3_entity(entity_id, name="Serra Angel", lines=(), **fields):
    return _entity(entity_id, name=name, lines=lines, **fields)


def _gen3_record(entities, links=(), players=None, **snapshot):
    """A gen-3 record: a snapshot with ``links`` is one."""
    defaults = {"entities": tuple(entities), "links": tuple(links)}
    if players is not None:
        defaults["players"] = players
    defaults.update(snapshot)
    return _record(state=_snapshot(**defaults))


class TestAttachedTo:
    """Story 5 acceptance 1 (T115, T116)."""

    def test_an_aura_relates_to_its_creature_and_no_card_input_says_so(self, gen3_records):
        record = gen3_records[f"{_RUN}.0-mvyv5jxepqmy.516"]
        aura = record.state.entity("E24")
        assert "aura" in aura.subtypes and aura.attached_to == "E22"
        surface = _build(record)
        assert _linked(surface, _subject_slot(surface, "E24"), _subject_slot(surface, "E22"),
                       "attached_to")
        assert "attached" not in card_segments(aura, record)
        assert card_features(replace(aura, attached_to=None), record) == card_features(
            aura, record)


_FALTER = ProvenanceKey("cardsfolder/f/falter.txt", 0, "spell", 0)


class TestCommandZoneSlots:
    """Root spec § 9.3 (T114) and story 5 acceptance 2."""

    def _falter_record(self, **effect):
        from effects.domain.state_snapshot import CommandZoneObject, LineRef, Link

        floating = CommandZoneObject(
            id="F40", source_lines=(_FALTER,), affected=("E1", "E2"),
            duration=effect.pop("duration", "end_of_turn"), **effect)
        caster = _gen3_entity("E3", name="Falter", zone="graveyard",
                              lines=(_line(_FALTER, kind="spell"),))
        return _gen3_record(
            (_gen3_entity("E1"), _gen3_entity("E2", controller="P1"), caster),
            links=(Link("F40", "E1", "affects"), Link("F40", "E2", "affects"),
                   Link("F40", LineRef("E3", _FALTER), "created_by")),
            global_=GlobalState(turn=3, phase="main1", active="P0", priority="P0",
                                stack_size=0, floating_effects=(floating,)),
        )

    def test_creatures_cant_block_this_turn_is_a_command_zone_card(self):
        from effects.domain.effect_head_input import EFFECT_DURATIONS, command_zone_entities

        record = self._falter_record()
        falter_e = (0.5, 0.5, 0.0, 0.0)
        surface = _build(record, e_for=lambda key: falter_e if key == _FALTER else None)
        at = _subject_slot(surface, "F40")
        assert at not in surface.entity_slots
        line = surface.slots[at + 1]
        assert line.kind is SlotKind.ABILITY and line.entity_id == "F40"
        assert line.e == falter_e and _flags(line)["granted"] == 1.0
        for creature in ("E1", "E2"):
            assert _linked(surface, at, _subject_slot(surface, creature), "affects")
        # created_by ends at the casting line's own slot, not its card.
        caster_line = _subject_slot(surface, "E3") + 1
        assert _linked(surface, at, caster_line, "created_by")
        effect, = command_zone_entities(record)
        assert effect.controller == "P0" and effect.zone == "command"
        duration = _card_part(effect, record, "command_zone")
        assert duration[EFFECT_DURATIONS.index("end_of_turn")] == 1.0
        assert sum(duration[:len(EFFECT_DURATIONS) + 1]) == 1.0

    def test_a_shield_reads_what_it_has_left(self):
        from effects.domain.effect_head_input import EFFECT_DURATIONS, command_zone_entities

        record = self._falter_record(duration="until_leaves", shield_left=3)
        effect, = command_zone_entities(record)
        part = _card_part(effect, record, "command_zone")
        assert part[EFFECT_DURATIONS.index("until_leaves")] == 1.0
        assert part[len(EFFECT_DURATIONS) + 1:] == pytest.approx((1.0, 3.0, math.log1p(3)))
        # Any other card reads zeros there.
        assert sum(_card_part(record.state.entities[0], record, "command_zone")) == 0.0

    def test_an_effect_card_on_the_board_reads_its_duration_and_gets_no_second_slot(self):
        from effects.domain.effect_head_input import EFFECT_DURATIONS
        from effects.domain.state_snapshot import CommandZoneObject

        card = _gen3_entity("E81", name="Regeneration", zone="command", lines=(
            _line(_FALTER, kind="trigger", granted=True),))
        record = _gen3_record((card,), global_=GlobalState(
            turn=3, phase="main1", active="P0", priority="P0", stack_size=0,
            floating_effects=(CommandZoneObject(id="E81", source_lines=(_FALTER,)),)))
        surface = _build(record)
        assert len(slots_of_kind(surface, SlotKind.CARD)) == 1
        assert _card_part(card, record, "command_zone")[
            EFFECT_DURATIONS.index("permanent")] == 1.0

    def test_a_real_floating_effect_affects_each_creature_it_pumps(self, gen3_records):
        record = gen3_records[f"{_RUN}.1-mvyw1pozylvp.13630"]
        floating = next(f for f in record.state.global_.floating_effects if f.id == "F91")
        rampage = floating.source_lines[0]
        vector = (0.0, 0.0, 0.0, 1.0)
        surface = _build(record, e_for=lambda key: vector if key == rampage else None)
        at = _subject_slot(surface, "F91")
        assert surface.slots[at + 1].e == vector
        assert floating.affected == ("E6", "E30", "E26", "E22", "E8")
        for creature in floating.affected:
            assert _linked(surface, at, _subject_slot(surface, creature), "affects")

    def test_a_real_delayed_trigger_is_a_card_reading_its_source_line(self, gen3_records):
        record = gen3_records[f"{_RUN}.6-mvyvq4x9ozb2.53904"]
        trigger, = record.state.global_.delayed_triggers
        vector = (0.0, 1.0, 1.0, 0.0)
        surface = _build(record, e_for=lambda key: vector if key in trigger.source_lines else None)
        at = _subject_slot(surface, trigger.id)
        assert surface.slots[at + 1].e == vector
        # Its rebounding card sits in exile, which no tier reaches: the
        # pending_on link names an object the surface has no slot for.
        assert record.state.entity("E12") is None

    def test_a_delayed_trigger_is_pending_on_its_card(self):
        from effects.domain.state_snapshot import CommandZoneObject, Link

        record = _gen3_record((_gen3_entity("E1"),), links=(Link("D7", "E1", "pending_on"),),
                              global_=GlobalState(
                                  turn=3, phase="main1", active="P0", priority="P0",
                                  stack_size=0, delayed_triggers=(CommandZoneObject(
                                      id="D7", source_lines=(_FALTER,), remembered=("E1",)),)))
        surface = _build(record)
        assert _linked(surface, _subject_slot(surface, "D7"), _subject_slot(surface, "E1"),
                       "pending_on")

    @pytest.mark.parametrize(("record_id", "family", "a", "b"), [
        (f"{_RUN}.0-mvyv5jxepqmy.127", "last_known_state_of", "L35", "E35"),
        (f"{_RUN}.9-mvyw6uvu1ii5.232", "merged_with", "E16", "E15"),
        (f"{_RUN}.7-mvyvapedpsus.1303", "in_dungeon", "E96", "P1"),
    ])
    def test_the_extra_cards_are_slots_their_links_relate(
        self, gen3_records, record_id, family, a, b,
    ):
        surface = _build(gen3_records[record_id])
        assert _linked(surface, _subject_slot(surface, a), _subject_slot(surface, b), family)

    def test_a_library_card_reads_its_position_as_order_rank(self, gen3_records):
        record, entity = next(
            (record, entity) for record in gen3_records.values()
            for entity in record.state.entities
            if entity.role == "library_card" and entity.order_rank)
        assert _card_part(entity, record, "order_rank")[:2] == (1.0, float(entity.order_rank))
        surface = _build(record)
        assert _subject_slot(surface, entity.id) in surface.entity_slots


class TestRelations:
    """T116: relations from the record's links and from the reader's own layout."""

    def test_a_cards_lines_its_modes_and_the_lines_of_one_card(self):
        modes = tuple(ProvenanceKey(_CHOICE.script_file, 0, "spell", 0, n) for n in range(2))
        charm = _gen3_entity("E1", lines=(_line(_CHOICE, kind="spell"), _line(_ANTHEM)))
        surface = _build(_gen3_record((charm,)),
                         options_for=lambda key: modes if key == _CHOICE else ())
        card = _subject_slot(surface, "E1")
        root, first, second, anthem = card + 1, card + 2, card + 3, card + 4
        for line in (root, first, second, anthem):
            assert _linked(surface, line, card, "line_of")
        assert _linked(surface, first, root, "mode_of")
        assert _linked(surface, second, root, "mode_of")
        assert not _linked(surface, anthem, root, "mode_of")
        same = [t for t in surface.relations if t[2] in (
            _relation("line_of_the_same_card"), _relation("line_of_the_same_card", "reverse"))]
        assert len(same) == 4 * 3

    def test_same_names_and_a_chosen_name_relate_and_a_face_down_card_does_not(self):
        from effects.domain.state_snapshot import CardChoices

        record = _gen3_record((
            _gen3_entity("E1", name="Plains"), _gen3_entity("E2", name="plains"),
            _gen3_entity("E3", name="Island"),
            _gen3_entity("E4", name="Pithing Needle", chosen=CardChoices(names=("Plains",))),
            _gen3_entity("E5", name="Plains", face_state="face_down"),
        ))
        surface = _build(record)
        e1, e2, e3, e4, e5 = (_subject_slot(surface, f"E{n}") for n in range(1, 6))
        assert _linked(surface, e1, e2, "has_the_same_name")
        assert _linked(surface, e4, e1, "has_the_same_name")
        assert _linked(surface, e4, e2, "has_the_same_name")
        named = {end for t in surface.relations if t[2] in (
            _relation("has_the_same_name"), _relation("has_the_same_name", "reverse"))
            for end in t[:2]}
        assert named == {e1, e2, e4}
        assert e3 not in named and e5 not in named

    def test_the_act_slot_targets_by_index_and_reads_its_source(self):
        record = _gen3_record(
            (_gen3_entity("E1"), _gen3_entity("E2")),
            refs=Refs(targets=("E2", "P1"), source="E1"),
        )
        surface = _build(record)
        assert _linked(surface, ACT_SLOT, _subject_slot(surface, "E2"), "targets", 0)
        assert _linked(surface, ACT_SLOT, _subject_slot(surface, "P1"), "targets", 1)
        assert _linked(surface, _subject_slot(surface, "E1"), ACT_SLOT, "source_of")

    def test_a_target_index_past_the_last_type_shares_it(self):
        from effects.domain.relation_types import TARGET_INDICES
        from effects.domain.state_snapshot import LineRef, Link

        record = _gen3_record(
            (_gen3_entity("E1", lines=(_line(_BOLT, kind="spell"),)), _gen3_entity("E2")),
            links=(Link(LineRef("E1", _BOLT), "E2", "targets", 5),))
        surface = _build(record)
        line = _subject_slot(surface, "E1") + 1
        assert _linked(surface, line, _subject_slot(surface, "E2"), "targets",
                       TARGET_INDICES - 1)

    @pytest.mark.parametrize("why", ["no line", "stripped"])
    def test_a_link_from_a_line_with_no_slot_reads_its_card(self, why):
        from effects.domain.state_snapshot import LineRef, Link

        ward = _line(_WARD, kind="keyword", fragment="Ward:2")
        record = _gen3_record(
            (_gen3_entity("E1", lines=(ward, _line(_ANTHEM))), _gen3_entity("E2")),
            links=(Link(LineRef("E1", _WARD), "E2", "affects"),))
        if why == "no line":
            surface = _build(record, has_line=lambda key: key != _WARD)
        else:
            surface = _build(record, stripped={"E1": frozenset({"ward"})})
        card = _subject_slot(surface, "E1")
        assert [slot.entity_id for slot in slots_of_kind(surface, SlotKind.ABILITY)] == ["E1"]
        assert _linked(surface, card, _subject_slot(surface, "E2"), "affects")
        assert _linked(surface, card + 1, card, "line_of")

    def test_exploration_affects_its_controllers_player_slot(self):
        """Story 5 acceptance 8."""
        from effects.domain.effect_head_input import THIS_TURN_COUNTERS
        from effects.domain.state_snapshot import LineRef, Link

        exploration_static = ProvenanceKey("cardsfolder/e/exploration.txt", 0, "static", 0)
        exploration = _gen3_entity("E1", name="Exploration", types=("enchantment",), pt=None,
                                   lines=(_line(exploration_static, affected=("P0",)),))
        players = (
            PlayerState(id="P0", life=20, hand=3, library=30, graveyard=0,
                        this_turn={"lands_played": 1}, max_land_plays=2),
            PlayerState(id="P1", life=20, hand=3, library=30, graveyard=0, max_land_plays=1),
        )
        record = _gen3_record(
            (exploration,), players=players,
            links=(Link(LineRef("E1", exploration_static), "P0", "affects"),))
        surface = _build(record)
        player = _subject_slot(surface, "P0")
        lands = 6 * 2 + THIS_TURN_COUNTERS.index("lands_played") * 2
        limit = 6 * 2 + len(THIS_TURN_COUNTERS) * 2
        features = surface.slots[player].features
        assert features[lands] == 1.0 and features[limit] == 2.0
        static = _subject_slot(surface, "E1") + 1
        assert _linked(surface, static, player, "affects")
        assert not _linked(surface, static, _subject_slot(surface, "P1"), "affects")

    def test_every_fixture_link_between_two_slots_is_two_triples(self, gen3_records):
        checked = 0
        for record in gen3_records.values():
            surface = _build(record)
            subjects = {slot.subject_id: index for index, slot in enumerate(surface.slots)
                        if slot.kind in (SlotKind.CARD, SlotKind.PLAYER)}
            for found in record.state.links:
                a, b = subjects.get(found.from_), subjects.get(found.to)
                if a is None or b is None or a == b:
                    continue
                index = (found.index or 0) if found.type == "targets" else None
                assert _linked(surface, a, b, found.type, index), (record.record_id, found)
                checked += 1
        assert checked > 10_000

    def test_relations_survive_context_dropout_with_remapped_indices(self, gen3_records):
        record = max(gen3_records.values(),
                     key=lambda r: sum(1 for link in r.state.links
                                       if not isinstance(link.from_, str)))
        full = _build(record)
        dropped = _build(record, rng=random.Random(3), context_dropout=0.5)
        assert len(slots_of_kind(dropped, SlotKind.ABILITY)) < len(
            slots_of_kind(full, SlotKind.ABILITY))
        assert all(max(i, j) < len(dropped.slots) for i, j, _ in dropped.relations)

        line_of = _relation("line_of")
        for i, j, r in dropped.relations:
            if r == line_of:
                assert dropped.slots[i].kind is SlotKind.ABILITY
                assert dropped.slots[j].kind is SlotKind.CARD
                assert dropped.slots[i].entity_id == dropped.slots[j].entity_id

        def board(surface):
            """The triples between cards, players and [ACT], by what their ends
            stand for rather than where they sit."""
            out = set()
            for i, j, r in surface.relations:
                a, b = surface.slots[i], surface.slots[j]
                if SlotKind.ABILITY not in (a.kind, b.kind):
                    out.add((a.kind, a.subject_id, b.kind, b.subject_id, r))
            return out

        # A dropped line's links read its card: only ever more card-level triples.
        assert board(full) <= board(dropped)

    def test_two_face_down_cards_build_the_same_relations(self):
        from effects.domain.state_snapshot import RealFace

        def board(real_name):
            hidden = _gen3_entity(
                "E1", name="", face_state="face_down", types=("creature",),
                real_face=RealFace(name=real_name), lines=(_line(_ANTHEM, active=False),))
            return _gen3_record((hidden, _gen3_entity("E2", name=real_name)))

        assert _build(board("Serra Angel")).relations == _build(board("Grizzly Bears")).relations


class TestTheGen3FixtureInTheModel:
    """The fixture's widths are the model's, relations included (T117)."""

    def test_the_fixture_runs_through_the_model(self, gen3_records):
        import torch

        from effects.domain.effect_head_input import ABILITY_FEATURE_WIDTH, feature_widths
        from effects.domain.effect_model import EffectModel, EffectModelConfig
        from effects.domain.surface_collation import collate_surfaces

        records = list(gen3_records.values())[:16]
        widths = feature_widths(records)
        entity, record = records[0].state.entities[0], records[0]
        assert widths[SlotKind.CARD] == len(card_features(entity, record)) == max(
            part.stop for part in card_segments(entity, record).values())
        assert widths[SlotKind.ABILITY] == ABILITY_FEATURE_WIDTH
        surfaces = [_build(record) for record in records]
        inputs = collate_surfaces(surfaces, e_dim=E_DIM, widths=widths)
        assert inputs.relations.shape[0] == sum(len(s.relations) for s in surfaces) > 0
        torch.manual_seed(0)
        model = EffectModel(EffectModelConfig(
            global_features=widths[SlotKind.GLOBAL], act_features=widths[SlotKind.ACT],
            player_features=widths[SlotKind.PLAYER], card_features=widths[SlotKind.CARD],
            ability_features=widths[SlotKind.ABILITY], e_dim=E_DIM,
            d_model=16, n_layers=1, n_heads=2, ff_dim=32,
        ))
        hidden = model(inputs)
        assert hidden.shape[:2] == inputs.slot_kinds.shape
        assert torch.isfinite(hidden).all()
