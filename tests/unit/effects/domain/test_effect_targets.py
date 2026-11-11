"""Events to per-entity head targets.

The two rules that shape everything else: an entity is affected when any event
names it (not when some field changed — a spell that targets a creature and
deals 0 damage still affected it), and a field with no event is *absent* rather
than zero, so the loss skips it instead of learning that every unmentioned
counter went to zero.
"""

from __future__ import annotations

import functools
import subprocess
import sys

import pytest

from effects.domain.effect_head_input import CARD_TYPES, OVERLAY_KEYWORDS
from effects.domain.effect_head_layout import LIBRARY_EVENTS, ZONE_OUTCOMES
from effects.domain.effect_targets import derive_targets, events_of
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    Candidate,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    EffectRecord,
    ForbiddenEntity,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import COLORS
from tests.unit.effects.builders import fresh_interpreter_env, make_record


def _record(payload, kind=RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}
    defaults.update(overrides)
    return make_record(kind, payload=payload, **defaults)


def _resolution(*events) -> EffectRecord:
    return _record(ResolutionPayload(events=tuple(events)))


class TestTheGate:
    def test_an_entity_named_by_any_event_is_affected(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3}),
        ))
        assert targets["E1"].affected

    def test_an_entity_no_event_names_is_absent_from_the_targets(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3}),
        ))
        assert "E2" not in targets

    def test_a_zero_magnitude_event_still_affects_its_subject(self):
        """A spell that targets and deals 0 damage affected the creature."""
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 0}),
        ))
        assert targets["E1"].affected
        assert targets["E1"].fields["damage_taken"] == 0.0

    def test_one_event_can_affect_several_subjects(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1", "E2", "E3")),
        ))
        assert set(targets) == {"E1", "E2", "E3"}


class TestAbsentFields:
    def test_an_unmentioned_field_is_absent_not_zero(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2}),
        ))
        assert "damage_taken" in targets["E1"].fields
        assert "life_delta" not in targets["E1"].fields
        assert "counters_delta_p1p1" not in targets["E1"].fields

    def test_a_record_with_no_events_produces_no_targets(self):
        assert derive_targets(_resolution()) == {}


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


class TestMagnitudes:
    def test_damage_accumulates_across_events(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 2}),
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 3}),
        ))
        assert targets["E1"].fields["damage_taken"] == 5.0

    def test_healing_subtracts_from_damage(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 4}),
            Event(type=EventType.DAMAGE_HEALED, subjects=("E1",),
                  params={"amount": 4}),
        ))
        assert targets["E1"].fields["damage_taken"] == 0.0

    def test_life_delta_keeps_its_sign(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.LIFE_CHANGE, subjects=("P1",),
                  params={"delta": -3}),
        ))
        assert targets["P1"].fields["life_delta"] == -3.0

    def test_a_tracked_counter_type_gets_its_own_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "P1P1", "delta": 2}),
        ))
        assert targets["E1"].fields["counters_delta_p1p1"] == 2.0

    def test_an_untracked_counter_type_falls_into_the_catch_all(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "FLYING", "delta": 1}),
        ))
        assert targets["E1"].fields["counters_delta_other"] == 1.0

    def test_a_pt_change_splits_into_two_fields(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.PT_CHANGE, subjects=("E1",),
                  params={"power_delta": 3, "toughness_delta": -1}),
        ))
        assert targets["E1"].fields["power_delta"] == 3.0
        assert targets["E1"].fields["toughness_delta"] == -1.0

    def test_mana_lands_in_the_colours_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.MANA_PRODUCED, subjects=("P0",),
                  params={"mana_by_color": {"R": 2, "G": 1}}),
        ))
        assert targets["P0"].fields["mana_delta_r"] == 2.0
        assert targets["P0"].fields["mana_delta_g"] == 1.0

    def test_lost_mana_is_negative(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.MANA_LOST, subjects=("P0",),
                  params={"mana_by_color": {"U": 2}}),
        ))
        assert targets["P0"].fields["mana_delta_u"] == -2.0

    def test_poison_reaches_its_own_field(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.POISON_CHANGE, subjects=("P1",),
                  params={"delta": 2}),
        ))
        assert targets["P1"].fields["poison_delta"] == 2.0


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


class TestPayloadKinds:
    def test_a_combat_records_events_are_read(self):
        record = _record(
            CombatPayload(events=(
                Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                      params={"amount": 4, "combat": True}),
            )),
            kind=RecordKind.COMBAT,
        )
        assert derive_targets(record)["E1"].fields["damage_taken"] == 4.0

    def test_a_rewrite_reads_its_outgoing_event(self):
        """What the replacement produced, not what it received."""
        record = _record(
            RewritePayload(
                incoming=Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                               params={"amount": 5}),
                outgoing=Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                               params={"amount": 0}),
            ),
            kind=RecordKind.REWRITE, moment=None,
        )
        assert derive_targets(record)["E1"].fields["damage_taken"] == 0.0

    def test_a_fired_trigger_supervises_no_entity_with_the_event_that_caused_it(self):
        """The event is the trigger's input overlay; its own effects arrive as the
        resolution record of the ability it put on the stack."""
        record = _record(
            TriggerPayload(
                event=Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                            params={"delta": 3}),
                fired=True,
            ),
            kind=RecordKind.TRIGGER, moment=None,
        )
        assert derive_targets(record) == {}

    def test_an_unfired_trigger_produces_no_entity_targets(self):
        record = _record(
            TriggerPayload(
                event=Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                            params={"delta": 3}),
                fired=False,
            ),
            kind=RecordKind.TRIGGER, moment=None,
        )
        assert derive_targets(record) == {}

    def test_a_continuous_records_contributions_are_its_targets(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", pt_boost=(1, 1), keywords=("flying",)),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].affected
        assert targets["E1"].fields["power_delta"] == 1.0
        assert targets["E1"].fields["keywords_gained"][
            OVERLAY_KEYWORDS.index("flying")
        ] == 1.0

    def test_a_bare_type_token_is_gained_and_a_dashed_one_is_lost(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", types=("creature", "-land")),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["types_gained"][
            CARD_TYPES.index("creature")
        ] == 1.0
        assert targets["E1"].fields["types_lost"][
            CARD_TYPES.index("land")
        ] == 1.0

    def test_an_added_colour_is_gained_and_nothing_is_lost(self):
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", colors=("G",)),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["colors_gained"][COLORS.index("G")] == 1.0
        # Absent, not an empty set: the record says nothing about losses.
        assert "colors_lost" not in targets["E1"].fields

    def test_a_replacement_reports_what_it_sets_and_no_loss(self):
        """``=`` says the static overwrites the line.

        What it displaced is not in the record — the snapshot beside it is the
        board *after* the static applied — so the lost field stays unset rather
        than being guessed at.
        """
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(entity="E1", colors=("=", "C"), types=("=", "land")),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["colors_gained"][COLORS.index("C")] == 1.0
        assert targets["E1"].fields["types_gained"][CARD_TYPES.index("land")] == 1.0
        assert "colors_lost" not in targets["E1"].fields
        assert "types_lost" not in targets["E1"].fields

    def test_tokens_outside_the_vocabulary_are_dropped_not_misfiled(self):
        """Subtypes and whole-class removals are real, and have no field.

        The record keeps them because they describe what the static did; the
        model has a slot for each core type and none for "every creature type
        at once", so they must fall out here rather than land on a neighbour.
        """
        record = _record(
            ContinuousPayload(contributions=(
                Contribution(
                    entity="E1",
                    types=("elf", "legendary", "all-creature-types",
                           "-all-land-types", "creature"),
                ),
            )),
            kind=RecordKind.CONTINUOUS, moment=None,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["types_gained"] == [
            1.0 if name == "creature" else 0.0 for name in CARD_TYPES
        ]
        assert "types_lost" not in targets["E1"].fields

    def test_events_of_reads_each_payload_kind(self):
        assert len(events_of(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1",)),
        ))) == 1
        assert events_of(_record(
            ContinuousPayload(), kind=RecordKind.CONTINUOUS, moment=None,
        )) == ()


class TestLegalityBits:
    def test_a_legal_target_sets_its_bit(self):
        record = _record(
            PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(), can_play=True, affordable=True,
                          has_legal_target=True, legal_targets=("E1",)),
            )),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.DECISION,
        )
        assert derive_targets(record)["E1"].fields["target_legal"] == 1

    def test_legal_and_forbidden_attackers_both_get_a_bit(self):
        record = _record(
            PlayabilityAttackersPayload(
                legal_attackers=("E1",),
                forbidden=(ForbiddenEntity(entity="E2"),),
            ),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.ATTACKERS,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["attacker_legal"] == 1
        assert targets["E2"].fields["attacker_legal"] == 0

    def test_min_blockers_is_read_at_the_anchored_attacker(self):
        """Not at the blockers — the restriction belongs to the attacker."""
        record = _record(
            PlayabilityBlockersPayload(
                anchor_attacker="E9", legal_blockers=("E1",), min_blockers=2,
            ),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.BLOCKERS,
        )
        targets = derive_targets(record)
        assert targets["E9"].fields["min_blockers"] == 2
        assert "min_blockers" not in targets["E1"].fields

    def test_a_forbidden_blocker_gets_a_zero_bit(self):
        record = _record(
            PlayabilityBlockersPayload(
                anchor_attacker="E9", legal_blockers=("E1",),
                forbidden=(ForbiddenEntity(entity="E2"),),
            ),
            kind=RecordKind.PLAYABILITY, moment=None,
            subkind=PlayabilitySubkind.BLOCKERS,
        )
        targets = derive_targets(record)
        assert targets["E1"].fields["blocker_legal"] == 1
        assert targets["E2"].fields["blocker_legal"] == 0


class TestFieldNames:
    def test_every_produced_field_is_a_real_head_output(self):
        from effects.domain.effect_head_layout import FIELDS_BY_NAME

        record = _resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                  params={"amount": 1}),
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": "P1P1", "delta": 1}),
            Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                  params={"delta": 1}),
            Event(type=EventType.MANA_PRODUCED, subjects=("P0",),
                  params={"mana_by_color": {"R": 1}}),
        )
        for entry in derive_targets(record).values():
            for name in entry.fields:
                assert name in FIELDS_BY_NAME, name

    @pytest.mark.parametrize("counter", ["P1P1", "M1M1", "LOYALTY"])
    def test_every_tracked_counter_type_names_a_real_field(self, counter):
        from effects.domain.effect_head_layout import FIELDS_BY_NAME

        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": counter, "delta": 1}),
        ))
        assert set(targets["E1"].fields) <= set(FIELDS_BY_NAME)


class TestEverySimpleEventType:
    """Each event type the lookup tables handle, pinned field by field."""

    @pytest.mark.parametrize(("event_type", "params", "expected"), [
        (EventType.DAMAGE_DEALT, {"amount": 3}, {"damage_taken": 3.0}),
        (EventType.DAMAGE_DEALT, {}, {"damage_taken": 0.0}),
        (EventType.DAMAGE_HEALED, {"amount": 2}, {"damage_taken": -2.0}),
        (EventType.LIFE_CHANGE, {"delta": -4}, {"life_delta": -4.0}),
        (EventType.POISON_CHANGE, {"delta": 1}, {"poison_delta": 1.0}),
        (EventType.CARD_DRAWN, {}, {"cards_drawn": 1.0}),
        (EventType.CARD_DRAWN, {"count": 3}, {"cards_drawn": 3.0}),
        (EventType.CARD_DISCARDED, {}, {"cards_discarded": 1.0}),
        (EventType.CARD_MILLED, {"count": 2}, {"cards_milled": 2.0}),
        (EventType.TAPPED, {}, {"tap_state": 1}),
        (EventType.UNTAPPED, {}, {"tap_state": 0}),
        (EventType.CONTROL_CHANGE, {}, {"control_change": 1}),
        (EventType.ATTACHED, {}, {"attached_to_change": 1}),
        (EventType.UNATTACHED, {}, {"attached_to_change": 1}),
        (EventType.DESTROYED, {}, {"zone_outcome": ZONE_OUTCOMES.index("died")}),
        (EventType.FACE_CHANGE, {}, {"zone_outcome": ZONE_OUTCOMES.index("face_changed")}),
        (EventType.ZONE_CHANGE, {"to_zone": "Exile"},
         {"zone_outcome": ZONE_OUTCOMES.index("exiled")}),
        (EventType.ZONE_CHANGE, {"to_zone": "command"},
         {"zone_outcome": ZONE_OUTCOMES.index("to_command")}),
        (EventType.ZONE_CHANGE, {"to_zone": "nowhere"},
         {"zone_outcome": ZONE_OUTCOMES.index("stayed")}),
    ])
    def test_its_fields(self, event_type, params, expected):
        targets = derive_targets(_resolution(
            Event(type=event_type, subjects=("E1",), params=params),
        ))
        assert targets["E1"].fields == expected

    @pytest.mark.parametrize(("event_type", "flag"), [
        (EventType.CARD_LOOKED_AT, "scry"),
        (EventType.LIBRARY_REORDERED, "reorder"),
        (EventType.CARD_REVEALED, "reveal"),
    ])
    def test_its_library_flag(self, event_type, flag):
        targets = derive_targets(_resolution(
            Event(type=event_type, subjects=("P0",), params={}),
        ))
        expected = [0.0] * len(LIBRARY_EVENTS)
        expected[LIBRARY_EVENTS.index(flag)] = 1.0
        assert targets["P0"].fields == {"library_events": expected}

    def test_heals_and_hits_on_one_entity_net(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.DAMAGE_DEALT, subjects=("E1",), params={"amount": 5}),
            Event(type=EventType.DAMAGE_HEALED, subjects=("E1",), params={"amount": 2}),
        ))
        assert targets["E1"].fields == {"damage_taken": 3.0}


def test_deriving_targets_loads_no_torch():
    """Every survey and write worker derives targets record by record, and
    none of them needs torch to do it.

    In a fresh interpreter, because any test before this one may have
    imported torch into this one.
    """
    subprocess.run(
        [sys.executable, "-c",
         "import sys, effects.application.corpus_survey, effects.domain.effect_targets; "
         "assert 'torch' not in sys.modules, 'torch was imported'"],
        check=True, env=fresh_interpreter_env(),
    )


# ── gen-3's outputs (root spec § 11.2-11.4, T119, T121, T122, T123) ──────────

_RUN = "e8aa2e54-4112-425c-a7fa-96c7bb7340db"


@functools.cache
def _fixture() -> dict:
    """The gen-3 fixture's records, parsed, by record id."""
    from effects.infrastructure.shard_io import read_shard
    from tests.unit.effects.builders import GEN3_RECORDS

    return {record.record_id: record for record in read_shard(GEN3_RECORDS)}


def _players(*ids, library=30):
    from effects.domain.state_snapshot import PlayerState

    return tuple(
        PlayerState(id=pid, life=20, hand=5, library=library, graveyard=0) for pid in ids
    )


def _on_board(payload, *entities, kind=RecordKind.RESOLUTION, refs=None, **overrides):
    from effects.domain.state_snapshot import Refs
    from tests.unit.effects.builders import make_snapshot

    state = make_snapshot(
        players=_players("P0", "P1"), entities=tuple(entities), refs=refs or Refs(),
    )
    return _record(payload, kind=kind, state=state, **overrides)


def _entity(entity_id, **overrides):
    from tests.unit.effects.builders import make_entity

    return make_entity(entity_id, entity_id, **overrides)


def _zone(name):
    return ZONE_OUTCOMES.index(name)


def _spec(name):
    from effects.domain.effect_head_layout import FIELDS_BY_NAME

    return FIELDS_BY_NAME[name]


class TestTargetLegalityOnADecisionRecord:
    """Story 5 acceptance 3: ``target_legal`` on every card and player of the
    record, gate-free (root spec section 11.4)."""

    BERSERK = f"{_RUN}.5-mvyv9f2emq04.33592"

    def test_it_is_one_inside_the_legal_targets_and_zero_on_every_other_entity(self):
        record = _fixture()[self.BERSERK]
        legal = set(record.payload.candidates[0].legal_targets)
        assert legal == {"E149", "E19", "E110"}
        targets = derive_targets(record)
        refs = [e.id for e in record.state.entities] + [p.id for p in record.state.players]
        assert {"P0", "P1"} <= set(refs)
        for ref in refs:
            assert targets[ref].fields["target_legal"] == (1 if ref in legal else 0), ref

    def test_the_record_trains_no_gate(self):
        targets = derive_targets(_fixture()[self.BERSERK])
        assert targets.gated is False
        assert not any(entry.affected for entry in targets.values())

    def test_a_candidate_with_no_targeting_clause_leaves_it_unsupervised(self):
        land_play = next(
            record for record in _fixture().values()
            if record.subkind == PlayabilitySubkind.DECISION
            and record.payload.candidates[0].land_play
        )
        targets = derive_targets(land_play)
        assert not any("target_legal" in entry.fields for entry in targets.values())

    def test_a_clause_with_no_legal_target_is_zero_everywhere(self):
        record = _on_board(
            PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(), can_play=False, affordable=True, has_legal_target=False),
            )),
            _entity("E1"), kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION,
        )
        targets = derive_targets(record)
        assert {ref: entry.fields["target_legal"] for ref, entry in targets.items()} == {
            "E1": 0, "P0": 0, "P1": 0,
        }

    def test_an_effect_record_still_trains_the_gate(self):
        assert derive_targets(_resolution(
            Event(type=EventType.DESTROYED, subjects=("E1",)),
        )).gated is True


class TestCombatLegality:
    def test_an_anchored_attackers_record_labels_every_defender(self):
        record = _on_board(
            PlayabilityAttackersPayload(
                anchor_attacker="E1", legal_defenders=("P1", "E3"), must_attack=True,
                legal_attackers=("E1",),
            ),
            _entity("E1", types=("creature",)),
            _entity("E3", types=("planeswalker",), controller="P1"),
            _entity("E4", types=("battle",), controller="P1"),
            _entity("E5", types=("creature",), controller="P1"),
            kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
        )
        targets = derive_targets(record)
        legal = {ref: e.fields["attack_target_legal"] for ref, e in targets.items()
                 if "attack_target_legal" in e.fields}
        assert legal == {"P0": 0, "P1": 1, "E3": 1, "E4": 0}
        assert targets["E1"].fields["combat_requirement"] == 1
        assert targets.gated is False

    def test_a_declared_anchor_carries_the_tap_its_attack_made(self):
        """FR-037: tapped where ``tapped_by_attack`` is true, unchanged where false."""
        for tapped_by_attack, expected in ((True, 1), (False, 0)):
            record = _on_board(
                PlayabilityAttackersPayload(
                    anchor_attacker="E1", legal_defenders=("P1",),
                    tapped_by_attack=tapped_by_attack,
                ),
                _entity("E1", types=("creature",)),
                kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.ATTACKERS,
            )
            entry = derive_targets(record)["E1"]
            assert entry.fields["tap_state"] == expected
            assert entry.affected

    def test_a_fixture_attack_that_tapped_its_anchor_reads_tapped(self):
        record = _fixture()[f"{_RUN}.0-mvyv5jxepqmy.79"]
        assert record.payload.tapped_by_attack is True
        assert derive_targets(record)["E77"].fields["tap_state"] == 1

    def test_a_blockers_record_labels_the_block_requirement_on_every_listed_creature(self):
        record = _fixture()[f"{_RUN}.1-mvyw1pozylvp.13631"]
        payload = record.payload
        targets = derive_targets(record)
        must = set(payload.must_block)
        for ref in (*payload.legal_blockers, *(f.entity for f in payload.forbidden)):
            assert targets[ref].fields["combat_requirement"] == (1 if ref in must else 0)
        assert targets.gated is False


class TestGameLost:
    """Story 5 acceptance 6: the loser 1, the winner 0."""

    def test_a_win_is_the_other_players_loss(self):
        targets = derive_targets(_fixture()[f"{_RUN}.1-mvyw1pozylvp.984"])
        assert targets["P0"].fields["game_lost"] == 0
        assert targets["P1"].fields["game_lost"] == 1
        assert targets["P0"].affected and targets["P1"].affected

    def test_a_draw_is_both_players_losing(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.GAME_DRAWN),
        ))))
        assert targets["P0"].fields["game_lost"] == targets["P1"].fields["game_lost"] == 1

    def test_a_record_without_the_games_end_says_nothing_of_it(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.LIFE_CHANGE, subjects=("P1",), params={"delta": -3}),
        ))))
        assert targets["P1"].supervised(_spec("game_lost")) is None


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
        )), artifact, creature)
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
        )), angel, artifact, kind=RecordKind.CONTINUOUS, moment=None)
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
        )), bears, kind=RecordKind.CONTINUOUS, moment=None)
        entry = derive_targets(record)["E1"]
        assert entry.fields["base_power_delta"] == -2.0
        assert entry.supervised(_spec("base_toughness_delta")) is None

    def test_a_continuous_control_change_is_a_control_change(self):
        record = _on_board(ContinuousPayload(contributions=(
            Contribution(entity="E1", controller="P0"),
        )), _entity("E1", controller="P1"), kind=RecordKind.CONTINUOUS, moment=None)
        assert derive_targets(record)["E1"].fields["control_change"] == 1

    def test_a_pump_says_nothing_of_the_base(self):
        entry = derive_targets(_resolution(
            Event(type=EventType.PT_CHANGE, subjects=("E1",),
                  params={"power_delta": 3, "toughness_delta": 3}),
        ))["E1"]
        assert entry.supervised(_spec("base_power_delta")) is None


class TestZoneOutcomeExtensions:
    def test_a_permanent_spell_resolving_entered_with_the_face_it_was_cast_with(self):
        from effects.domain.effect_head_input import FACE_STATES

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

    def test_a_move_to_no_zone_ceased_to_exist(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",), params={"to_zone": "none"}),
        ))
        assert targets["E1"].fields["zone_outcome"] == _zone("ceased_to_exist")

    def test_phasing_in_and_out_are_distinct(self):
        from effects.domain.effect_head_input import DESIGNATIONS

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

    def test_a_search_then_shuffle_is_a_tutor(self):
        flags = derive_targets(_resolution(
            Event(type=EventType.ZONE_CHANGE, subjects=("E9",),
                  params={"from_zone": "library", "to_zone": "battlefield"}),
            Event(type=EventType.LIBRARY_SHUFFLED, subjects=("P0",)),
        ))["P0"].fields["library_events"]
        assert flags[LIBRARY_EVENTS.index("tutor")] == 1.0
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
        )), _entity("E1", zone="stack")))
        assert targets["E1"].fields["copied"] == 1

    def test_token_copies_place_nothing_and_leave_copied_unknown(self):
        targets = derive_targets(_on_board(ResolutionPayload(events=(
            Event(type=EventType.PERMANENT_COPIED,
                  params={"copy_source": "Mountain", "count": 1, "cause": "E1"}),
            Event(type=EventType.TAPPED, subjects=("E1",)),
        )), _entity("E1")))
        assert targets["E1"].supervised(_spec("copied")) is None

    def test_a_spell_copy_lands_on_the_targeted_stack_object(self):
        from effects.domain.state_snapshot import Refs

        targets = derive_targets(_on_board(
            ResolutionPayload(events=(
                Event(type=EventType.SPELL_COPIED, params={"count": 2, "cause": "E1"}),
            )),
            _entity("E1", zone="stack"), _entity("E2", zone="stack"),
            refs=Refs(targets=("E2",)),
        ))
        assert targets["E2"].fields["spell_copies"] == 2.0
        assert "E1" not in targets

    def test_a_retarget_lands_on_the_retargeted_spell(self):
        from effects.domain.state_snapshot import Refs

        targets = derive_targets(_on_board(
            ResolutionPayload(events=(
                Event(type=EventType.TARGETS_CHANGED, params={"targets": ["P0"]}),
            )),
            _entity("E2", zone="stack"), refs=Refs(targets=("E2",)),
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
        (EventType.PLAYER_LOST, "P0", {}, {"game_lost": 1}),
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
        )), _entity("E1", controller="P1")))
        assert targets["P1"].fields == {"ventured": 1}

    def test_a_goad_is_a_designation_gained(self):
        from effects.domain.effect_head_input import DESIGNATIONS

        fields = derive_targets(_resolution(
            Event(type=EventType.RESTRICTION_CHANGE, subjects=("E1",),
                  params={"restriction": "goad", "value": True}),
        ))["E1"].fields
        assert fields["designations_gained"][DESIGNATIONS.index("goaded")] == 1.0

    def test_a_type_change_reads_subtypes_and_supertypes_too(self):
        from effects.domain.effect_head_input import SUPERTYPES
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


class TestEveryFixtureTargetFitsItsField:
    """Every field's target has the shape its ``FieldSpec`` declares, over every
    curated record of the gen-3 fixture."""

    def test_widths_and_classes(self):
        from effects.domain.effect_head_layout import FIELD_SLICES
        from effects.domain.field_types import FieldType
        from tests.unit.effects.builders import curated_fixture_records

        seen = set()
        for record in curated_fixture_records():
            for entry in derive_targets(record).values():
                for name, value in entry.fields.items():
                    spec = _spec(name)
                    start, end = FIELD_SLICES[name]
                    assert end - start == spec.width
                    if spec.type is FieldType.MULTI_BINARY:
                        assert len(value) == spec.arity, name
                    elif spec.type is FieldType.CATEGORICAL:
                        assert 0 <= value < spec.arity, name
                    seen.add(name)
        assert {"target_legal", "attack_target_legal", "combat_requirement",
                "game_lost", "face_after"} <= seen
