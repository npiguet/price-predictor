"""Events to per-entity head targets.

The two rules that shape everything else: an entity is affected when any event
names it (not when some field changed — a spell that targets a creature and
deals 0 damage still affected it), and a field with no event is *absent* rather
than zero, so the loss skips it instead of learning that every unmentioned
counter went to zero.
"""

from __future__ import annotations

import pytest

from effects.domain.effect_head_input import CARD_TYPES, OVERLAY_KEYWORDS
from effects.domain.effect_model import LIBRARY_EVENTS, ZONE_OUTCOMES
from effects.domain.effect_targets import derive_targets, events_of
from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    Candidate,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    EffectRecord,
    ForbiddenEntity,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot

_SNAPSHOT = StateSnapshot(
    global_=GlobalState(
        turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
    ),
    players=(),
    entities=(),
)


def _record(payload, kind=RecordKind.RESOLUTION, **overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "r.0.1", "run_id": "r", "timestamp": "t", "game_id": "r.0.1",
        "kind": kind, "actor_player": "P0", "state": _SNAPSHOT, "payload": payload,
    }
    if kind is RecordKind.RESOLUTION:
        defaults["moment"] = Moment.RESOLUTION
    defaults.update(overrides)
    return EffectRecord(**defaults)


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
                  params={"counter_type": "PAGE", "delta": 1}),
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

    def test_an_untracked_keyword_sets_no_bit(self):
        targets = derive_targets(_resolution(
            Event(type=EventType.KEYWORD_CHANGE, subjects=("E1",),
                  params={"keywords": ["bushido"]}),
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

    def test_a_fired_trigger_reads_its_event(self):
        record = _record(
            TriggerPayload(
                event=Event(type=EventType.LIFE_CHANGE, subjects=("P0",),
                            params={"delta": 3}),
                fired=True,
            ),
            kind=RecordKind.TRIGGER, moment=None,
        )
        assert derive_targets(record)["P0"].fields["life_delta"] == 3.0

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
        from effects.domain.effect_model import FIELDS_BY_NAME

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
        from effects.domain.effect_model import FIELDS_BY_NAME

        targets = derive_targets(_resolution(
            Event(type=EventType.COUNTER_CHANGE, subjects=("E1",),
                  params={"counter_type": counter, "delta": 1}),
        ))
        assert set(targets["E1"].fields) <= set(FIELDS_BY_NAME)
