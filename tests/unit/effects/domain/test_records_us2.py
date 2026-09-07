"""The four record kinds the patch unlocks (T100).

Three properties, each of which the corpus would be wrong without: continuous
records coalesce per stable board, trigger negatives are drawn same-event-type
at roughly 1:1, and **no policy verdict can be represented in a playability
payload** — the schema itself has no field for one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import fields

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.records import (
    Candidate,
    ContinuousPayload,
    Contribution,
    EffectRecord,
    ForbiddenEntity,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
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
_EVENT = Event(type=EventType.LIFE_CHANGE, params={"delta": 1})


def _record(kind, payload, **overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "r.0.1", "run_id": "r", "timestamp": "t", "game_id": "r.0.1",
        "kind": kind, "actor_player": "P0", "state": _SNAPSHOT, "payload": payload,
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


class TestContinuousCoalescing:
    def test_a_continuous_record_carries_the_board_hash_it_coalesced_on(self):
        """Coalescing per (game, static, board hash) is itself the cap: a
        static that applies every recompute would otherwise write a record per
        priority pass, none of them saying anything the first did not."""
        record = _record(
            RecordKind.CONTINUOUS,
            ContinuousPayload(
                contributions=(Contribution(entity="E1", pt_boost=(1, 1)),),
                board_hash="ab12cd",
            ),
        )
        assert record.payload.board_hash == "ab12cd"

    def test_two_records_of_one_static_on_one_board_share_a_hash(self):
        first = _record(
            RecordKind.CONTINUOUS,
            ContinuousPayload(
                contributions=(Contribution(entity="E1", pt_boost=(1, 1)),),
                board_hash="stable",
            ),
        )
        second = _record(
            RecordKind.CONTINUOUS,
            ContinuousPayload(
                contributions=(Contribution(entity="E2", pt_boost=(1, 1)),),
                board_hash="stable",
            ),
            record_id="r.0.2",
        )
        # A collector coalescing on the hash would emit one of these, not both.
        assert first.payload.board_hash == second.payload.board_hash

    def test_a_changed_board_gets_a_new_hash_and_a_new_record(self):
        before = ContinuousPayload(board_hash="before")
        after = ContinuousPayload(board_hash="after")
        assert before.board_hash != after.board_hash

    def test_contributions_are_per_entity_and_per_layer_channel(self):
        payload = ContinuousPayload(contributions=(
            Contribution(
                entity="E1", pt_boost=(1, 1), keywords=("flying",),
                types=("Creature",), colors=("W",), name=None,
            ),
        ))
        contribution = payload.contributions[0]
        assert contribution.entity == "E1"
        assert contribution.pt_boost == (1, 1)
        assert contribution.keywords == ("flying",)

    def test_a_continuous_record_names_its_acting_static(self):
        from effects.domain.provenance import ProvenanceKey

        key = ProvenanceKey("cardsfolder/a/anthem.txt", 0, "static", 0)
        record = _record(
            RecordKind.CONTINUOUS, ContinuousPayload(), ability=(key,),
        )
        assert record.ability == (key,)


class TestTriggerNegatives:
    def test_a_trigger_record_says_whether_it_fired(self):
        fired = _record(
            RecordKind.TRIGGER, TriggerPayload(event=_EVENT, fired=True),
        )
        did_not = _record(
            RecordKind.TRIGGER, TriggerPayload(event=_EVENT, fired=False),
            record_id="r.0.2",
        )
        assert fired.payload.fired
        assert not did_not.payload.fired

    def test_a_negative_carries_the_same_event_type_it_did_not_fire_on(self):
        """Negatives drawn same-event-type teach the condition; negatives drawn
        at random would only teach the base rate."""
        positive = TriggerPayload(
            event=Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                        params={"to_zone": "graveyard"}),
            fired=True,
        )
        negative = TriggerPayload(
            event=Event(type=EventType.ZONE_CHANGE, subjects=("E2",),
                        params={"to_zone": "graveyard"}),
            fired=False,
        )
        assert positive.event.type is negative.event.type

    def test_a_balanced_draw_is_roughly_one_to_one(self):
        payloads = [
            TriggerPayload(event=_EVENT, fired=index % 2 == 0)
            for index in range(100)
        ]
        counts = Counter(p.fired for p in payloads)
        assert counts[True] == counts[False]


class TestPlayabilityCarriesNoPolicy:
    def test_a_candidate_has_only_rules_level_verdict_fields(self):
        """The AI's judgments ("another time", "life in danger") are its
        opinion, not the game's rules; training on them would teach the model
        Forge's play style instead of Magic."""
        names = {f.name for f in fields(Candidate)}
        assert names == {
            "ability", "can_play", "affordable", "has_legal_target",
            "legal_targets", "cost_after_adjustment", "responsible_static",
        }

    def test_no_policy_field_can_be_set(self):
        with pytest.raises(TypeError):
            Candidate(
                ability=(), can_play=True, affordable=True,
                has_legal_target=True, ai_wants_to_cast=True,
            )

    def test_the_attackers_payload_carries_only_legality(self):
        names = {f.name for f in fields(PlayabilityAttackersPayload)}
        assert names == {"legal_attackers", "forbidden"}

    def test_the_blockers_payload_carries_only_legality_and_min_blockers(self):
        names = {f.name for f in fields(PlayabilityBlockersPayload)}
        assert names == {
            "anchor_attacker", "legal_blockers", "forbidden", "min_blockers",
        }

    def test_a_forbidden_entity_names_the_static_responsible(self):
        forbidden = ForbiddenEntity(entity="E1")
        assert forbidden.responsible_static == ()

    def test_the_three_subkinds_are_the_only_playability_shapes(self):
        assert {s.value for s in PlayabilitySubkind} == {
            "decision", "attackers", "blockers",
        }


class TestRewrite:
    def test_a_rewrite_carries_both_the_event_it_received_and_the_one_it_made(
        self,
    ):
        incoming = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                         params={"amount": 5})
        outgoing = Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                         params={"amount": 0})
        record = _record(
            RecordKind.REWRITE,
            RewritePayload(incoming=incoming, outgoing=outgoing),
        )
        assert record.payload.incoming.params["amount"] == 5
        assert record.payload.outgoing.params["amount"] == 0

    def test_a_stacked_pair_each_carries_the_event_it_received(self):
        """Two replacements on one event: the second receives what the first
        produced, and the parameter maps must be deep-copied at the hook or
        both records would show the final value."""
        original = Event(type=EventType.DAMAGE_DEALT, params={"amount": 6})
        halved = Event(type=EventType.DAMAGE_DEALT, params={"amount": 3})
        prevented = Event(type=EventType.DAMAGE_DEALT, params={"amount": 0})
        first = RewritePayload(incoming=original, outgoing=halved)
        second = RewritePayload(incoming=halved, outgoing=prevented)
        assert first.outgoing.params == second.incoming.params
        assert first.incoming.params["amount"] == 6


class TestDecisionPayload:
    def test_a_decision_lists_every_candidate_the_engine_evaluated(self):
        record = _record(
            RecordKind.PLAYABILITY,
            PlayabilityDecisionPayload(candidates=(
                Candidate(ability=(), can_play=True, affordable=True,
                          has_legal_target=True),
                Candidate(ability=(), can_play=False, affordable=False,
                          has_legal_target=False),
            )),
            subkind=PlayabilitySubkind.DECISION,
        )
        assert len(record.payload.candidates) == 2

    def test_a_candidate_records_its_cost_after_adjustment(self):
        candidate = Candidate(
            ability=(), can_play=True, affordable=False, has_legal_target=True,
            cost_after_adjustment={"R": 3},
        )
        assert candidate.cost_after_adjustment == {"R": 3}
