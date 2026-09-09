"""The four record kinds the patch unlocks (T100).

Three properties, each of which the corpus would be wrong without: continuous
records coalesce per stable board, trigger negatives are drawn same-event-type
at roughly 1:1, and **no policy verdict can be represented in a playability
payload** — the schema itself has no field for one.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import fields
from pathlib import Path

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    REWRITE_RESULTS_RUNNING_AN_ABILITY,
    REWRITE_RESULTS_WITHOUT_ABILITY,
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
    RewriteResult,
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
#: The line a replacement runs *instead*, keyed the way the record's own
#: ``ability`` field is.
_REPLACED_BY = ProvenanceKey(
    script_file="cardsfolder/l/leyline_of_the_void.txt",
    face=0, trait_kind="replacement", index_within_kind=0,
)


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


#: Forge's own enum, in the sibling checkout the converter and the collector
#: both already assume (``collect_variants`` defaults to ``../forge``).
_REPLACEMENT_RESULT_JAVA = (
    Path(__file__).resolve().parents[4].parent
    / "forge" / "forge-game" / "src" / "main" / "java" / "forge" / "game"
    / "replacement" / "ReplacementResult.java"
)


def _java_replacement_results() -> tuple[str, ...]:
    """The enum members declared in ``ReplacementResult.java``.

    Parsed rather than hardcoded, so this test tracks the file instead of
    tracking a copy of it made on the day it was written.
    """
    source = _REPLACEMENT_RESULT_JAVA.read_text(encoding="utf-8")
    body = source.split("enum ReplacementResult", 1)[1].split("{", 1)[1]
    body = body.split("}", 1)[0]
    body = re.sub(r"//[^\n]*", "", body)
    return tuple(
        member.strip() for member in body.split(",") if member.strip()
    )


def _snake(name: str) -> str:
    """``NotReplaced`` -> ``not_replaced``, the contract's spelling rule."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class TestRewriteResultTracksForge:
    """The closed vocabulary, pinned to the enum it mirrors -- both ways.

    A closed vocabulary with nothing holding its two ends together is the
    defect that cost a corpus this week: the collector learned two new reason
    values, the reader raised on every record carrying one, and 187 of 391,168
    records were lost to a file nobody thought to reread. ``result`` is closed
    for the same reasons ``ResolutionOutcome`` is, so it needs the tie the
    reasons did not have.

    Skipped rather than failed when the Forge checkout is absent: this repo is
    cloned on machines that do not carry its sibling, and a test that fails
    there teaches people to ignore it.
    """

    @pytest.fixture(scope="class")
    def java_members(self) -> tuple[str, ...]:
        if not _REPLACEMENT_RESULT_JAVA.exists():
            pytest.skip(f"no Forge checkout at {_REPLACEMENT_RESULT_JAVA}")
        return _java_replacement_results()

    def test_every_forge_member_has_a_python_value(self, java_members):
        """Forge grows a sixth outcome -> the reader must learn it first."""
        assert java_members, "the enum body parsed as empty"
        missing = [
            name for name in java_members
            if _snake(name) not in {r.value for r in RewriteResult}
        ]
        assert not missing, (
            f"ReplacementResult declares {missing}, which RewriteResult has no "
            "value for; a record carrying one would raise on read"
        )

    def test_every_python_value_has_a_forge_member(self, java_members):
        """The other direction: a value Forge cannot produce is dead weight,
        and a stale one hides that the enum was renamed rather than extended."""
        expected = {_snake(name) for name in java_members}
        assert {r.value for r in RewriteResult} == expected

    def test_the_two_partitions_only_hold_real_results(self, java_members):
        for result in REWRITE_RESULTS_WITHOUT_ABILITY:
            assert result in RewriteResult
        for result in REWRITE_RESULTS_RUNNING_AN_ABILITY:
            assert result in RewriteResult
        assert not (
            REWRITE_RESULTS_WITHOUT_ABILITY & REWRITE_RESULTS_RUNNING_AN_ABILITY
        )
        # `not_replaced` is in neither: it runs no ability but its prevention
        # branch still writes PreventedAmount into the map on the way out.
        assert RewriteResult.NOT_REPLACED not in (
            REWRITE_RESULTS_WITHOUT_ABILITY | REWRITE_RESULTS_RUNNING_AN_ABILITY
        )


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
            RewritePayload(
                incoming=incoming, outgoing=outgoing,
                result=RewriteResult.UPDATED,
            ),
        )
        assert record.payload.incoming.params["amount"] == 5
        assert record.payload.outgoing.params["amount"] == 0
        assert record.payload.rewritten

    def test_a_stacked_pair_each_carries_the_event_it_received(self):
        """Two replacements on one event: the second receives what the first
        produced, and the parameter maps must be deep-copied at the hook or
        both records would show the final value."""
        original = Event(type=EventType.DAMAGE_DEALT, params={"amount": 6})
        halved = Event(type=EventType.DAMAGE_DEALT, params={"amount": 3})
        prevented = Event(type=EventType.DAMAGE_DEALT, params={"amount": 0})
        first = RewritePayload(incoming=original, outgoing=halved,
                               result=RewriteResult.UPDATED)
        second = RewritePayload(incoming=halved, outgoing=prevented,
                                result=RewriteResult.UPDATED)
        assert first.outgoing.params == second.incoming.params
        assert first.incoming.params["amount"] == 6

    def test_a_substitution_rewrites_nothing_and_names_what_ran_instead(self):
        """An "exile it instead" replacement edits no parameter.

        Forge carries that out by running a different ability, so the event's
        map comes back untouched. ``outgoing`` is null and ``replaced_by``
        holds the line that ran -- which is the whole of what the record has to
        say, and none of it was representable before.
        """
        payload = RewritePayload(
            incoming=Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                           params={"to_zone": "graveyard"}),
            result=RewriteResult.REPLACED,
            replaced_by=(_REPLACED_BY,),
        )
        assert payload.outgoing is None
        assert not payload.rewritten
        assert payload.replaced_by == (_REPLACED_BY,)

    def test_a_declined_replacement_is_still_a_record(self):
        """The channel's negatives, kept the way the trigger channel keeps
        its non-fired evaluations: what teaches *when* a replacement applies."""
        payload = RewritePayload(
            incoming=_EVENT, result=RewriteResult.NOT_REPLACED,
        )
        assert payload.result is RewriteResult.NOT_REPLACED
        assert payload.outgoing is None
        assert payload.replaced_by == ()

    def test_a_pre_contract_record_reports_no_result_rather_than_a_negative(
        self,
    ):
        """``None`` is a third state. Reading it as ``not_replaced`` would
        reinterpret every record collected before the hook passed one."""
        payload = RewritePayload(incoming=_EVENT, outgoing=_EVENT)
        assert payload.result is None
        assert payload.result is not RewriteResult.NOT_REPLACED


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
