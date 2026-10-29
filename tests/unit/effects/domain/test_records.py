"""The record envelope's invariants (T022).

The schema is frozen before collection, so these are contract rather than
implementation detail: a corpus written against a violated invariant cannot be
repaired without recollecting it. Among them: no policy verdict can be
represented in a playability payload, since the schema has no field for one,
and a rewrite's result vocabulary tracks Forge's own enum both ways.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    REWRITE_RESULTS_RUNNING_AN_ABILITY,
    REWRITE_RESULTS_WITHOUT_ABILITY,
    ActivationPayload,
    Candidate,
    CombatPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    UnresolvedReason,
)
from tests.unit.effects.builders import make_record as build_record


class TestFlagSignatures:
    """The three signatures from the schema contract's flag table."""

    def test_an_ordinary_observation_carries_neither_flag(self, make_record):
        record = make_record()
        assert (record.interventional, record.fork) == (False, False)
        assert not record.is_probe

    def test_an_interventional_resolution_is_both(self, make_record):
        record = make_record(interventional=True, fork=True)
        assert record.kind is RecordKind.RESOLUTION
        assert (record.interventional, record.fork) == (True, True)
        assert not record.is_probe

    def test_a_damage_step_probe_is_a_fork_that_intervened_in_nothing(
        self, make_record,
    ):
        record = make_record(kind=RecordKind.COMBAT, fork=True)
        assert (record.interventional, record.fork) == (False, True)
        assert record.is_probe

    def test_an_intervention_must_be_a_resolution(self, make_record):
        with pytest.raises(ValueError, match="forced resolution"):
            make_record(kind=RecordKind.COMBAT, interventional=True, fork=True)

    def test_an_intervention_is_always_a_fork(self, make_record):
        with pytest.raises(ValueError, match="always a fork"):
            make_record(interventional=True, fork=False)

    def test_mirror_of_requires_the_fork_flag(self, make_record):
        with pytest.raises(ValueError, match="mirror_of"):
            make_record(kind=RecordKind.COMBAT, mirror_of="run.3.9")

    def test_variant_of_requires_the_synthetic_flag(self, make_record):
        with pytest.raises(ValueError, match="variant_of"):
            make_record(variant_of="Serra Angel")


class TestLinkId:
    """``link_id`` joins the two halves of a resolution pair, where there is one."""

    @pytest.mark.parametrize(
        "outcome",
        [
            ResolutionOutcome.FIZZLED,
            ResolutionOutcome.DECLINED,
            ResolutionOutcome.COUNTERED,
        ],
    )
    def test_a_partnerless_outcome_carries_no_link_id(self, make_record, outcome):
        with pytest.raises(ValueError, match="no linked effect half"):
            make_record(
                moment=Moment.ACTIVATION,
                payload=ActivationPayload(costs=Costs(), outcome=outcome),
                link_id="pair-1",
            )

    @pytest.mark.parametrize(
        "outcome",
        [ResolutionOutcome.RESOLVED, ResolutionOutcome.PARTIALLY_FIZZLED],
    )
    def test_a_resolving_outcome_may_carry_one(self, make_record, outcome):
        record = make_record(
            moment=Moment.ACTIVATION,
            payload=ActivationPayload(costs=Costs(), outcome=outcome),
            link_id="pair-1",
        )
        assert record.link_id == "pair-1"

    def test_an_interventional_effect_half_has_no_partner_to_link_to(
        self, make_record,
    ):
        with pytest.raises(ValueError, match="no activation partner"):
            make_record(interventional=True, fork=True, link_id="pair-1")

    def test_a_partnerless_outcome_without_a_link_id_is_fine(self, make_record):
        record = make_record(
            moment=Moment.ACTIVATION,
            payload=ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.COUNTERED,
            ),
        )
        assert record.link_id is None
        assert not record.payload.has_partner


class TestActingAbility:
    """``ability`` names the acting line, where one line acts."""

    def test_combat_records_name_no_acting_line(self, make_record, ability_key):
        with pytest.raises(ValueError, match="no single acting line"):
            make_record(kind=RecordKind.COMBAT, ability=(ability_key,))

    def test_playability_records_name_no_acting_line(
        self, make_record, ability_key, snapshot,
    ):
        with pytest.raises(ValueError, match="no single acting line"):
            EffectRecord(
                record_id="run.3.11", run_id="run", timestamp="t",
                game_id="run.3.2", kind=RecordKind.PLAYABILITY,
                subkind=PlayabilitySubkind.ATTACKERS,
                actor_player="P0", state=snapshot,
                payload=PlayabilityAttackersPayload(),
                ability=(ability_key,),
            )


class TestAbilityUnresolved:
    """Why no key was found, where a line was looked for and not reached.

    An empty ``ability`` alone conflates two different things — the Monarch has
    no printed line in any tree and never will, while a resolver that cannot
    reach one is a bug — and that ambiguity is what let a broken resolver
    survive a whole eight-hour collection run without a single failure.
    """

    def test_an_unresolved_record_names_why(self, make_record):
        record = make_record(ability=(), ability_unresolved="engine_effect")
        assert record.ability_unresolved == "engine_effect"

    def test_a_record_cannot_both_name_a_line_and_fail_to_find_one(
        self, make_record,
    ):
        with pytest.raises(ValueError, match="could not find one"):
            make_record(ability_unresolved="unindexable")

    def test_a_kind_that_looks_for_no_line_reports_no_reason(self, make_record):
        with pytest.raises(ValueError, match="look for no acting line"):
            make_record(
                kind=RecordKind.COMBAT, ability=None,
                ability_unresolved="engine_effect",
            )

    def test_the_reason_comes_from_the_closed_vocabulary(self, make_record):
        with pytest.raises(ValueError, match="closed vocabulary"):
            make_record(ability=(), ability_unresolved="dunno")


class TestDiscriminators:
    def test_moment_belongs_to_the_resolution_kind(self, make_record):
        with pytest.raises(ValueError, match="moment is the resolution kind"):
            make_record(
                kind=RecordKind.COMBAT,
                moment=Moment.RESOLUTION,
                payload=CombatPayload(),
            )

    def test_a_resolution_without_a_moment_is_rejected(self, snapshot):
        with pytest.raises(ValueError, match="moment is the resolution kind"):
            EffectRecord(
                record_id="r.0.1", run_id="r", timestamp="t", game_id="r.0.1",
                kind=RecordKind.RESOLUTION, actor_player="P0", state=snapshot,
                payload=ResolutionPayload(),
            )

    def test_subkind_belongs_to_the_playability_kind(self, make_record):
        with pytest.raises(ValueError, match="subkind is the playability kind"):
            make_record(
                kind=RecordKind.COMBAT,
                subkind=PlayabilitySubkind.DECISION,
                payload=CombatPayload(),
            )

    def test_the_payload_type_must_match_the_kind(self, make_record):
        with pytest.raises(ValueError, match="takes a ResolutionPayload"):
            make_record(moment=Moment.RESOLUTION, payload=CombatPayload())

    def test_the_payload_type_must_match_the_moment(self, make_record):
        with pytest.raises(ValueError, match="takes an? ActivationPayload"):
            make_record(moment=Moment.ACTIVATION, payload=ResolutionPayload())


#: The Java side of the contract. Read rather than mirrored in a constant,
#: because a mirrored list is a third copy to drift.
_JAVA_UNRESOLVED_REASON = (
    Path(__file__).resolve().parents[4]
    / "forge-connector" / "src" / "main" / "java" / "com" / "pricepredictor"
    / "connector" / "effects" / "UnresolvedReason.java"
)
_PYTHON_REASONS = {reason.value for reason in UnresolvedReason}


class TestTheUnresolvedVocabularyIsOneSet:
    """Every reason the collector writes is a reason the reader accepts.

    It was not. The collector learned to say ``granted_keyword`` and
    ``granted_trait`` -- a keyword or trait granted by something that keeps no
    back-reference to what granted it -- while ``UnresolvedReason`` still
    listed four values, so ``record_from_dict`` rejected every record carrying
    one. 187 of 391,168 records in a smoke corpus, spread over 17 of 32 shards,
    and the reader raises rather than skips: the first such record ends the
    read, which would have ended an eight-hour corpus at the first consumer.

    A closed vocabulary is the right shape -- an unrecognised reason should be
    loud, since the field exists to separate "no printed line exists" from "the
    resolver failed". What was missing is anything holding the two ends of it
    together, which is what this test is.
    """

    def _java_reasons(self) -> set[str]:
        """The wire name of every ``UnresolvedReason`` member the collector has."""
        source = _JAVA_UNRESOLVED_REASON.read_text(encoding="utf-8")
        return set(re.findall(
            r'^\s*[A-Z_]+\("([a-z_]+)"\)[,;]', source, re.MULTILINE,
        ))

    def test_the_collector_writes_no_reason_the_reader_rejects(self):
        java = self._java_reasons()
        assert java, "no UnresolvedReason members found; did the file move?"
        assert java <= _PYTHON_REASONS, (
            "the collector writes reasons the reader would reject: "
            f"{sorted(java - _PYTHON_REASONS)}"
        )

    def test_the_reader_accepts_no_reason_the_collector_never_writes(self):
        """The other direction, so a dead value cannot sit in the set unnoticed.

        A reason nothing writes reads as evidence that a resolver path exists
        when it does not.
        """
        assert _PYTHON_REASONS <= self._java_reasons(), (
            "the reader accepts reasons nothing writes: "
            f"{sorted(_PYTHON_REASONS - self._java_reasons())}"
        )


class TestGen2Flags:
    """``random_seat`` on every kind; ``what_if`` on legality records only (FR-031)."""

    def test_random_seat_defaults_to_on_policy(self, make_record):
        assert make_record().random_seat is False

    @pytest.mark.parametrize("subkind", ["attackers", "blockers"])
    def test_a_legality_record_may_carry_what_if(self, make_record, subkind):
        record = make_record(**_legality(subkind), what_if=False)
        assert record.what_if is False

    def test_a_decision_record_never_carries_what_if(self, make_record):
        with pytest.raises(ValueError, match="what_if"):
            make_record(**_legality("decision"), what_if=True)

    def test_a_resolution_never_carries_what_if(self, make_record):
        with pytest.raises(ValueError, match="what_if"):
            make_record(what_if=True)


def _legality(subkind: str) -> dict:
    from effects.domain.records import (
        PlayabilityAttackersPayload,
        PlayabilityBlockersPayload,
        PlayabilityDecisionPayload,
        PlayabilitySubkind,
    )

    payloads = {
        "attackers": PlayabilityAttackersPayload(legal_attackers=("E1",)),
        "blockers": PlayabilityBlockersPayload(anchor_attacker="E1"),
        "decision": PlayabilityDecisionPayload(candidates=()),
    }
    return dict(
        kind=RecordKind.PLAYABILITY, ability=None,
        subkind=PlayabilitySubkind(subkind), payload=payloads[subkind],
    )


#: The line a replacement runs *instead*, keyed the way the record's own
#: ``ability`` field is.
_REPLACED_BY = ProvenanceKey(
    script_file="cardsfolder/l/leyline_of_the_void.txt",
    face=0, trait_kind="replacement", index_within_kind=0,
)


def _record(kind, payload, **overrides) -> EffectRecord:
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}
    defaults.update(overrides)
    return build_record(kind, payload=payload, **defaults)


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

    def test_the_attackers_payload_carries_only_legality(self):
        names = {f.name for f in fields(PlayabilityAttackersPayload)}
        assert names == {"legal_attackers", "forbidden"}

    def test_the_blockers_payload_carries_only_legality_and_min_blockers(self):
        names = {f.name for f in fields(PlayabilityBlockersPayload)}
        assert names == {
            "anchor_attacker", "legal_blockers", "forbidden", "min_blockers",
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
