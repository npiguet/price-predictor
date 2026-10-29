"""The four record kinds the patch unlocks (T100).

Three properties, each of which the corpus would be wrong without: continuous
records coalesce per stable board, trigger negatives are drawn same-event-type
at roughly 1:1, and **no policy verdict can be represented in a playability
payload** — the schema itself has no field for one.
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
    Candidate,
    EffectRecord,
    PlayabilityAttackersPayload,
    PlayabilityBlockersPayload,
    RecordKind,
    RewritePayload,
    RewriteResult,
)
from tests.unit.effects.builders import make_record

#: The line a replacement runs *instead*, keyed the way the record's own
#: ``ability`` field is.
_REPLACED_BY = ProvenanceKey(
    script_file="cardsfolder/l/leyline_of_the_void.txt",
    face=0, trait_kind="replacement", index_within_kind=0,
)


def _record(kind, payload, **overrides) -> EffectRecord:
    defaults: dict = {"record_id": "r.0.1", "run_id": "r", "game_id": "r.0.1"}
    defaults.update(overrides)
    return make_record(kind, payload=payload, **defaults)


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
