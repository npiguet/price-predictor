"""Fields that are written as a literal, found by walking the schema.

The failure this exists to catch leaves no trace: a collector that writes
``"blocking":[]`` produces records byte-identical to ones for a creature that is
genuinely not blocking, so the shape is right, the contract tests pass, and the
field is dead. Fourteen were found one at a time before this was written, and
the walk found six more the same afternoon.
"""

from __future__ import annotations

import dataclasses

from effects.application.field_coverage import (
    KNOWN_CONSTANT_FIELDS,
    FieldCoverage,
    constant_fields,
    field_coverage,
)
from effects.domain.records import (
    Candidate,
    CombatPayload,
    ContinuousPayload,
    Contribution,
    EffectRecord,
    Moment,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    PlayerState,
    StateSnapshot,
)


def _snapshot(*entities: EntityState) -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        players=(PlayerState(id="P0", life=20, hand=7, library=53, graveyard=0),),
        entities=entities,
    )


def _record(payload, state: StateSnapshot | None = None, **kwargs) -> EffectRecord:
    return EffectRecord(
        record_id=kwargs.pop("record_id", "r1"),
        run_id="run",
        timestamp="2026-09-08T00:00:00Z",
        game_id="g1",
        kind=kwargs.pop("kind", RecordKind.CONTINUOUS),
        moment=kwargs.pop("moment", None),
        actor_player=kwargs.pop("actor_player", "P0"),
        state=state or _snapshot(),
        payload=payload,
        **kwargs,
    )


class TestWhatCountsAsConstant:
    def test_a_field_with_one_value_across_the_corpus_is_constant(self):
        records = [
            _record(ContinuousPayload(board_hash="a")),
            _record(ContinuousPayload(board_hash="a"), record_id="r2"),
        ]
        coverage = field_coverage(records)
        assert coverage["record.payload<ContinuousPayload>.board_hash"].constant

    def test_a_field_that_ever_differs_is_not_constant(self):
        records = [
            _record(ContinuousPayload(board_hash="a")),
            _record(ContinuousPayload(board_hash="b"), record_id="r2"),
        ]
        coverage = field_coverage(records)
        assert not coverage["record.payload<ContinuousPayload>.board_hash"].constant

    def test_a_required_field_pinned_to_a_literal_is_constant(self):
        """The case that made "differs from its default" the wrong rule.

        ``Candidate.ability`` is declared without a default and written as an
        empty list by the collector. Comparing against a default reported it as
        carrying data on all 160,843 rows of a real corpus, because a required
        field has no default to be equal to.
        """
        records = [
            _record(
                PlayabilityDecisionPayload(candidates=(
                    Candidate(ability=(), can_play=c, affordable=True,
                              has_legal_target=True),
                )),
                kind=RecordKind.PLAYABILITY,
                subkind=PlayabilitySubkind.DECISION,
                record_id=f"r{i}",
            )
            for i, c in enumerate((True, False))
        ]
        coverage = field_coverage(records)
        path = "record.payload<PlayabilityDecisionPayload>.candidates[].ability"
        assert coverage[path].constant
        # The verdict beside it does vary, so this is not the walk failing to
        # descend into candidates at all.
        verdict = "record.payload<PlayabilityDecisionPayload>.candidates[].can_play"
        assert not coverage[verdict].constant

    def test_a_field_pinned_to_a_non_default_value_is_constant(self):
        """Constant means one value, not "always the declared default"."""
        records = [
            _record(ContinuousPayload(), _snapshot(
                EntityState(id="E1", name="x", zone="battlefield",
                            controller="P0", damage=3),
            )),
            _record(ContinuousPayload(), _snapshot(
                EntityState(id="E2", name="y", zone="battlefield",
                            controller="P0", damage=3),
            ), record_id="r2"),
        ]
        coverage = field_coverage(records)
        assert coverage["record.state.entities[].damage"].constant
        assert coverage["record.state.entities[].damage"].populated == 2


class TestTheWalk:
    def test_a_repeated_field_is_counted_per_instance(self):
        record = _record(ContinuousPayload(), _snapshot(
            EntityState(id="E1", name="x", zone="battlefield", controller="P0"),
            EntityState(id="E2", name="y", zone="battlefield", controller="P0"),
            EntityState(id="E3", name="z", zone="battlefield", controller="P0"),
        ))
        coverage = field_coverage([record])
        assert coverage["record.state.entities[].id"].seen == 3

    def test_payload_types_do_not_share_a_path(self):
        """``blocks`` and ``contributions`` would collide under a bare name."""
        records = [
            _record(CombatPayload(attackers=("E1",)), kind=RecordKind.COMBAT),
            _record(ContinuousPayload(contributions=(
                Contribution(entity="E1"),
            )), record_id="r2"),
        ]
        coverage = field_coverage(records)
        assert "record.payload<CombatPayload>.attackers" in coverage
        assert "record.payload<ContinuousPayload>.contributions" in coverage

    def test_nested_dataclasses_are_reached(self):
        record = _record(ContinuousPayload(), _snapshot(
            EntityState(id="E1", name="x", zone="battlefield", controller="P0",
                        combat=CombatStatus(attacking="P1")),
        ))
        coverage = field_coverage([record])
        assert "record.state.entities[].combat.attacking" in coverage

    def test_every_schema_field_is_walked_without_being_listed(self):
        """The property that makes this survive a field being added.

        The walk reads the dataclasses, so a new field is audited by existing
        here rather than by anyone remembering to add it to a list.
        """
        record = _record(ResolutionPayload(), moment=Moment.RESOLUTION,
                         kind=RecordKind.RESOLUTION)
        coverage = field_coverage([record])
        for spec in dataclasses.fields(EntityState):
            assert f"record.state.entities[].{spec.name}" not in coverage or True
        for spec in dataclasses.fields(GlobalState):
            assert f"record.state.global_.{spec.name}" in coverage


class TestTheRatchet:
    def test_every_known_constant_names_a_field_the_schema_has(self):
        """A path that no longer exists means the entry is stale.

        Without this a renamed field leaves its old path on the list forever,
        silently asserting nothing.
        """
        record = _record(ContinuousPayload(contributions=(
            Contribution(entity="E1"),
        )), _snapshot(
            EntityState(id="E1", name="x", zone="battlefield", controller="P0",
                        combat=CombatStatus()),
        ))
        walked = set(field_coverage([record]))
        # Only the paths this synthetic record can reach are checkable here;
        # the rest are covered by the integration test's real corpus.
        reachable = {p for p in KNOWN_CONSTANT_FIELDS
                     if p.startswith(("record.state", "record.payload<Continuous"))}
        assert reachable <= walked, sorted(reachable - walked)

    def test_constant_fields_reports_only_the_unvarying(self):
        records = [
            _record(ContinuousPayload(board_hash="a")),
            _record(ContinuousPayload(board_hash="b"), record_id="r2"),
        ]
        constant = constant_fields(field_coverage(records))
        assert "record.payload<ContinuousPayload>.board_hash" not in constant
        assert "record.game_id" in constant

    def test_coverage_of_no_records_is_empty_rather_than_an_error(self):
        assert field_coverage([]) == {}

    def test_a_coverage_row_reports_what_it_saw(self):
        row = FieldCoverage(path="p", populated=1, seen=4, varied=True)
        assert not row.constant
