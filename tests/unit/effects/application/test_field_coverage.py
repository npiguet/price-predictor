"""Fields that are written as a literal, found by walking the schema.

The failure this exists to catch leaves no trace: a collector that writes
``"blocking":[]`` produces records byte-identical to ones for a creature that is
genuinely not blocking, so the shape is right, the contract tests pass, and the
field is dead. Fourteen were found one at a time before this was written, and
the walk found six more the same afternoon.
"""

from __future__ import annotations

import dataclasses

import pytest

from effects.application.field_coverage import (
    KNOWN_CONSTANT_FIELDS,
    FieldCoverage,
    event_param_lines,
    field_coverage,
    gen3_report_lines,
    report_lines,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.gen3_members import FIXED_PATHS, WALK_PATHS
from effects.domain.provenance import ProvenanceKey
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
    RewritePayload,
    RewriteResult,
)
from effects.domain.state_snapshot import (
    CombatStatus,
    EntityState,
    GlobalState,
    PlayerState,
    StateSnapshot,
)
from tests.unit.effects.builders import make_record, make_snapshot


def _snapshot(*entities: EntityState) -> StateSnapshot:
    return make_snapshot(
        players=(PlayerState(id="P0", life=20, hand=7, library=53, graveyard=0),),
        entities=entities,
    )


def _record(payload, state: StateSnapshot | None = None, **kwargs) -> EffectRecord:
    return make_record(
        kwargs.pop("kind", RecordKind.CONTINUOUS),
        record_id=kwargs.pop("record_id", "r1"),
        timestamp="2026-09-08T00:00:00Z",
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

    def test_each_event_parameter_has_a_cell_of_its_own(self):
        """Every parameter collapsed into one ``params`` cell, so a
        parameter the collector never writes could not show."""
        events = tuple(
            Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                  params={"from_zone": "battlefield", "to_zone": zone, "cause": "E2"})
            for zone in ("graveyard", "exile"))
        record = _record(ResolutionPayload(events=events), kind=RecordKind.RESOLUTION,
                         moment=Moment.RESOLUTION)
        coverage = field_coverage([record])
        cell = "record.payload<ResolutionPayload>.events[].params<zone_change>"
        assert coverage[f"{cell}.to_zone"].varied
        assert coverage[f"{cell}.from_zone"].constant
        assert coverage[f"{cell}.library_position"].populated == 0
        lines = event_param_lines(coverage)
        assert any("[absent]" in line and "zone_change.library_position" in line
                   for line in lines)
        assert any("[varies]" in line and "zone_change.to_zone" in line for line in lines)

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
        entity = EntityState(id="E1", name="x", zone="battlefield", controller="P0",
                             combat=CombatStatus())
        record = _record(ResolutionPayload(), _snapshot(entity), moment=Moment.RESOLUTION,
                         kind=RecordKind.RESOLUTION)
        coverage = field_coverage([record])

        def walked(prefix: str) -> bool:
            return any(path == prefix or path.startswith((f"{prefix}.", f"{prefix}[]"))
                       for path in coverage)

        for owner, prefix in ((EntityState, "record.state.entities[]"),
                              (GlobalState, "record.state.global_"),
                              (ResolutionPayload, "record.payload<ResolutionPayload>")):
            # A field outside equality is the codec's bookkeeping, not data.
            missing = [spec.name for spec in dataclasses.fields(owner)
                       if spec.compare and not walked(f"{prefix}.{spec.name}")]
            assert not missing, (owner.__name__, missing)


class TestTheRewriteChannelIsNotExcused:
    """The channel that was empty because its payload modelled the wrong thing.

    ``rewrite`` held 34 records in 1.88M, and it would have been easy to read
    that as format scarcity and put its fields on the excused list. It was not:
    Forge substitutes by running a different ability, so ``incoming`` and
    ``outgoing`` came back byte-identical and 87% of the channel was dropped.
    ``result`` and ``replaced_by`` are what those records were trying to say, so
    a corpus where they read constant is evidence about the *hook*, and the one
    thing that must not happen is for the list to absorb them.
    """

    def _rewrite(self, record_id: str, **payload_kwargs) -> EffectRecord:
        return _record(
            RewritePayload(
                incoming=Event(type=EventType.DAMAGE_DEALT,
                               params={"amount": 3}),
                **payload_kwargs,
            ),
            kind=RecordKind.REWRITE,
            record_id=record_id,
        )

    def test_neither_new_field_is_on_the_excused_list(self):
        for name in ("result", "replaced_by"):
            path = f"record.payload<RewritePayload>.{name}"
            assert path not in KNOWN_CONSTANT_FIELDS, (
                f"{path} was constant because the hook passed nothing, which is "
                "a dead channel rather than an unexercised one"
            )

    def test_a_hook_that_passes_no_result_reads_as_a_constant_field(self):
        """The shape of the pre-contract corpus, and what it should report."""
        records = [self._rewrite("r1"), self._rewrite("r2")]
        coverage = field_coverage(records)
        assert coverage["record.payload<RewritePayload>.result"].constant
        assert coverage["record.payload<RewritePayload>.replaced_by"].constant

    def test_a_wired_hook_clears_them(self):
        records = [
            self._rewrite("r1", result=RewriteResult.NOT_REPLACED),
            self._rewrite("r2", result=RewriteResult.REPLACED, replaced_by=(
                ProvenanceKey(
                    script_file="cardsfolder/l/leyline_of_the_void.txt",
                    face=0, trait_kind="replacement", index_within_kind=0,
                ),
            )),
        ]
        coverage = field_coverage(records)
        assert not coverage["record.payload<RewritePayload>.result"].constant
        assert not coverage["record.payload<RewritePayload>.replaced_by"].constant

    def test_a_null_outgoing_counts_as_unpopulated_rather_than_as_data(self):
        """``outgoing`` gained a default, so "not rewritten" no longer reads as
        a populated field the way a required one always did."""
        coverage = field_coverage([self._rewrite("r1")])
        assert coverage["record.payload<RewritePayload>.outgoing"].populated == 0


class TestTheRatchet:
    def test_every_known_constant_names_a_field_the_collector_writes(self):
        """A path that no longer exists means the entry is stale: without this
        a renamed field leaves its old path on the list forever, silently
        asserting nothing. Checked against the walk over every Java golden
        record, the collector's own shapes (the list is empty today)."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import java_golden_records

        walked = set(field_coverage([record_from_dict(raw) for raw in java_golden_records()]))
        assert set(KNOWN_CONSTANT_FIELDS) <= walked, sorted(set(KNOWN_CONSTANT_FIELDS) - walked)

    def test_coverage_of_no_records_is_empty_rather_than_an_error(self):
        assert field_coverage([]) == {}

class TestReportLines:
    """The rows ``field-coverage`` prints, tagged by what the list says."""

    def _coverage(self, *rows: FieldCoverage) -> dict[str, FieldCoverage]:
        return {row.path: row for row in rows}

    #: A listed field for the tags' sake: the real list is empty.
    _KNOWN = "record.state.global_.emblems"

    @pytest.fixture(autouse=True)
    def _listed(self, monkeypatch):
        from effects.application import field_coverage as module

        monkeypatch.setattr(module, "KNOWN_CONSTANT_FIELDS", frozenset({self._KNOWN}))

    def test_constant_fields_are_tagged_known_or_new(self):
        known = self._KNOWN
        lines = report_lines(self._coverage(
            FieldCoverage(path=known, populated=0, seen=3, varied=False),
            FieldCoverage(path="record.x", populated=0, seen=3, varied=False),
            FieldCoverage(path="record.y", populated=2, seen=3, varied=True),
        ))
        assert lines == [f"  [known ] {known}   (seen 3)", "  [NEW   ] record.x   (seen 3)"]

    def test_all_fields_adds_the_varying_ones(self):
        lines = report_lines(self._coverage(
            FieldCoverage(path="record.y", populated=2, seen=3, varied=True),
        ), all_fields=True)
        assert lines == ["  [varies] record.y   (2/3)"]

    def test_a_listed_field_that_varied_is_named_for_removal(self):
        known = self._KNOWN
        lines = report_lines(self._coverage(
            FieldCoverage(path=known, populated=1, seen=3, varied=True),
        ))
        assert lines[-2:] == [
            "listed as constant but carrying data — remove from the list:",
            f"  {known}",
        ]


_PAYLOAD = "record.payload"


def _held_constant_by_the_pilot() -> set[str]:
    """The gen-3 fixture's constant members outside its events, each one its
    pilot holds constant too."""
    return {
        # One run, one mode.
        "record.mode", "record.run_id",
        # No rad counter, speed or initiative in any pilot game.
        "record.state.players[].rad", "record.state.players[].speed",
        "record.state.players[].initiative",
        # No rewrite of an event Forge's bus does not name, and no attacker
        # forbidden by a line that keys to nothing or held elsewhere.
        f"{_PAYLOAD}<RewritePayload>.incoming_unknown",
        f"{_PAYLOAD}<PlayabilityAttackersPayload>.forbidden[].responsible_unresolved",
        f"{_PAYLOAD}<PlayabilityAttackersPayload>.forbidden[].required_elsewhere",
        # No contribution changing a name.
        f"{_PAYLOAD}<ContinuousPayload>.contributions[].name",
    }


def _event_members_held_constant() -> set[str]:
    """The pilot's events: cost halves of one event type, and the durations
    and attributions a cost half, a rewrite's own events, a trigger's event
    and a turn-based action's hold fixed."""
    outgoing = f"{_PAYLOAD}<RewritePayload>.outgoing"
    durations = (f"{_PAYLOAD}<ActivationPayload>.events[]",
                 f"{_PAYLOAD}<RewritePayload>.incoming", outgoing,
                 f"{_PAYLOAD}<TriggerPayload>.event", "record.state.pending_event")
    attributions = (f"{_PAYLOAD}<ActivationPayload>.events[]",
                    f"{_PAYLOAD}<RewritePayload>.incoming", outgoing,
                    f"{_PAYLOAD}<TriggerPayload>.event",
                    f"{_PAYLOAD}<TurnActionPayload>.events[]", "record.state.pending_event")
    return {
        f"{_PAYLOAD}<ActivationPayload>.events[].type",
        *(f"{where}.duration" for where in durations),
        *(f"{where}.attributed_to" for where in attributions),
    }


class TestTheGen3Fields:
    """Each member of the gen-3 table, with a row of its own on a gen-3 corpus."""

    @staticmethod
    def _golden() -> list[EffectRecord]:
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import java_golden_records

        return [record_from_dict(raw) for raw in java_golden_records()]

    @staticmethod
    def _turn_action() -> EffectRecord:
        from dataclasses import replace

        from effects.domain.records import TurnActionPayload, TurnActionSubkind

        golden = TestTheGen3Fields._golden()[0]
        return replace(golden, kind=RecordKind.TURN_ACTION, moment=None,
                       subkind=TurnActionSubkind.DRAW, ability=None, link_id=None,
                       payload=TurnActionPayload(events=(
                           Event(type=EventType.ZONE_CHANGE, subjects=("E9",)),)))

    def test_every_listed_path_is_one_the_walk_reaches(self):
        """The goldens hold no failed check; the fixture holds no turn wrap's
        own payload beside a marker, so the two are read together."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import gen3_fixture_records

        walked = field_coverage([
            *self._golden(), self._turn_action(),
            *(record_from_dict(raw) for raw in gen3_fixture_records())])
        assert [path for path in WALK_PATHS if path not in walked] == []

    def test_the_gen3_fixture_reaches_every_listed_path_and_varies_most(self):
        """The fixture's records reach every member of the table, a real
        turn_action included, and vary each but the ones the 2026-11-21
        pilot it was cut from holds constant itself (its ``field-coverage``
        report): the members the writer fixes, the one run's mode, and what
        no pilot game showed."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import gen3_fixture_records

        coverage = field_coverage(record_from_dict(raw) for raw in gen3_fixture_records())
        assert [path for path in WALK_PATHS if path not in coverage] == []
        constant = {path for path in WALK_PATHS if coverage[path].constant}
        assert {path for path in constant if path in FIXED_PATHS} == {
            "record.generation", "record.variant_of", "record.synthetic", "record.curation",
            "record.state.tiers",
        }
        held = _held_constant_by_the_pilot() | _event_members_held_constant()
        assert constant - set(FIXED_PATHS) == held, sorted(constant - set(FIXED_PATHS))

    def test_the_members_the_fixture_holds_constant_are_written_and_vary_on_the_goldens(self):
        """The collector's own records write and vary the members an earlier
        fixture held constant, a rewrite's unknown incoming event among them,
        and write the gen-3 record moments' members, so a constant excused
        above never hides a dead channel."""
        from effects.infrastructure.record_codec import record_from_dict
        from tests.unit.effects.builders import java_golden_records

        coverage = field_coverage(record_from_dict(raw) for raw in java_golden_records())
        varying = [
            "record.payload<ResolutionPayload>.outcome", "record.copy_dropped",
            "record.payload<TurnActionPayload>.events",
            *(f"record.payload<RewritePayload>.{name}" for name in (
                "incoming_unknown", "buffered", "buffered_ran", "buffered_events")),
            *(f"record.payload<PlayabilityBlockersPayload>.forbidden[].{name}"
              for name in ("required_elsewhere", "responsible_own", "responsible_unresolved")),
        ]
        # A cost half's events are no optional member: the goldens' empty
        # lists are their default, so they read unpopulated here.
        written = [*varying, "record.payload<TurnActionPayload>.records_withheld",
                   "record.state.players[].max_land_plays"]
        assert [path for path in written if not coverage[path].populated] == []
        assert [path for path in varying if coverage[path].constant] == []

    def test_the_report_rows_each_gen3_field(self):
        lines = gen3_report_lines(field_coverage(self._golden()))
        assert lines[1].startswith("gen-3 record members: ")
        assert len(lines) == 2 + len(WALK_PATHS)
        status = {}
        for line in lines[2:]:
            tag, rest = line.strip()[1:].split("] ", 1)
            status[rest.split()[0]] = tag.strip()
        assert list(status) == list(WALK_PATHS)
        assert status["record.generation"] == "fixed"
        assert status["record.state.entities[].owner"] == "varies"
        # The goldens' turn actions: the game-end markers, whose events are
        # always empty, and a turn wrap's six actions, each with its own.
        assert status["record.payload<TurnActionPayload>.events"] == "varies"
        assert status["record.payload<RewritePayload>.buffered_events"] == "varies"
        assert status[
            "record.payload<PlayabilityBlockersPayload>.forbidden[].required_elsewhere"] == "varies"
        assert status[
            "record.payload<PlayabilityBlockersPayload>.forbidden[].responsible_own"] == "varies"
        assert status["record.state.global_.dungeons"] == "const"

    def test_an_empty_read_gets_no_section(self):
        assert gen3_report_lines(field_coverage([])) == []
        assert not any("gen-3 record" in line for line in report_lines(field_coverage([])))

    def test_the_section_follows_the_constant_fields(self):
        """After every constant-field row, set off by a blank line, and
        before the event parameters."""
        lines = report_lines(field_coverage(self._golden()))
        header = next(i for i, line in enumerate(lines) if line.startswith("gen-3 record"))
        events = next(i for i, line in enumerate(lines) if line.startswith("event parameters"))
        rows = [i for i, line in enumerate(lines)
                if line.startswith(("  [known ]", "  [NEW   ]"))]
        assert rows and max(rows) < header < events
        # The gen-3 section rows a fixed member too, under the same tag.
        assert lines.index(next(line for line in lines if line.startswith("  [fixed ]"))) < header
        assert lines[header - 1] == ""
