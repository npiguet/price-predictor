"""Every invariant the first collected corpus broke, planted and caught.

Each test builds a small shard directory that reproduces one measured defect
from that run and asserts the validator names it. The point is not that the
checks exist but that they *fire*: the corpus that motivated them parsed
cleanly, satisfied every schema test, and was worthless, so a validator that
passed it would be worse than none.

The synthetic shards are written through :class:`ShardWriter` rather than by
hand, so a defect is planted in the records themselves and has to survive a
real serialize/parse round trip to be found — the same path a collector's
output takes.
"""

from __future__ import annotations

import pytest

from effects.application.validate_corpus import (
    Finding,
    Thresholds,
    read_window,
    validate_corpus,
)
from effects.domain.event_schema import (
    ATTRIBUTION_ROOT,
    ATTRIBUTION_UNRESOLVED,
    Event,
    EventType,
)
from effects.domain.provenance import (
    ProvenanceKey,
    ProvenanceSidecar,
    SidecarLine,
)
from effects.domain.records import (
    ActivationPayload,
    CollectionMode,
    CombatPayload,
    Costs,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    RewriteResult,
    TriggerPayload,
)
from effects.domain.state_snapshot import (
    EntityState,
    GlobalState,
    InclusionTier,
    StateSnapshot,
)
from effects.infrastructure.record_io import ShardWriter
from effects.infrastructure.sidecar_io import SidecarCache, write_sidecar

_KEY = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)
_BEARS = "cardsfolder/g/grizzly_bears.txt"
_ALL_TIERS = frozenset(InclusionTier)
_STAGE_TWO = frozenset({
    InclusionTier.REFERENCED, InclusionTier.CORE, InclusionTier.UNREFERENCED_STACK,
})


def _snapshot(turn: int = 3, names: tuple[str, ...] = (), tiers=None) -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=turn, phase="main1", active="P0", priority="P0", stack_size=0,
        ),
        players=(),
        entities=tuple(
            EntityState(
                id=f"E{index}", name=name, zone="battlefield", controller="P0",
            )
            for index, name in enumerate(names)
        ),
        tiers=_STAGE_TWO if tiers is None else tiers,
    )


def _resolution(record_id: str, game_id: str = "run.0-L1.0", **overrides):
    fields: dict = {
        "record_id": record_id,
        "run_id": "run",
        "timestamp": "2026-09-09T10:00:00.000000Z",
        "game_id": game_id,
        "kind": RecordKind.RESOLUTION,
        "moment": Moment.RESOLUTION,
        "actor_player": "P0",
        "state": _snapshot(),
        "ability": (_KEY,),
        # The mode a run collected under is a property of the checkout, not of
        # the record, so the fixture states the healthy one rather than
        # inheriting the dataclass default: a window of `degraded` records is
        # itself one of the defects under test.
        "mode": CollectionMode.PATCHED,
        "payload": ResolutionPayload(
            events=(
                Event(
                    type=EventType.DAMAGE_DEALT,
                    subjects=("E1",),
                    attributed_to=ATTRIBUTION_ROOT,
                ),
            ),
        ),
    }
    fields.update(overrides)
    return EffectRecord(**fields)


def _activation(record_id: str, outcome=ResolutionOutcome.RESOLVED, **overrides):
    fields: dict = {
        "moment": Moment.ACTIVATION,
        "payload": ActivationPayload(
            costs=overrides.pop("costs", Costs(mana_by_color={"R": 1})),
            outcome=outcome,
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _trigger(record_id: str, ability=(_KEY,), fired=True, mode="ChangesZone",
             **overrides):
    fields: dict = {
        "kind": RecordKind.TRIGGER,
        "moment": None,
        "ability": ability,
        "payload": TriggerPayload(
            event=Event(
                type=EventType.ZONE_CHANGE,
                params={
                    "mode": mode, "from_zone": "battlefield",
                    "to_zone": "graveyard",
                },
            ),
            fired=fired,
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _mana(record_id: str, turn: int, **overrides):
    """The effect half of a mana ability: what the reservoir defers.

    Its snapshot is of the turn the mana was made, and it is written when the
    game ends, so it lands after every later turn's records.
    """
    fields: dict = {
        "state": _snapshot(turn=turn),
        "payload": ResolutionPayload(
            events=(
                Event(
                    type=EventType.MANA_PRODUCED,
                    subjects=("P0",),
                    params={"mana_by_color": {"G": 1}},
                    attributed_to=ATTRIBUTION_ROOT,
                ),
            ),
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _rewrite(record_id: str, ability=(_KEY,), **overrides):
    """A genuine in-place edit: 3 damage down to 1, reported as ``Updated``.

    The healthy default is the *rare* shape deliberately, because it is the one
    that exercises every rewrite field at once. The substitutions and the
    negatives that make up most of the channel are built per test.
    """
    fields: dict = {
        "kind": RecordKind.REWRITE,
        "moment": None,
        "ability": ability,
        "payload": RewritePayload(
            incoming=Event(type=EventType.DAMAGE_DEALT, params={"amount": 3}),
            outgoing=Event(type=EventType.DAMAGE_DEALT, params={"amount": 1}),
            result=RewriteResult.UPDATED,
            replaced_by=(_KEY,),
        ),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _combat(record_id: str, **overrides):
    fields: dict = {
        "kind": RecordKind.COMBAT,
        "moment": None,
        "ability": None,
        "payload": CombatPayload(attackers=("E0",)),
    }
    fields.update(overrides)
    return _resolution(record_id, **fields)


def _healthy() -> list[EffectRecord]:
    """A window nothing is wrong with, for every test to perturb.

    Deliberately varied where a defect would show as sameness: two outcomes,
    two games, paid costs, distinct payloads.
    """
    records: list[EffectRecord] = []
    for game in (0, 1):
        gid = f"run.0-L1.{game}"
        base = game * 100
        records += [
            _activation(f"run.0-L1.{base}", link_id=f"link.{game}", game_id=gid),
            _resolution(f"run.0-L1.{base + 1}", link_id=f"link.{game}", game_id=gid),
            _activation(
                f"run.0-L1.{base + 2}", outcome=ResolutionOutcome.FIZZLED,
                game_id=gid, costs=Costs(mana_by_color={"U": 2}, tapped=("E4",)),
            ),
            _trigger(f"run.0-L1.{base + 3}", game_id=gid, state=_snapshot(turn=4)),
            # The negative half of the trigger sample. Without it every window
            # reads 100% fired, which is the very skew the check looks for.
            _trigger(
                f"run.0-L1.{base + 6}", game_id=gid, fired=False,
                state=_snapshot(turn=4),
            ),
            _rewrite(f"run.0-L1.{base + 4}", game_id=gid, state=_snapshot(turn=5)),
            _combat(f"run.0-L1.{base + 5}", game_id=gid, state=_snapshot(turn=6)),
        ]
    return records


def _named(findings: list[Finding], fragment: str) -> Finding:
    return next(f for f in findings if fragment in f.name)


def _broken(findings: list[Finding]) -> set[str]:
    return {f.name for f in findings if not f.ok}


class TestAHealthyWindowPasses:
    """The baseline every other test perturbs by exactly one thing."""

    def test_nothing_is_reported_broken(self):
        assert _broken(validate_corpus(_healthy())) == set()

    def test_every_invariant_reports_a_measurement_not_just_a_verdict(self):
        for finding in validate_corpus(_healthy()):
            assert finding.measured
            assert finding.measured != "ok"
            assert finding.lines()[0].startswith(
                "[WATCH]" if finding.watched else "[PASS]"
            )

    def test_a_watched_finding_reports_a_number_and_judges_nothing(self):
        """The separation the report rests on: measured, never a verdict."""
        watched = [f for f in validate_corpus(_healthy()) if f.watched]
        assert watched, "at least the probe count is watched"
        for finding in watched:
            assert finding.ok, "a watched finding can never fail a run"
            assert finding.measured
            assert finding.lines()[0].startswith("[WATCH]")


class TestRecordIdCollision:
    """The defect that made 71-75% of the first corpus's ids repeats.

    Both counters behind an id live in one JVM's heap and the supervisor
    replaces that JVM hundreds of times per run, so a restart reissued ids the
    previous lifetime had already used.
    """

    def test_a_planted_collision_fails_the_run(self):
        records = _healthy()
        collided = _resolution("run.0-L1.0", game_id="run.0-L1.0")
        findings = validate_corpus([*records, collided])
        unique = _named(findings, "record_id is unique")
        assert not unique.ok
        assert "1 repeats" in unique.measured
        assert any("run.0-L1.0" in line for line in unique.detail)

    def test_a_collision_planted_in_a_real_shard_directory_fails(self, tmp_path):
        """End to end: written, gzip-free or not, discovered by glob, reparsed.

        A collision that only exists in memory proves nothing about a corpus —
        the id has to survive the writer and the reader to be the thing an
        operator would actually have on disk.
        """
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        # The second lifetime of the same worker slot, counting from zero again
        # into its own shard: exactly what the 60-second recycle produces.
        with ShardWriter(tmp_path, "run", 0, "L2") as writer:
            writer.write(_resolution("run.0-L1.0", game_id="run.0-L1.0"))

        findings = validate_corpus(read_window(tmp_path))
        unique = _named(findings, "record_id is unique")
        assert not unique.ok, unique.measured
        assert "run.0-L1.0" in " ".join(unique.detail)

    def test_two_lifetimes_that_namespace_their_ids_pass(self, tmp_path):
        """The same restart, with the fix in place, is not a defect."""
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        with ShardWriter(tmp_path, "run", 0, "L2") as writer:
            writer.write(_resolution("run.0-L2.0", game_id="run.0-L2.0"))
        assert _named(validate_corpus(read_window(tmp_path)), "record_id").ok


class TestGameIdNamesOneGame:
    def test_a_backward_turn_jump_inside_one_game_id_fails(self):
        records = _healthy() + [
            _resolution("run.0-L1.900", game_id="run.0-L1.0", state=_snapshot(turn=9)),
            # A second game's records landing in the first game's bucket: the
            # turn counter restarts, which no single game ever does.
            _resolution("run.0-L1.901", game_id="run.0-L1.0", state=_snapshot(turn=1)),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert not finding.ok
        assert "1 span a backward turn jump" in finding.measured

    def test_a_deferred_half_lagging_one_turn_is_not_a_game_boundary(self):
        """Records are written when observed but snapshot what they describe."""
        records = _healthy() + [
            _resolution("run.0-L1.900", game_id="run.0-L1.0", state=_snapshot(turn=9)),
            _resolution("run.0-L1.901", game_id="run.0-L1.0", state=_snapshot(turn=8)),
        ]
        assert _named(validate_corpus(records), "one game").ok

    def test_a_second_deck_under_one_game_id_fails(self):
        """804 game ids held 31,662 games, so a bucket showed ~460 card names."""
        crowd = tuple(f"Card {index}" for index in range(200))
        records = _healthy() + [
            _resolution(
                "run.0-L1.902", game_id="run.0-L1.0",
                state=_snapshot(turn=7, names=crowd),
            ),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert not finding.ok
        assert "distinct card names" in " ".join(finding.detail)

    def test_the_name_ceiling_is_configurable(self):
        crowd = tuple(f"Card {index}" for index in range(200))
        records = _healthy() + [
            _resolution(
                "run.0-L1.902", game_id="run.0-L1.0",
                state=_snapshot(turn=7, names=crowd),
            ),
        ]
        loose = Thresholds(max_names_per_game=500)
        assert _named(validate_corpus(records, loose), "one game").ok


class TestLinkHalvesPair:
    def test_an_unpaired_half_fails(self):
        records = _healthy() + [
            _activation(f"run.0-L1.{index}", link_id=f"lonely.{index}")
            for index in range(200, 210)
        ]
        finding = _named(validate_corpus(records), "link_id")
        assert not finding.ok
        assert "unpaired" in finding.measured

    def test_a_link_id_shared_by_three_halves_fails(self):
        """A link id whose tail is a JVM-static counter collides like an id does."""
        records = _healthy() + [
            _resolution("run.0-L1.910", link_id="link.0"),
        ]
        finding = _named(validate_corpus(records), "link_id")
        assert not finding.ok
        assert "3 half(s)" in " ".join(finding.detail)

    def test_a_few_truncated_tails_stay_within_tolerance(self):
        """A killed worker loses its last block, so some halves have no partner."""
        paired = []
        for index in range(100):
            paired += [
                _activation(f"run.0-L1.{2 * index}", link_id=f"pair.{index}"),
                _resolution(f"run.0-L1.{2 * index + 1}", link_id=f"pair.{index}"),
            ]
        records = paired + [_activation("run.0-L1.999", link_id="tail")]
        finding = _named(validate_corpus(records), "link_id")
        assert finding.ok, finding.measured


class TestActingLines:
    def test_a_trigger_that_names_no_line_fails(self):
        """727,308 trigger records said only that some unnamed trigger saw an event."""
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", ability=None) for index in range(300, 310)
        ]
        finding = _named(validate_corpus(records), "trigger and rewrite")
        assert not finding.ok
        assert "carry the field" in finding.measured

    def test_a_rewrite_that_names_no_line_fails(self):
        records = _healthy() + [
            _rewrite(f"run.0-L1.{index}", ability=None) for index in range(300, 310)
        ]
        assert not _named(validate_corpus(records), "trigger and rewrite").ok

    def test_resolution_records_below_the_keyed_floor_fail(self):
        """41% of the first corpus's resolution records resolved to no key."""
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(), ability_unresolved="unindexable",
            )
            for index in range(400, 410)
        ]
        finding = _named(validate_corpus(records), "resolution records name")
        assert not finding.ok
        assert "unindexable" in " ".join(finding.detail)

    def test_the_keyed_floor_is_configurable(self):
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(), ability_unresolved="engine_effect",
            )
            for index in range(400, 410)
        ]
        loose = Thresholds(min_keyed_rate=0.1)
        assert _named(validate_corpus(records, loose), "resolution records name").ok

    def test_a_kind_absent_from_the_window_is_not_a_failure(self):
        """A two-minute window need not contain a continuous record."""
        finding = _named(validate_corpus(_healthy()), "continuous records name")
        assert finding.ok
        assert "no records of this kind" in finding.measured


class TestOutcome:
    def test_one_outcome_everywhere_fails(self):
        """The activation half hardcoded `resolved`, so 100% of them said so."""
        records = [
            record for record in _healthy()
            if not (
                isinstance(record.payload, ActivationPayload)
                and record.payload.outcome is ResolutionOutcome.FIZZLED
            )
        ]
        finding = _named(validate_corpus(records), "outcome")
        assert not finding.ok
        assert "1 distinct" in finding.measured
        assert "resolved" in finding.measured

    def test_two_outcomes_pass_and_the_counts_are_reported(self):
        finding = _named(validate_corpus(_healthy()), "outcome")
        assert finding.ok
        assert "fizzled 2" in finding.measured
        assert "resolved 2" in finding.measured


class TestCosts:
    def test_costs_empty_on_every_record_fails(self):
        """Four cost lists were never once populated in 14.7M records."""
        records = [
            _activation(f"run.0-L1.{index}", costs=Costs())
            for index in range(500, 510)
        ]
        finding = _named(validate_corpus(records), "cost fields")
        assert not finding.ok
        assert "0/10" in finding.measured

    def test_the_report_names_each_dead_channel(self):
        finding = _named(validate_corpus(_healthy()), "cost fields")
        assert finding.ok
        assert "sacrificed 0" in finding.measured
        assert "mana_by_color 4" in finding.measured

    def test_a_channel_every_deck_pays_reading_zero_fails(self):
        """`tapped` was populated zero times in 14.7M records.

        The whole cost object was not empty — mana came off the printed cost —
        so a check that only asks whether anything was ever paid passes the
        corpus that motivated it.
        """
        records = [
            _activation(
                f"run.0-L1.{index}", costs=Costs(mana_by_color={"R": 1}),
            )
            for index in range(200, 500)
        ]
        finding = _named(validate_corpus(records), "cost fields")
        assert not finding.ok
        assert "never populated over 300 activation records: tapped" in (
            " ".join(finding.detail)
        )

    def test_a_window_too_small_to_judge_says_so_rather_than_failing(self):
        """Scarcity and a dead channel look identical until the window is big."""
        records = [
            _activation(
                f"run.0-L1.{index}", costs=Costs(mana_by_color={"R": 1}),
            )
            for index in range(200, 210)
        ]
        finding = _named(validate_corpus(records), "cost fields")
        assert finding.ok
        assert "under the 200" in " ".join(finding.detail)

    def test_the_evidence_floor_is_configurable(self):
        records = [
            _activation(
                f"run.0-L1.{index}", costs=Costs(mana_by_color={"R": 1}),
            )
            for index in range(200, 210)
        ]
        strict = Thresholds(min_cost_evidence=5)
        assert not _named(validate_corpus(records, strict), "cost fields").ok


class TestSnapshotTiers:
    def test_a_tier_depth_chosen_per_collector_fails(self):
        """Tier 4 appeared on the interventional records and nowhere else.

        That makes ``state.tiers`` a perfect predictor of a field the schema
        forbids the model to see, which is a leak rather than a depth choice.
        """
        records = _healthy() + [
            _resolution(
                "run.0-L1.600", interventional=True, fork=True,
                state=_snapshot(tiers=_ALL_TIERS),
            ),
        ]
        finding = _named(validate_corpus(records), "tier depth")
        assert not finding.ok
        assert "2 distinct tier vectors" in finding.measured
        assert any("[1, 2, 3, 4]" in line for line in finding.detail)

    def test_one_vector_everywhere_passes(self):
        assert _named(validate_corpus(_healthy()), "tier depth").ok


class TestDuplicates:
    def test_a_flood_of_identical_records_in_one_game_fails(self):
        """N triggers of one mode against one event rendered byte-identically."""
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", state=_snapshot(turn=4))
            for index in range(700, 720)
        ]
        finding = _named(validate_corpus(records), "duplicates")
        assert not finding.ok
        assert "trigger" in " ".join(finding.detail)

    def test_identical_records_in_different_games_are_not_duplicates(self):
        """Two games can legitimately reach the same board and the same answer."""
        records = _healthy() + [
            _trigger("run.0-L1.800", game_id="run.0-L1.7", state=_snapshot(turn=4)),
        ]
        assert _named(validate_corpus(records), "duplicates").ok

    def test_the_identity_fields_do_not_hide_a_duplicate(self):
        """Two records differing only in id and timestamp are one observation."""
        records = _healthy() + [
            _trigger(
                f"run.0-L1.{index}", state=_snapshot(turn=4),
                timestamp=f"2026-09-09T11:00:{index - 700:02d}.000000Z",
            )
            for index in range(700, 720)
        ]
        assert not _named(validate_corpus(records), "duplicates").ok


class TestTheWindow:
    def test_limit_stops_reading(self, tmp_path):
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert len(list(read_window(tmp_path, limit=2))) == 2
        assert len(list(read_window(tmp_path))) == len(_healthy())

    def test_an_empty_window_reports_rather_than_raising(self):
        findings = validate_corpus([])
        assert all(finding.ok for finding in findings)
        assert _named(findings, "record_id is unique").measured.startswith("0 repeats")


class TestTheCli:
    """The subcommand exits non-zero on a broken window and zero on a good one."""

    def _run(self, directory, **overrides) -> int:
        from effects.infrastructure.cli import build_parser

        args = build_parser().parse_args(
            ["validate-corpus", "--effect-records", str(directory)]
        )
        for name, value in overrides.items():
            setattr(args, name, value)
        return args.func(args)

    def test_a_clean_directory_exits_zero(self, tmp_path, capsys):
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert self._run(tmp_path) == 0
        assert "invariants hold" in capsys.readouterr().out

    def test_a_planted_collision_exits_non_zero_and_names_it(self, tmp_path, capsys):
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        with ShardWriter(tmp_path, "run", 0, "L2") as writer:
            writer.write(_resolution("run.0-L1.0", game_id="run.0-L1.0"))
        assert self._run(tmp_path) == 1
        out = capsys.readouterr().out
        assert "[FAIL] record_id is unique across the run's shards" in out
        assert "invariants broken" in out

    def test_an_empty_directory_exits_non_zero(self, tmp_path, capsys):
        assert self._run(tmp_path) == 1
        assert "no shards" in capsys.readouterr().out

    def test_the_new_thresholds_are_the_contracts(self):
        """Pinned here because a default that drifts changes what a corpus means."""
        from effects.infrastructure.cli import build_parser

        args = build_parser().parse_args(["validate-corpus"])
        assert args.max_duplicate_event_rate == 0.02
        assert args.max_trigger_fired_share == 0.65
        assert args.min_zone_change_from_zone_rate is None
        assert args.min_attributed_rate is None
        assert args.cards_folders is None

    def test_watched_findings_are_counted_apart_from_judged_ones(
        self, tmp_path, capsys,
    ):
        """The summary must not claim a verdict on a number nothing judged."""
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert self._run(tmp_path) == 0
        out = capsys.readouterr().out
        assert "[WATCH]" in out
        assert "watched, not judged" in out

    def test_a_watched_number_alone_never_fails_the_run(self, tmp_path, capsys):
        """Zero probes and no attribution are reported, and exit stays zero."""
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert self._run(tmp_path) == 0
        out = capsys.readouterr().out
        assert "[WATCH] probe forks were taken: 0 damage-step probe forks" in out

    def test_a_floor_passed_on_the_command_line_turns_a_watch_into_a_failure(
        self, tmp_path, capsys,
    ):
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
            writer.write(
                _resolution(
                    "run.0-L1.900",
                    payload=ResolutionPayload(events=(
                        Event(
                            type=EventType.ZONE_CHANGE, subjects=("E1",),
                            params={"to_zone": "graveyard"},
                        ),
                    )),
                )
            )
        assert self._run(tmp_path, min_zone_change_from_zone_rate=0.9) == 1
        out = capsys.readouterr().out
        assert "[FAIL] zone_change events say where the card came from" in out

    def test_a_keyword_sidecar_mismatch_exits_non_zero_end_to_end(
        self, tmp_path, capsys,
    ):
        """The launch blocker, planted in a shard directory and rejected.

        The whole path an operator would run: converted tree on disk, shards
        beside it, one key numbered the way the collector numbered it and the
        way the converter did not.
        """
        _write_bears_sidecar(tmp_path, (0, 2))
        shards = tmp_path / "records"
        with ShardWriter(shards, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
            writer.write(
                _trigger(
                    "run.0-L1.900", state=_snapshot(turn=7),
                    ability=(ProvenanceKey(_BEARS, 0, "keyword", 1),),
                )
            )
        assert self._run(
            shards, cards_folders=[str(tmp_path / "cardsfolder")],
        ) == 1
        out = capsys.readouterr().out
        assert "[FAIL] keyword provenance keys join their sidecar" in out

    def test_the_same_directory_with_matching_ordinals_exits_zero(
        self, tmp_path, capsys,
    ):
        _write_bears_sidecar(tmp_path, (0, 2))
        shards = tmp_path / "records"
        with ShardWriter(shards, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
            writer.write(
                _trigger(
                    "run.0-L1.900", state=_snapshot(turn=7),
                    ability=(ProvenanceKey(_BEARS, 0, "keyword", 2),),
                )
            )
        assert self._run(
            shards, cards_folders=[str(tmp_path / "cardsfolder")],
        ) == 0
        assert "[PASS] keyword provenance keys join their sidecar" in (
            capsys.readouterr().out
        )

    def test_a_missing_converted_tree_reports_unchecked_rather_than_failing(
        self, tmp_path, capsys,
    ):
        """No tree on this machine is not evidence that the corpus joins."""
        with ShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert self._run(
            tmp_path, cards_folders=[str(tmp_path / "not-here")],
        ) == 0
        assert "[WATCH] keyword provenance keys join their sidecar" in (
            capsys.readouterr().out
        )


class TestTheManaReservoirFlushIsNotAGameBoundary:
    """The exemption that stops the checklist from crying wolf.

    Mana activations are reservoir-sampled and flushed at ``close()`` carrying
    the snapshot of the turn the mana was made, so every healthy game ends with
    records that step backwards several turns. Un-exempted, the check failed
    168 of 177 games on a corpus with nothing wrong with it — and an invariant
    that fails on every good run is one an operator stops reading.
    """

    def test_a_flushed_mana_record_does_not_read_as_a_second_game(self):
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _mana("run.0-L1.901", turn=2),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert finding.ok, finding.measured

    def test_the_report_says_how_many_records_were_exempted(self):
        """Silently dropping records from a check is how a check starts lying."""
        records = _healthy() + [_mana(f"run.0-L1.{i}", turn=1) for i in (901, 902)]
        finding = _named(validate_corpus(records), "one game")
        assert "2 records exempted as deferred mana-reservoir flushes" in (
            finding.measured
        )

    def test_a_healthy_window_reports_zero_exempted(self):
        assert "0 records exempted" in _named(
            validate_corpus(_healthy()), "one game"
        ).measured

    def test_a_flushed_record_does_not_raise_the_bar_for_later_records(self):
        """Exempt in both directions, or the exemption invents a jump.

        A mana record whose snapshot is *ahead* of the game would otherwise set
        a high-water mark the next ordinary record steps back from.
        """
        records = _healthy() + [
            _mana("run.0-L1.900", turn=30),
            _resolution("run.0-L1.901", state=_snapshot(turn=7)),
        ]
        assert _named(validate_corpus(records), "one game").ok

    def test_a_real_game_boundary_is_still_caught_beside_a_flush(self):
        """Precise, not permissive: only the mana half is exempt."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _mana("run.0-L1.901", turn=2),
            _resolution("run.0-L1.902", state=_snapshot(turn=1)),
        ]
        finding = _named(validate_corpus(records), "one game")
        assert not finding.ok
        assert "1 span a backward turn jump" in finding.measured

    def test_a_resolution_that_makes_mana_and_more_is_not_exempt(self):
        """A dual-purpose line is not a reservoir record, so it is not exempt."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _resolution(
                "run.0-L1.901", state=_snapshot(turn=2),
                payload=ResolutionPayload(events=(
                    Event(type=EventType.MANA_PRODUCED, subjects=("P0",)),
                    Event(type=EventType.LIFE_CHANGE, subjects=("P0",)),
                )),
            ),
        ]
        assert not _named(validate_corpus(records), "one game").ok

    def test_an_activation_half_is_never_exempt(self):
        """Nothing defers a cost half, so a jump on one is a game boundary."""
        records = _healthy() + [
            _resolution("run.0-L1.900", state=_snapshot(turn=9)),
            _activation("run.0-L1.901", state=_snapshot(turn=2)),
        ]
        assert not _named(validate_corpus(records), "one game").ok


class TestAnEmptyAbilitySaysWhy:
    """`ability: []` alone conflates a missing line with a broken resolver."""

    def test_an_empty_ability_with_no_reason_fails(self):
        records = _healthy() + [
            _resolution(f"run.0-L1.{index}", ability=())
            for index in range(400, 405)
        ]
        finding = _named(validate_corpus(records), "empty ability says why")
        assert not finding.ok
        assert "5 of 5" in finding.measured
        assert any("run.0-L1.400" in line for line in finding.detail)

    def test_an_empty_ability_that_names_its_reason_passes(self):
        records = _healthy() + [
            _resolution(
                f"run.0-L1.{index}", ability=(),
                ability_unresolved="engine_effect",
            )
            for index in range(400, 405)
        ]
        finding = _named(validate_corpus(records), "empty ability says why")
        assert finding.ok, finding.measured
        assert "0 of 5" in finding.measured

    def test_a_null_ability_is_the_other_defect_and_not_this_one(self):
        """A kind that never calls .ability() is the acting-line check's job."""
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", ability=None)
            for index in range(300, 305)
        ]
        assert _named(validate_corpus(records), "empty ability says why").ok
        assert not _named(validate_corpus(records), "trigger and rewrite").ok


def _write_bears_sidecar(root, keyword_indices):
    """A sidecar for Grizzly Bears declaring exactly these keyword ordinals."""
    sidecar = ProvenanceSidecar(
        card="Grizzly Bears",
        script_file=_BEARS,
        lines=tuple(
            SidecarLine(
                line_index=row,
                line_kind="keyword",
                provenance=(ProvenanceKey(_BEARS, 0, "keyword", index),),
            )
            for row, index in enumerate(keyword_indices)
        ),
    )
    write_sidecar(
        sidecar, root / "cardsfolder" / "g" / "grizzly_bears.provenance.json",
    )
    return SidecarCache({"cardsfolder": root / "cardsfolder"})


class TestKeywordKeysJoinTheirSidecar:
    """The launch blocker: two sides numbering one card's keywords differently.

    The converter and the collector disagreed on a keyword's ordinal, so every
    record still joined — to the wrong printed line. Nothing downstream could
    notice, because a wrong row and a right one are the same shape.
    """

    def _record(self, index: int, record_id: str = "run.0-L1.900"):
        return _trigger(
            record_id, ability=(ProvenanceKey(_BEARS, 0, "keyword", index),),
        )

    def test_a_planted_ordinal_mismatch_is_rejected(self, tmp_path):
        # The sidecar declares keyword ordinals 0 and 2; the collector's key
        # says 1, which is inside what the face declared and names no line.
        sidecars = _write_bears_sidecar(tmp_path, (0, 2))
        findings = validate_corpus([*_healthy(), self._record(1)], None, sidecars)
        finding = _named(findings, "keyword provenance keys")
        assert not finding.ok, finding.measured
        assert "1/1 keyword keys fail to join" in finding.measured
        assert any("keyword[1]" in line for line in finding.detail)

    def test_a_key_the_sidecar_declares_joins(self, tmp_path):
        sidecars = _write_bears_sidecar(tmp_path, (0, 2))
        findings = validate_corpus([*_healthy(), self._record(2)], None, sidecars)
        finding = _named(findings, "keyword provenance keys")
        assert finding.ok, finding.detail
        assert "0/1 keyword keys fail to join" in finding.measured

    def test_a_runtime_only_trait_past_the_declared_tail_is_not_a_mismatch(
        self, tmp_path,
    ):
        """Level up and bestow append traits no card script declares."""
        sidecars = _write_bears_sidecar(tmp_path, (0, 1))
        findings = validate_corpus([*_healthy(), self._record(7)], None, sidecars)
        assert _named(findings, "keyword provenance keys").ok

    def test_a_card_with_no_sidecar_fails_rather_than_being_skipped(
        self, tmp_path,
    ):
        sidecars = _write_bears_sidecar(tmp_path, (0,))
        orphan = _trigger(
            "run.0-L1.901",
            ability=(ProvenanceKey("cardsfolder/x/nowhere.txt", 0, "keyword", 0),),
        )
        finding = _named(
            validate_corpus([*_healthy(), orphan], None, sidecars),
            "keyword provenance keys",
        )
        assert not finding.ok
        assert any("no sidecar" in line for line in finding.detail)

    def test_a_key_in_a_tree_this_run_was_not_given_is_unchecked_not_broken(
        self, tmp_path,
    ):
        """A stage-four corpus names variant-scripts keys; absence is not proof.

        A validate run without that tree has nothing to join them against, and
        counting them as mismatches would fail every variant run for a reason
        that says nothing about the corpus.
        """
        sidecars = _write_bears_sidecar(tmp_path, (0,))
        variant = _trigger(
            "run.0-L1.901",
            ability=(
                ProvenanceKey("variant-scripts/bears_x1.txt", 0, "keyword", 0),
            ),
        )
        finding = _named(
            validate_corpus([*_healthy(), variant], None, sidecars),
            "keyword provenance keys",
        )
        assert finding.ok, finding.detail
        assert "1 unchecked" in finding.measured
        assert any("variant-scripts" in line for line in finding.detail)

    def test_without_a_converted_tree_the_check_reports_unchecked(self):
        finding = _named(validate_corpus(_healthy()), "keyword provenance keys")
        assert finding.watched
        assert "none checked" in finding.measured

    def test_a_planted_mismatch_in_a_real_shard_directory_is_rejected(
        self, tmp_path,
    ):
        """End to end: written, discovered by glob, reparsed, then joined.

        A key that only exists in memory proves nothing about a corpus — the
        mismatch has to survive the writer and the reader to be the thing an
        operator would actually have on disk.
        """
        sidecars = _write_bears_sidecar(tmp_path, (0, 2))
        shards = tmp_path / "records"
        with ShardWriter(shards, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
            writer.write(self._record(1))
        finding = _named(
            validate_corpus(read_window(shards), None, sidecars),
            "keyword provenance keys",
        )
        assert not finding.ok, finding.measured


class TestCollectionMode:
    def test_a_degraded_record_fails_the_run(self):
        """One stock checkout in the pool silently loses four record kinds."""
        records = _healthy() + [
            _resolution("run.0-L1.900", mode=CollectionMode.DEGRADED),
        ]
        finding = _named(validate_corpus(records), "patched mode")
        assert not finding.ok
        assert "degraded 1" in finding.measured
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_patched_window_passes_and_reports_the_count(self):
        finding = _named(validate_corpus(_healthy()), "patched mode")
        assert finding.ok
        assert "0/14 not patched" in finding.measured


class TestDuplicateEventsInsideOneRecord:
    """13.68% of the smoke corpus's events, invisible to the record check."""

    def _twice(self, index: int):
        """One record whose single outcome is written twice.

        Each record sits on its own turn so the *records* stay distinct: the
        point of this class is that the record-level duplicate check sees
        nothing, and it would see these if they were copies of each other.
        """
        event = Event(
            type=EventType.DAMAGE_DEALT, subjects=("E1",),
            params={"amount": 3}, attributed_to=ATTRIBUTION_ROOT,
        )
        return _resolution(
            f"run.0-L1.{index}", state=_snapshot(turn=index - 890),
            payload=ResolutionPayload(events=(event, event)),
        )

    def test_one_outcome_written_twice_fails(self):
        records = _healthy() + [
            self._twice(i) for i in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "repeats an event")
        assert not finding.ok
        assert "damage_dealt" in " ".join(finding.detail)

    def test_the_record_level_check_cannot_see_it(self):
        """Why this check exists at all: the records themselves are unique."""
        records = _healthy() + [
            self._twice(i) for i in range(900, 910)
        ]
        assert _named(validate_corpus(records), "exact duplicates").ok

    def test_two_events_differing_only_in_subject_are_not_duplicates(self):
        """Two damage events on two creatures are two outcomes."""
        records = _healthy() + [
            _resolution(
                "run.0-L1.900",
                payload=ResolutionPayload(events=(
                    Event(type=EventType.DAMAGE_DEALT, subjects=("E1",)),
                    Event(type=EventType.DAMAGE_DEALT, subjects=("E2",)),
                )),
            ),
        ]
        assert _named(validate_corpus(records), "repeats an event").ok

    def test_a_rewrite_carrying_one_event_twice_is_not_a_duplicate(self):
        """A pre-contract shard wrote the same event into both slots.

        Those records are still in the corpus, so the event-duplication check
        still has to ignore a rewrite's own pair — the *rewrite* checks are
        what report the copy, and they report it as the collector defect it is
        rather than as a duplicated outcome.
        """
        same = Event(type=EventType.DAMAGE_DEALT, params={"amount": 3})
        records = _healthy() + [
            _rewrite(
                "run.0-L1.900",
                payload=RewritePayload(incoming=same, outgoing=same),
            ),
        ]
        assert _named(validate_corpus(records), "repeats an event").ok

    def test_the_rate_is_configurable(self):
        records = _healthy() + [
            self._twice(i) for i in range(900, 910)
        ]
        loose = Thresholds(max_duplicate_event_rate=0.9)
        assert _named(validate_corpus(records, loose), "repeats an event").ok


class TestRewriteRecordsSayWhatHappened:
    """The field the whole channel hangs off.

    Without ``result`` the payload models an edit-the-event mechanism Forge
    does not have: it substitutes by running a different ability, so the before
    and after maps come back byte-identical, 87% of the hook's calls were
    dropped as identity rewrites, and the channel held 34 records in 1.88M
    against a 7% share of the trainer's mix.
    """

    def _substitution(self, record_id: str, **payload_kwargs):
        return _rewrite(
            record_id,
            payload=RewritePayload(
                incoming=Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                               params={"to_zone": "graveyard"}),
                **payload_kwargs,
            ),
        )

    def test_a_healthy_window_reports_every_record_carrying_a_result(self):
        finding = _named(validate_corpus(_healthy()), "which replacement result")
        assert finding.ok
        assert "2/2" in finding.measured
        assert "updated 2" in finding.measured

    def test_a_record_with_no_result_fails_and_says_which_two_causes(self):
        records = _healthy() + [
            self._substitution("run.0-L1.900", outgoing=None),
        ]
        finding = _named(validate_corpus(records), "which replacement result")
        assert not finding.ok
        assert "absent 1" in finding.measured
        assert any("argument slot" in line for line in finding.detail)

    def test_the_histogram_names_every_outcome_the_window_saw(self):
        records = _healthy() + [
            self._substitution(f"run.0-L1.{900 + index}", result=result)
            for index, result in enumerate(
                (RewriteResult.REPLACED, RewriteResult.PREVENTED,
                 RewriteResult.SKIPPED, RewriteResult.NOT_REPLACED)
            )
        ]
        measured = _named(
            validate_corpus(records), "which replacement result"
        ).measured
        for name in ("replaced 1", "prevented 1", "skipped 1", "not_replaced 1"):
            assert name in measured

    def test_a_window_with_no_rewrites_says_so_rather_than_failing(self):
        records = [r for r in _healthy() if r.kind is not RecordKind.REWRITE]
        assert _named(validate_corpus(records), "which replacement result").ok


class TestARewriteDoesNotCopyItsEvent:
    """The defect that made the channel unreadable, now measured.

    A record whose two halves are byte-identical cannot be told from a
    replacement that changed a parameter back to its own value, and there is no
    such thing. Null is the honest spelling.
    """

    def _copy(self, record_id: str, result=RewriteResult.REPLACED, **overrides):
        same = Event(type=EventType.DAMAGE_DEALT, params={"amount": 4})
        return _rewrite(
            record_id,
            payload=RewritePayload(incoming=same, outgoing=same, result=result),
            **overrides,
        )

    def test_a_healthy_window_copies_nothing(self):
        finding = _named(validate_corpus(_healthy()), "is not a copy of")
        assert finding.ok
        assert "0/2" in finding.measured

    def test_a_copied_event_fails_and_names_the_record(self):
        records = _healthy() + [self._copy("run.0-L1.900")]
        finding = _named(validate_corpus(records), "is not a copy of")
        assert not finding.ok
        assert "1/3" in finding.measured
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_pre_contract_record_is_not_counted_against_the_writer(self):
        """It had no null to write. Failing on its shards would report a
        defect that was fixed rather than one that is present."""
        same = Event(type=EventType.DAMAGE_DEALT, params={"amount": 4})
        records = _healthy() + [
            _rewrite(
                "run.0-L1.900",
                payload=RewritePayload(incoming=same, outgoing=same),
            ),
        ]
        finding = _named(validate_corpus(records), "is not a copy of")
        assert finding.ok
        assert "0/2" in finding.measured, "the no-result record is not judged"

    def test_a_genuine_edit_back_to_the_same_value_is_still_reported(self):
        """There is no such replacement, so this is the collector, not Magic."""
        records = _healthy() + [
            self._copy("run.0-L1.900", result=RewriteResult.UPDATED),
        ]
        assert not _named(validate_corpus(records), "is not a copy of").ok


class TestOutgoingAgreesWithTheResult:
    """``prevented`` and ``skipped`` return above the ability call.

    Both are bare ``return`` statements in ``executeReplacementInternal``: no
    ability ran and nothing was written to the parameter map. A record of one
    carrying a payload is a reflective listener reading a stale argument slot,
    which nothing compiles against and so nothing else would catch.
    """

    def _with(self, record_id: str, result, **payload_kwargs):
        return _rewrite(
            record_id,
            payload=RewritePayload(
                incoming=Event(type=EventType.DAMAGE_DEALT,
                               params={"amount": 3}),
                result=result,
                **payload_kwargs,
            ),
        )

    def test_a_healthy_window_reports_none(self):
        finding = _named(validate_corpus(_healthy()), "outgoing is null on")
        assert finding.ok
        assert finding.measured.startswith("0 ")

    def test_a_prevented_record_carrying_an_outgoing_fails(self):
        records = _healthy() + [
            self._with(
                "run.0-L1.900", RewriteResult.PREVENTED,
                outgoing=Event(type=EventType.DAMAGE_DEALT,
                               params={"amount": 0}),
            ),
        ]
        finding = _named(validate_corpus(records), "outgoing is null on")
        assert not finding.ok
        assert any("run.0-L1.900" in line for line in finding.detail)

    def test_a_skipped_record_naming_a_substituted_ability_fails(self):
        records = _healthy() + [
            self._with(
                "run.0-L1.900", RewriteResult.SKIPPED, replaced_by=(_KEY,),
            ),
        ]
        assert not _named(validate_corpus(records), "outgoing is null on").ok

    def test_a_prevented_record_with_neither_passes(self):
        records = _healthy() + [
            self._with("run.0-L1.900", RewriteResult.PREVENTED),
        ]
        assert _named(validate_corpus(records), "outgoing is null on").ok

    def test_not_replaced_may_carry_an_outgoing(self):
        """Its prevention branch writes PreventedAmount into the map on the
        way out, so this one is a real shape rather than a stale read."""
        records = _healthy() + [
            self._with(
                "run.0-L1.900", RewriteResult.NOT_REPLACED,
                outgoing=Event(type=EventType.DAMAGE_DEALT,
                               params={"amount": 0}),
            ),
        ]
        assert _named(validate_corpus(records), "outgoing is null on").ok


class TestRewritesNameTheAbilityThatRanInstead:
    """Watched, because a ``ReplacementResult$`` script runs no ability at all.

    The ceiling is genuinely below 1 and nobody has measured how far, so this
    reports the number and judges nothing until someone passes a floor.
    """

    def _substitution(self, record_id: str, **payload_kwargs):
        payload_kwargs.setdefault("result", RewriteResult.REPLACED)
        return _rewrite(
            record_id,
            payload=RewritePayload(
                incoming=Event(type=EventType.ZONE_CHANGE, subjects=("E1",),
                               params={"to_zone": "graveyard"}),
                **payload_kwargs,
            ),
        )

    def test_it_is_watched_by_default(self):
        finding = _named(validate_corpus(_healthy()), "ran instead")
        assert finding.watched
        assert finding.ok
        assert "watched, no floor" in finding.measured

    def test_a_resolver_that_keys_nothing_reads_as_zero(self):
        records = _healthy() + [self._substitution("run.0-L1.900")]
        finding = _named(validate_corpus(records), "ran instead")
        assert "2/3" in finding.measured
        assert finding.ok, "watched findings never fail a run"

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [self._substitution("run.0-L1.900")]
        strict = Thresholds(min_rewrite_replaced_by_rate=0.9)
        finding = _named(validate_corpus(records, strict), "ran instead")
        assert not finding.watched
        assert not finding.ok

    def test_results_that_run_no_ability_are_outside_the_rate(self):
        """``prevented`` never reaches an ability, so counting it would drag
        the rate down and blame the resolver for the handler's control flow."""
        records = _healthy() + [
            self._substitution("run.0-L1.900", result=RewriteResult.PREVENTED),
        ]
        assert "2/2" in _named(validate_corpus(records), "ran instead").measured


class TestTriggerNegatives:
    def test_a_window_of_positives_only_fails(self):
        records = _healthy() + [
            _trigger(f"run.0-L1.{index}", state=_snapshot(turn=7 + index))
            for index in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "draw negatives")
        assert not finding.ok
        assert "fired:not-fired" in finding.measured

    def test_a_balanced_window_passes_and_reports_the_ratio(self):
        finding = _named(validate_corpus(_healthy()), "draw negatives")
        assert finding.ok, finding.measured
        assert "2:2 fired:not-fired" in finding.measured

    def test_the_ratio_is_reported_per_evaluated_mode(self):
        """One mode drawing no negatives hides behind every other mode."""
        records = _healthy() + [
            _trigger(
                f"run.0-L1.{index}", mode="Attacks",
                state=_snapshot(turn=7 + index),
            )
            for index in range(900, 903)
        ]
        finding = _named(validate_corpus(records), "draw negatives")
        assert any(
            line.startswith("Attacks: 3:0") for line in finding.detail
        ), finding.detail

    def test_a_window_with_no_triggers_is_not_a_failure(self):
        records = [
            record for record in _healthy()
            if record.kind is not RecordKind.TRIGGER
        ]
        assert _named(validate_corpus(records), "draw negatives").ok


class TestZoneChanges:
    def _arrival(self, record_id: str, params: dict):
        return _resolution(
            record_id,
            payload=ResolutionPayload(events=(
                Event(type=EventType.ZONE_CHANGE, subjects=("E1",), params=params),
            )),
        )

    def test_the_from_zone_share_is_watched_by_default(self):
        records = _healthy() + [
            self._arrival(f"run.0-L1.{i}", {"to_zone": "graveyard"})
            for i in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "where the card came from")
        assert finding.watched
        assert finding.ok, "a watched number never fails a run"
        assert "carry from_zone" in finding.measured

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [
            self._arrival(f"run.0-L1.{i}", {"to_zone": "graveyard"})
            for i in range(900, 910)
        ]
        strict = Thresholds(min_zone_change_from_zone_rate=0.9)
        finding = _named(
            validate_corpus(records, strict), "where the card came from",
        )
        assert not finding.watched
        assert not finding.ok
        assert "floor 90.0%" in finding.measured

    def test_events_that_do_say_where_from_clear_the_floor(self):
        records = _healthy() + [
            self._arrival(
                f"run.0-L1.{i}",
                {"from_zone": "battlefield", "to_zone": "graveyard"},
            )
            for i in range(900, 910)
        ]
        strict = Thresholds(min_zone_change_from_zone_rate=0.9)
        assert _named(
            validate_corpus(records, strict), "where the card came from",
        ).ok

    def test_the_share_landing_on_the_stack_rides_along(self):
        records = _healthy() + [
            self._arrival(
                f"run.0-L1.{i}", {"from_zone": "hand", "to_zone": "stack"},
            )
            for i in range(900, 910)
        ]
        finding = _named(validate_corpus(records), "where the card came from")
        assert "10 say to_zone=stack" in finding.measured


class TestAttribution:
    def _with(self, record_id, attributed_to):
        return _resolution(
            record_id,
            payload=ResolutionPayload(events=(
                Event(
                    type=EventType.DAMAGE_DEALT, subjects=("E1",),
                    attributed_to=attributed_to,
                ),
            )),
        )

    def test_the_three_states_are_counted_apart_from_absent(self):
        records = _healthy() + [
            self._with("run.0-L1.900", "2"),
            self._with("run.0-L1.901", ATTRIBUTION_ROOT),
            self._with("run.0-L1.902", ATTRIBUTION_UNRESOLVED),
            self._with("run.0-L1.903", None),
        ]
        detail = " ".join(
            _named(validate_corpus(records), "name what produced them").detail
        )
        assert "sub_ability: 1" in detail
        assert "unresolved: 1" in detail
        assert "absent: 1" in detail

    def test_a_null_pointer_is_not_counted_as_the_root(self):
        """The ambiguity the sentinels end, asserted rather than assumed."""
        records = [self._with(f"run.0-L1.{i}", None) for i in range(900, 910)]
        finding = _named(validate_corpus(records), "name what produced them")
        assert "0/10 resolution events name a clause" in finding.measured

    def test_the_rate_is_watched_until_a_floor_is_given(self):
        records = [self._with(f"run.0-L1.{i}", None) for i in range(900, 910)]
        assert _named(validate_corpus(records), "name what produced them").watched
        strict = Thresholds(min_attributed_rate=0.9)
        judged = _named(
            validate_corpus(records, strict), "name what produced them",
        )
        assert not judged.watched
        assert not judged.ok


class TestProbeVisibility:
    def test_a_window_with_no_probes_says_so_without_failing(self):
        finding = _named(validate_corpus(_healthy()), "probe forks")
        assert finding.watched and finding.ok
        assert "0 damage-step probe forks" in finding.measured
        assert any("--probe-keywords" in line for line in finding.detail)

    def test_a_probe_fork_is_counted_and_its_keyword_named(self):
        records = _healthy() + [
            _combat(
                "run.0-L1.900", fork=True,
                payload=CombatPayload(
                    attackers=("E0",), probed_keyword="trample",
                    probed_entity="E0",
                ),
            ),
        ]
        finding = _named(validate_corpus(records), "probe forks")
        assert "1 damage-step probe forks" in finding.measured
        assert "trample 1" in finding.measured

    def test_an_intervention_is_counted_apart_from_a_probe(self):
        """The flag signatures differ, and so must the two counts."""
        records = _healthy() + [
            _resolution("run.0-L1.900", interventional=True, fork=True),
        ]
        finding = _named(validate_corpus(records), "probe forks")
        assert "0 damage-step probe forks, 1 interventional forks" in (
            finding.measured
        )


class TestAForksTurnAgreesWithItsMirror:
    """The residue of the game_id check, named where it can be acted on.

    A probe re-runs one damage step from the state the real combat was in, so
    the fork and the record it mirrors describe one moment. When the fork's
    snapshot lags, the game_id check sees a backward turn jump and blames a
    game merge that never happened — which is how 3.9% of the smoke corpus's
    probe forks were reported for a whole pass.
    """

    def _probe(self, record_id: str, mirror: str, turn: int):
        return _combat(
            record_id, fork=True, mirror_of=mirror, state=_snapshot(turn=turn),
            payload=CombatPayload(attackers=("E0",), probed_keyword="trample"),
        )

    def _real(self, record_id: str, turn: int):
        return _combat(record_id, state=_snapshot(turn=turn))

    def test_a_fork_at_its_mirrors_turn_holds(self):
        records = _healthy() + [
            self._real("run.0-L1.900", 6),
            self._probe("run.0-L1.901", "run.0-L1.900", 6),
        ]
        finding = _named(validate_corpus(records), "agrees with the record it mirrors")
        assert finding.ok, finding.detail
        assert "0/1 forks disagree" in finding.measured

    def test_a_stale_fork_snapshot_is_named_with_both_turns(self):
        records = _healthy() + [
            self._real("run.0-L1.900", 9),
            self._probe("run.0-L1.901", "run.0-L1.900", 8),
        ]
        finding = _named(validate_corpus(records), "agrees with the record it mirrors")
        assert not finding.ok, finding.measured
        assert "1/1 forks disagree" in finding.measured
        assert any(
            "run.0-L1.901 at turn 8 mirrors run.0-L1.900 at turn 9" in line
            for line in finding.detail
        )

    def test_a_fork_ahead_of_its_mirror_is_caught_too(self):
        """Only lag has been observed, but the invariant is agreement."""
        records = _healthy() + [
            self._real("run.0-L1.900", 6),
            self._probe("run.0-L1.901", "run.0-L1.900", 7),
        ]
        assert not _named(
            validate_corpus(records), "agrees with the record it mirrors"
        ).ok

    def test_a_mirror_outside_the_window_is_counted_apart_and_judged_not_at_all(
        self,
    ):
        """``--limit`` cuts the stream mid-game; the missing half is the
        reader's doing, not the collector's."""
        records = _healthy() + [self._probe("run.0-L1.901", "run.0-L1.900", 8)]
        finding = _named(validate_corpus(records), "agrees with the record it mirrors")
        assert finding.ok, finding.detail
        assert "0/0 forks disagree" in finding.measured
        assert "1 mirrors fell outside the window" in finding.measured

    def test_a_fork_written_before_the_record_it_mirrors_still_joins(self):
        """A shard may carry either half first, so the comparison waits for the
        end of the pass rather than for the next record."""
        records = _healthy() + [
            self._probe("run.0-L1.901", "run.0-L1.900", 8),
            self._real("run.0-L1.900", 9),
        ]
        assert not _named(
            validate_corpus(records), "agrees with the record it mirrors"
        ).ok


class TestForkEventsNameAClause:
    """Attribution on the records the fork collectors write, measured apart.

    The aggregate hid this: the observed records reached 84.8% naming a clause
    while every one of the 1,698 fork events in the same window said nothing,
    and the two averaged into a number that read as a channel warming up.
    """

    def _fork(self, record_id: str, attributed_to):
        return _resolution(
            record_id, interventional=True, fork=True,
            payload=ResolutionPayload(
                events=(
                    Event(
                        type=EventType.DAMAGE_DEALT, subjects=("E1",),
                        attributed_to=attributed_to,
                    ),
                ),
            ),
        )

    def test_a_window_with_no_forks_says_so_without_failing(self):
        finding = _named(validate_corpus(_healthy()), "fork records")
        assert finding.watched and finding.ok
        assert "0/0" in finding.measured

    def test_silent_fork_events_are_watched_by_default(self):
        records = _healthy() + [self._fork("run.0-L1.900", None)]
        finding = _named(validate_corpus(records), "fork records")
        assert finding.watched and finding.ok
        assert "0/1 events on fork records name a clause" in finding.measured
        assert "absent: 1" in finding.detail

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [self._fork("run.0-L1.900", None)]
        finding = _named(
            validate_corpus(records, Thresholds(min_fork_attributed_rate=0.9)),
            "fork records",
        )
        assert not finding.ok and not finding.watched

    def test_a_fork_naming_the_root_line_clears_the_floor(self):
        records = _healthy() + [self._fork("run.0-L1.900", ATTRIBUTION_ROOT)]
        assert _named(
            validate_corpus(records, Thresholds(min_fork_attributed_rate=0.9)),
            "fork records",
        ).ok

    def test_an_explicit_unresolved_counts_as_naming_something(self):
        """Saying the pointer did not land is a fact; saying nothing is not."""
        records = _healthy() + [self._fork("run.0-L1.900", ATTRIBUTION_UNRESOLVED)]
        finding = _named(validate_corpus(records), "fork records")
        assert "1/1 events on fork records name a clause" in finding.measured


class TestEventsThatDeclareACauseNameOne:
    """The identity channel the duplicate-event check leans on.

    Two attackers dealing 1 to the same player differ in nothing but their
    cause, so an empty one turns two real outcomes into one outcome written
    twice — the collector is right not to dedupe them and the duplicate check
    is right to complain, and only this number says which side to fix.
    """

    def _zone_change(self, record_id: str, cause):
        params = {"from_zone": "battlefield", "to_zone": "graveyard"}
        if cause is not None:
            params["cause"] = cause
        return _resolution(
            record_id,
            payload=ResolutionPayload(
                events=(
                    Event(
                        type=EventType.ZONE_CHANGE, subjects=("E1",),
                        params=params, attributed_to=ATTRIBUTION_ROOT,
                    ),
                ),
            ),
        )

    def test_a_missing_cause_is_watched_by_default(self):
        records = _healthy() + [self._zone_change("run.0-L1.900", None)]
        finding = _named(validate_corpus(records), "declare a cause")
        assert finding.watched and finding.ok
        # Four of the healthy window's trigger records carry a zone_change of
        # their own, and none of them names a cause either.
        assert "0/5 events" in finding.measured

    def test_a_populated_cause_is_counted(self):
        records = _healthy() + [self._zone_change("run.0-L1.900", "E7")]
        finding = _named(validate_corpus(records), "declare a cause")
        assert "1/5 events" in finding.measured
        assert "zone_change: 1/5" in finding.detail

    def test_a_floor_turns_the_measurement_into_a_verdict(self):
        records = _healthy() + [self._zone_change("run.0-L1.900", None)]
        finding = _named(
            validate_corpus(records, Thresholds(min_cause_rate=0.5)),
            "declare a cause",
        )
        assert not finding.ok and not finding.watched

    def test_a_type_whose_row_declares_no_cause_is_not_measured(self):
        """Every type *may* carry a cause and most correctly never do, so a
        rate over the whole vocabulary would read near zero and mean nothing.
        The healthy window's resolutions are damage_dealt, which declares
        ``source`` instead, so only its four zone_changes are counted."""
        finding = _named(validate_corpus(_healthy()), "declare a cause")
        assert "0/4 events" in finding.measured
        assert not any("damage_dealt" in line for line in finding.detail)


class TestNonKeywordKeysAreWatchedNotJudged:
    """Every trait kind is joined; only the keyword kind carries the verdict.

    A spell or static mismatch has a cause a keyword mismatch does not — a
    reconversion between collection and training, or a line the converter never
    emits at all — and nobody has measured what its healthy value is. Reported
    as its own number rather than folded into the keyword line's tail, because
    a number in a passing check's detail is a number nobody reads.
    """

    def _spell_record(self, index: int):
        return _trigger(
            "run.0-L1.900", ability=(ProvenanceKey(_BEARS, 0, "spell", index),),
        )

    def test_a_spell_mismatch_is_reported_without_failing_the_run(self, tmp_path):
        sidecars = _write_bears_sidecar(tmp_path, (0,))
        findings = validate_corpus(
            [*_healthy(), self._spell_record(3)], None, sidecars,
        )
        finding = _named(findings, "non-keyword provenance keys")
        assert finding.watched and finding.ok
        assert "keys of every other trait kind fail to join" in finding.measured
        # Both spell keys in the window: the planted mismatch, and the healthy
        # window's own line, whose card has no sidecar in this tree.
        assert "spell: 2/2 unjoinable" in finding.detail
        assert _named(findings, "keyword provenance keys").ok

    def test_the_keyword_line_reports_only_keyword_keys(self):
        """It reported both tallies in one sentence, and the one that could
        fail the run was the one an operator stopped reading."""
        finding = _named(validate_corpus(_healthy()), "keyword provenance keys")
        assert "over all" not in finding.measured

    def test_both_findings_say_so_when_no_tree_was_readable(self):
        findings = validate_corpus(_healthy(), None, None)
        for fragment in ("keyword provenance keys", "non-keyword provenance keys"):
            finding = _named(findings, fragment)
            assert finding.watched and finding.ok
            assert "none checked" in finding.measured


@pytest.mark.parametrize(
    "fragment",
    [
        "record_id is unique",
        "one game",
        "link_id",
        "trigger and rewrite",
        "resolution records name",
        "empty ability says why",
        "keyword provenance keys",
        "patched mode",
        "outcome",
        "cost fields",
        "tier depth",
        "exact duplicates",
        "repeats an event",
        "draw negatives",
        "where the card came from",
        "name what produced them",
        "fork records",
        "declare a cause",
        "agrees with the record it mirrors",
        "non-keyword provenance keys",
        "probe forks",
    ],
)
def test_every_named_invariant_is_reported(fragment):
    """The list is the contract; a check silently dropped is a check that lies."""
    assert _named(validate_corpus(_healthy()), fragment)
