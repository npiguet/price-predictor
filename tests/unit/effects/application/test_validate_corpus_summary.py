"""validate-corpus's report as a whole: a healthy window passes, the summary
lines, record ids, the window, every named check, the command's directory
pass and its CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from effects.application.corpus_validation.findings import Finding, Findings
from effects.application.validate_corpus import validate_corpus, validate_directory
from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import ResolutionPayload
from effects.infrastructure.shard_io import NoShardsError, read_window
from tests.unit.effects.application.validate_corpus_support import (
    _BEARS,
    _healthy,
    _named,
    _resolution,
    _snapshot,
    _trigger,
    _write_bears_sidecar,
)
from tests.unit.effects.builders import RawShardWriter


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


class TestTheSummary:
    """The closing lines count judged invariants only: a watched one cannot fail."""

    def test_a_clean_pass_says_every_judged_invariant_holds(self):
        findings = Findings([
            Finding(name="a", ok=True, measured="1"),
            Finding(name="b", ok=True, measured="2", watched=True),
        ])
        assert findings.broken() == []
        assert findings.summary() == [
            "all 1 judged invariants hold (1 watched, not judged)",
        ]

    def test_a_breach_names_each_broken_invariant(self):
        findings = Findings([
            Finding(name="a", ok=False, measured="1"),
            Finding(name="b", ok=True, measured="2"),
        ])
        assert findings.summary() == ["1 of 2 judged invariants broken:", "  a"]


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

    def test_two_lifetimes_that_namespace_their_ids_pass(self, tmp_path):
        """The same restart, with the fix in place, is not a defect."""
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        with RawShardWriter(tmp_path, "run", 0, "L2") as writer:
            writer.write(_resolution("run.0-L2.0", game_id="run.0-L2.0"))
        assert _named(validate_corpus(read_window(tmp_path)), "record_id").ok


class TestValidateDirectory:
    """The command's whole pass: the shards listed once, then read."""

    def test_a_directory_without_shards_is_refused(self, tmp_path):
        with pytest.raises(NoShardsError, match="no shards under"):
            validate_directory(tmp_path)

    def test_it_reports_the_shards_it_read_and_their_findings(self, tmp_path):
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        with RawShardWriter(tmp_path / "deeper", "run", 1, "L1") as writer:
            writer.write(_resolution("run.1-L1.0", game_id="run.1-L1.0"))

        report = validate_directory(tmp_path)

        assert report.shards == 2
        assert _named(report.findings, "record_id is unique").ok
        # The per-shard check runs over both shards, the nested one included.
        envelope = _named(report.findings, "what_if is on every blockers record")
        assert envelope.ok, envelope.detail
        assert envelope.measured.startswith("2 shard(s): 0 with")


class TestTheWindow:
    def test_an_empty_window_reports_rather_than_raising(self):
        findings = validate_corpus([])
        assert all(finding.ok for finding in findings)
        assert _named(findings, "record_id is unique").measured.startswith("0 repeats")


class TestTheCli:
    """The subcommand exits non-zero on a broken window and zero on a good one."""

    def _run(self, directory, **overrides) -> int:
        from tests.unit.effects.cli_args import parse_cli

        args = parse_cli(["validate-corpus", "--effect-records", str(directory)])
        for name, value in overrides.items():
            setattr(args, name, value)
        return args.func(args)

    def test_a_clean_directory_exits_zero(self, tmp_path, capsys):
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert self._run(tmp_path) == 0
        out = capsys.readouterr().out
        assert "invariants hold" in out
        # A watched figure is printed apart and fails nothing: no probe fork.
        assert "watched, not judged" in out
        assert "[WATCH] probe forks were taken: 0 damage-step probe forks" in out

    def test_a_planted_collision_exits_non_zero_and_names_it(self, tmp_path, capsys):
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        with RawShardWriter(tmp_path, "run", 0, "L2") as writer:
            writer.write(_resolution("run.0-L1.0", game_id="run.0-L1.0"))
        assert self._run(tmp_path) == 1
        out = capsys.readouterr().out
        assert "[FAIL] record_id is unique across the run's shards" in out
        assert "invariants broken" in out

    def test_an_undecodable_line_is_a_failed_finding_and_the_rest_is_read(
            self, tmp_path, capsys):
        """V5 #20: one line that does not decode is the shard's finding, not a
        stop: the other shards and the lines before it are still measured."""
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        bad = tmp_path / "run.1-L1.jsonl"
        bad.write_text('{"record_id": "x", "generation": 3, "kind": "nonesuch"}\n',
                       encoding="utf-8")
        assert self._run(tmp_path) == 1
        out = capsys.readouterr().out
        assert "[FAIL] every shard line decodes" in out
        assert "1/2 shards" in out and "run.1-L1.jsonl, line 1" in out
        assert "[PASS] record_id is unique" in out

    def test_values_outside_the_closed_lists_are_watched_figures(self, tmp_path, capsys):
        """The decode tally the read keeps (WP9) is reported, never judged."""
        from effects.infrastructure.record_codec import record_to_dict

        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        raw = record_to_dict(_resolution("run.0-L1.950"))
        raw.pop("mode")
        (tmp_path / "run.1-L1.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
        self._run(tmp_path)
        out = capsys.readouterr().out
        assert "[WATCH] values outside the wire's closed lists: 1" in out

    def test_variant_scripts_names_the_variant_tree_for_the_key_joins(self, tmp_path):
        from effects.infrastructure.cli import build_parser

        args = build_parser().parse_args([
            "validate-corpus", "--effect-records", str(tmp_path),
            "--variant-scripts", str(tmp_path / "variants")])
        assert args.variant_scripts == tmp_path / "variants"

    def test_an_empty_directory_exits_non_zero(self, tmp_path, caplog):
        """Refused through exits_with, as every command's refusal is."""
        assert self._run(tmp_path) == 1
        assert "no shards" in caplog.text

    def test_the_new_thresholds_are_the_contracts(self):
        """Pinned here because a default that drifts changes what a corpus means."""
        from tests.unit.effects.cli_args import parse_cli

        args = parse_cli(["validate-corpus"])
        assert args.max_duplicate_event_rate == 0.02
        assert args.max_trigger_fired_share == 0.65
        assert args.min_zone_change_from_zone_rate is None
        assert args.min_attributed_rate is None
        assert args.cards_folders == (
            Path("output/effects/gen3/converted/cardsfolder"),
            Path("output/effects/gen3/converted/tokenscripts"),
        )

    def test_a_floor_passed_on_the_command_line_turns_a_watch_into_a_failure(
        self, tmp_path, capsys,
    ):
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
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
        with RawShardWriter(shards, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
            writer.write(
                _trigger(
                    "run.0-L1.900", state=_snapshot(turn=7),
                    ability=(ProvenanceKey(_BEARS, 0, "keyword", 1),),
                )
            )
        assert self._run(
            shards, cards_folders=(tmp_path / "cardsfolder",),
        ) == 1
        out = capsys.readouterr().out
        assert "[FAIL] keyword provenance keys join their sidecar" in out

    def test_the_same_directory_with_matching_ordinals_exits_zero(
        self, tmp_path, capsys,
    ):
        _write_bears_sidecar(tmp_path, (0, 2))
        shards = tmp_path / "records"
        with RawShardWriter(shards, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
            writer.write(
                _trigger(
                    "run.0-L1.900", state=_snapshot(turn=7),
                    ability=(ProvenanceKey(_BEARS, 0, "keyword", 2),),
                )
            )
        assert self._run(
            shards, cards_folders=(tmp_path / "cardsfolder",),
        ) == 0
        assert "[PASS] keyword provenance keys join their sidecar" in (
            capsys.readouterr().out
        )

    def test_a_missing_converted_tree_reports_unchecked_rather_than_failing(
        self, tmp_path, capsys,
    ):
        """No tree on this machine is not evidence that the corpus joins."""
        with RawShardWriter(tmp_path, "run", 0, "L1") as writer:
            for record in _healthy():
                writer.write(record)
        assert self._run(
            tmp_path, cards_folders=(tmp_path / "not-here",),
        ) == 0
        assert "[WATCH] keyword provenance keys join their sidecar" in (
            capsys.readouterr().out
        )


#: Every check validate-corpus runs, in its report order: the list is the
#: contract, so a check silently dropped or renamed fails here.
_EVERY_CHECK = (
    "record_id is unique across the run's shards",
    'each game_id names one game',
    'games end with their game-end marker',
    'closed games end with their game-end marker',
    'records the per-game cap withheld',
    'every record kind a game writes is present',
    ("every link_id joins one activation to its resolution(s): one, or one per chosen "
     "mode of a charm, or none for an activation whose item never resolved"),
    'every cost half is met by its effect half unless its item never resolved',
    'trigger and rewrite records name an acting line',
    'resolution records name an acting line',
    'continuous records name an acting line',
    'every empty ability says why it is empty',
    'keyword provenance keys join their sidecar',
    'non-keyword provenance keys join their sidecar',
    'provenance keys outside every declared range',
    "a copy keys its keyword lines into the copied card's file",
    'every record was collected in patched mode',
    'outcome takes more than one value',
    'cost fields are not all empty',
    'snapshot tier depth is uniform across kinds',
    'exact duplicates within a game stay rare, per kind',
    'no record repeats an event inside itself',
    'one-event-per-unit types repeat inside a record',
    'trigger records draw negatives against positives',
    'trigger records hold positives',
    'rewrite records say which replacement result happened',
    "a rewrite's outgoing event is not a copy of its incoming one",
    'outgoing is null on the results that rewrote nothing',
    'rewrites name the ability that ran instead',
    'zone_change events say where the card came from',
    'resolution events name what produced them',
    "fork records' events name what produced them",
    'events that declare a cause name one',
    "a fork's turn agrees with the record it mirrors",
    'probe forks were taken',
    'fork records whose copy left something out',
    'mana_produced events name the mana they made',
    'declared event types are observed in the window',
)


def test_every_named_invariant_is_reported():
    """The list is the contract; a check silently dropped is a check that lies."""
    assert tuple(f.name for f in validate_corpus(_healthy())) == _EVERY_CHECK
