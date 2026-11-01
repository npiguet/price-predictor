"""The dropped-event table a run under ``--count-dropped-events`` prints (FR-041).

Each worker lifetime appends its games' counts beside its shard, one
``event_type<TAB>phase<TAB>count`` line per pair and game; the supervisor sums
every lifetime's file of its run when the run ends and prints the table. Both
collecting supervisors do: ``sealed match-outcomes`` and the two ``effects``
collectors.
"""

from __future__ import annotations

from pathlib import Path

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.collector_connector import CollectorSupervisor
from sealed.application.match_outcomes import MatchOutcomeSupervisor
from sealed.infrastructure.effect_collection_flags import (
    DROPPED_EVENTS_SUFFIX,
    dropped_events_report,
    sum_dropped_events,
)


def _lifetime(directory: Path, run_id: str, shard: str, *lines: str) -> Path:
    """One worker lifetime's counts file, as the JVM appends it."""
    path = directory / f"{run_id}.{shard}{DROPPED_EVENTS_SUFFIX}"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


class TestSummingAcrossWorkers:
    def test_every_lifetime_and_every_game_is_summed(self, tmp_path):
        _lifetime(tmp_path, "run", "0-aaa", "zone_change\tmain1\t2", "zone_change\tmain1\t1")
        _lifetime(tmp_path, "run", "1-bbb", "zone_change\tmain1\t4", "mana_produced\tupkeep\t1")

        counts, lifetimes = sum_dropped_events(tmp_path, "run")

        assert lifetimes == 2
        assert counts == {("zone_change", "main1"): 7, ("mana_produced", "upkeep"): 1}

    def test_another_runs_files_are_not_summed(self, tmp_path):
        _lifetime(tmp_path, "run", "0-aaa", "zone_change\tmain1\t2")
        _lifetime(tmp_path, "other", "0-ccc", "zone_change\tmain1\t50")

        counts, lifetimes = sum_dropped_events(tmp_path, "run")

        assert (counts, lifetimes) == ({("zone_change", "main1"): 2}, 1)

    def test_a_line_a_killed_worker_cut_short_is_skipped(self, tmp_path):
        """Workers are killed, not stopped, so a file can end mid-line."""
        path = _lifetime(tmp_path, "run", "0-aaa", "zone_change\tmain1\t2")
        with path.open("a", encoding="utf-8") as tail:
            tail.write("damage_dealt\tcomb")

        counts, _ = sum_dropped_events(tmp_path, "run")

        assert counts == {("zone_change", "main1"): 2}

    def test_shards_and_logs_beside_the_files_are_not_read(self, tmp_path):
        _lifetime(tmp_path, "run", "0-aaa", "zone_change\tmain1\t2")
        (tmp_path / "run.0-aaa.jsonl.gz").write_bytes(b"\x1f\x8b")
        (tmp_path / "run.0.log").write_text("zone_change\tmain1\t9\n", encoding="utf-8")

        assert sum_dropped_events(tmp_path, "run") == ({("zone_change", "main1"): 2}, 1)


class TestTheTable:
    def test_it_lists_each_type_and_phase_most_dropped_first_with_a_total(self, tmp_path):
        _lifetime(tmp_path, "run", "0-aaa", "mana_produced\tupkeep\t1", "zone_change\tmain1\t3")

        lines = dropped_events_report(tmp_path, "run").splitlines()

        assert lines[0] == "Dropped events (no record open), summed over 1 worker lifetime:"
        assert lines[1].split() == ["event", "type", "phase", "count"]
        assert lines[2].split() == ["zone_change", "main1", "3"]
        assert lines[3].split() == ["mana_produced", "upkeep", "1"]
        assert lines[4].split() == ["total", "4"]

    def test_a_run_that_dropped_nothing_says_so(self, tmp_path):
        assert dropped_events_report(tmp_path, "run") == (
            "Dropped events: none counted over 0 worker lifetimes"
        )


class TestMatchOutcomesPrintsItWhenTheRunEnds:
    def _supervisor(self, tmp_path, count: str) -> MatchOutcomeSupervisor:
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "match-outcomes.txt", best_of=1,
            effect_records_dir=tmp_path,
            collection_caps={"effect.count.dropped.events": count},
        )

        def played() -> None:
            _lifetime(tmp_path, supervisor.run_id, "0-aaa", "zone_change\tmain1\t5")

        supervisor._pool.run = played
        return supervisor

    def test_a_counting_run_prints_the_table(self, tmp_path, capsys):
        self._supervisor(tmp_path, "true").run()

        out = capsys.readouterr().out
        assert "Dropped events (no record open), summed over 1 worker lifetime:" in out
        assert "zone_change" in out

    def test_a_run_that_does_not_count_prints_nothing(self, tmp_path, capsys):
        self._supervisor(tmp_path, "false").run()

        assert "Dropped events" not in capsys.readouterr().out


class TestTheCollectorsPrintItWhenTheRunEnds:
    def _supervisor(self, tmp_path, *, count: bool) -> CollectorSupervisor:
        return CollectorSupervisor(
            worker_count=1, effect_records=tmp_path, run_id="run",
            caps=CollectionCaps(count_dropped_events=count),
        )

    def test_a_counting_run_prints_the_table_summed_over_its_rounds(self, tmp_path, capsys):
        supervisor = self._supervisor(tmp_path, count=True)
        # Two rounds' workers under the one run id.
        _lifetime(tmp_path, "run", "0-aaa", "zone_change\tmain1\t2")
        _lifetime(tmp_path, "run", "0-bbb", "zone_change\tmain1\t3")

        supervisor.stop()

        out = capsys.readouterr().out
        assert "summed over 2 worker lifetimes" in out
        assert any(line.split() == ["zone_change", "main1", "5"] for line in out.splitlines())

    def test_a_run_that_does_not_count_prints_nothing(self, tmp_path, capsys):
        _lifetime(tmp_path, "run", "0-aaa", "zone_change\tmain1\t2")

        self._supervisor(tmp_path, count=False).stop()

        assert "Dropped events" not in capsys.readouterr().out
