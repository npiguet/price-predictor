"""Unit tests for MatchOutcomeSupervisor."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from sealed.application.match_outcomes import MatchOutcomeSupervisor
from tests.unit.sealed.conftest import FakeProcess


class TestSupervisorSpawnCount:
    def test_spawns_correct_number_of_workers(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        spawned = []
        lock = threading.Lock()

        def fake_start_worker(worker_id):
            with lock:
                spawned.append(worker_id)
                if len(spawned) >= 3:
                    supervisor._shutdown_event.set()
            return FakeProcess(pid=1000 + worker_id, hang=True)

        supervisor = MatchOutcomeSupervisor(worker_count=3, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert len(spawned) == 3

class TestSupervisorCrashRestartBehavior:
    def test_crashed_worker_is_restarted(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        start_count = [0]
        call_number = [0]

        def fake_start_worker(worker_id):
            start_count[0] += 1
            call_number[0] += 1
            n = call_number[0]
            if n == 1:
                # First call: crash immediately (non-zero exit)
                return FakeProcess(pid=1000, returncode=1)
            else:
                # Second call: trigger shutdown after a moment
                proc = FakeProcess(pid=1001, hang=True)
                # Signal shutdown after worker starts
                threading.Timer(0.05, lambda: supervisor._shutdown_event.set()).start()
                return proc

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert start_count[0] >= 2, "Crashed worker must be restarted at least once"

    def test_worker_not_restarted_after_shutdown(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        start_count = [0]

        def fake_start_worker(worker_id):
            start_count[0] += 1
            # Signal shutdown when worker starts
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert start_count[0] == 1, "Worker must not be restarted after shutdown"


class TestSupervisorShutdownSignal:
    def test_shutdown_event_stops_run(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"

        def fake_start_worker(worker_id):
            proc = FakeProcess(pid=1000, hang=True)
            # Trigger shutdown shortly after starting
            threading.Timer(0.05, lambda: supervisor._shutdown_event.set()).start()
            return proc

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker

        start = time.monotonic()
        supervisor.run()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, "run() should return quickly after shutdown event"

    def test_all_workers_terminated_on_shutdown(self, tmp_path, killed_process_trees):
        output_file = tmp_path / "match-outcomes.txt"
        processes = []

        def fake_start_worker(worker_id):
            proc = FakeProcess(pid=1000 + worker_id, hang=True)
            processes.append(proc)
            if len(processes) == 2:
                threading.Timer(0.05, lambda: supervisor._shutdown_event.set()).start()
            return proc

        supervisor = MatchOutcomeSupervisor(worker_count=2, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert len(processes) == 2
        for proc in processes:
            assert proc in killed_process_trees, "every worker's tree is killed on shutdown"


class TestSupervisorWorkerCount:
    """US2: --workers argument controls exact number of spawned processes."""

    def test_single_worker_spawns_exactly_one(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        spawned_ids = []

        def fake_start_worker(worker_id):
            spawned_ids.append(worker_id)
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert len(spawned_ids) == 1


class TestSupervisorRecycleOldest:
    def test_oldest_worker_is_terminated(self, tmp_path, killed_process_trees):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=2, output_path=output_file, best_of=3)

        old_proc = FakeProcess(pid=100, hang=True)
        new_proc = FakeProcess(pid=200, hang=True)

        with supervisor._processes_lock:
            supervisor._processes.extend([old_proc, new_proc])
            supervisor._start_times[old_proc] = 1000.0
            supervisor._start_times[new_proc] = 2000.0

        supervisor._kill_oldest_worker()

        assert killed_process_trees == [old_proc], "only the oldest worker's tree is killed"

class TestSupervisorStatusReporting:
    def test_output_dir_created_on_run(self, tmp_path):
        output_file = tmp_path / "sealed" / "match-outcomes.txt"

        def fake_start_worker(worker_id):
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert output_file.parent.exists(), "Output directory must be created"


class TestSupervisorSideDecks:
    """side_a_decks_path / side_b_decks_path / side_b_decks_weight are
    stored on the supervisor and forwarded to MatchWorkerConnector.start()."""

    def test_side_paths_forwarded_to_connector(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        side_a = tmp_path / "side-a.txt"
        side_b = tmp_path / "side-b.txt"

        supervisor = MatchOutcomeSupervisor(
            worker_count=1,
            output_path=output_file,
            best_of=3,
            side_a_decks_path=side_a,
            side_b_decks_path=side_b,
            side_b_decks_weight=6,
        )

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        proc = supervisor._start_worker(0)

        kwargs = fake_connector.start.call_args.kwargs
        assert kwargs.get("side_a_decks_path") == side_a
        assert kwargs.get("side_b_decks_path") == side_b
        assert kwargs.get("side_b_decks_weight") == 6
        assert proc.returncode == 0

class TestSupervisorRunId:
    """Supervisor generates a UUID run_id at construction and forwards the label."""

    def test_run_id_is_generated_at_construction(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)
        # uuid4 strings are 36 chars (8-4-4-4-12) with dashes
        assert len(supervisor.run_id) == 36
        assert supervisor.run_id.count("-") == 4

    def test_run_id_stable_across_worker_restarts(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        supervisor._start_worker(0)
        supervisor._start_worker(1)
        supervisor._start_worker(2)

        run_ids = [call.kwargs["run_id"] for call in fake_connector.start.call_args_list]
        assert len(set(run_ids)) == 1, "run_id must be identical across worker starts"
        assert run_ids[0] == supervisor.run_id

    def test_a_respawned_slot_is_started_with_the_same_arguments(self, tmp_path):
        """Nothing the supervisor passes distinguishes two lifetimes of one slot.

        The pool recycles the longest-running worker every status interval and
        restarts crashed ones, always under the same run id and the same slot
        index. This pins that, because it is the reason the effect-record shard
        writer has to mint a per-JVM lifetime token of its own: the counters it
        stamps into record and game ids restart at zero with the JVM, and the
        run id and slot index cannot tell the two runs of zero apart.
        """
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=output_file, best_of=3,
            effect_records_dir=tmp_path / "records",
        )

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        supervisor._start_worker(0)
        supervisor._start_worker(0)  # the same slot, a new JVM

        first, second = fake_connector.start.call_args_list
        assert first.kwargs == second.kwargs
        assert first.args == second.args
        assert first.kwargs["worker_index"] == 0

    def test_run_ids_differ_between_supervisor_instances(self, tmp_path):
        out1 = tmp_path / "out1.txt"
        out2 = tmp_path / "out2.txt"
        s1 = MatchOutcomeSupervisor(worker_count=1, output_path=out1, best_of=3)
        s2 = MatchOutcomeSupervisor(worker_count=1, output_path=out2, best_of=3)
        assert s1.run_id != s2.run_id

class TestSupervisorWorkerLogs:
    """F4: a worker can only trip one of the five latched effect-record
    failure reporters while effect-record collection is on, so that is the
    one case ``_start_worker`` gives it a real log destination instead of
    DEVNULL -- a plain match-outcomes run keeps discarding everything, per
    the deliberate I/O-avoidance note on ``_start_worker`` itself."""

    def test_no_log_file_without_effect_records_dir(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file, best_of=3)

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        supervisor._start_worker(0)

        assert fake_connector.start.call_args.kwargs.get("log_file") is None

    def test_a_real_log_file_when_effect_records_dir_is_set(self, tmp_path):
        records = tmp_path / "records"
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=output_file, best_of=3,
            effect_records_dir=records,
        )

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        supervisor._start_worker(0)

        log_file = fake_connector.start.call_args.kwargs.get("log_file")
        assert log_file is not None
        assert log_file.closed is False
        assert (records / "logs" / f"{supervisor.run_id}.0.log").exists()
        assert not list(records.glob("*.log")), "never beside the shards"

    def test_a_respawned_slot_reuses_the_same_log_file(self, tmp_path):
        records = tmp_path / "records"
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=output_file, best_of=3,
            effect_records_dir=records,
        )

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        supervisor._start_worker(0)
        supervisor._start_worker(0)  # same slot, a new JVM

        first, second = fake_connector.start.call_args_list
        assert first.kwargs["log_file"] is second.kwargs["log_file"]

    def test_run_closes_the_worker_logs_when_the_pool_finishes(self, tmp_path):
        records = tmp_path / "records"
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=output_file, best_of=3,
            effect_records_dir=records,
        )
        opened: list = []

        def fake_start_worker(worker_id):
            opened.append(supervisor._worker_logs.get(worker_id))
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert opened and all(handle.closed for handle in opened)


class TestSupervisorBestOf:
    """Supervisor stores best_of and forwards it to every worker start."""

    def test_best_of_forwarded_to_connector(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=output_file, best_of=17,
        )

        fake_connector = MagicMock()
        fake_connector.start.return_value = FakeProcess(pid=1000, returncode=0)
        supervisor._connector = fake_connector

        supervisor._start_worker(0)

        kwargs = fake_connector.start.call_args.kwargs
        assert kwargs.get("best_of") == 17


class TestSupervisorStatusLineFormat:
    """The status interval is also the recycle of a run without the stop
    protocol: every interval the pool kills the longest-running worker, which
    bounds how long a stuck match holds a slot (``_kill_oldest_worker`` says not
    to lengthen it). The line's exact shape is pinned on the pool
    (``test_forge_worker_pool.py``)."""

    def test_status_interval_still_sixty_seconds(self, tmp_path):
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
        )
        assert supervisor.STATUS_INTERVAL == 60


class TestTheRunReportsWhatItsWorkersLogged:
    """V4 #3 on match-outcomes: worker failure lines went to the logs and
    nowhere else, and a run that recorded nothing said nothing."""

    def _run(self, tmp_path, *, matches: int, record_lines: int, log_lines=(),
             random_seat_matches: int = 0):
        records = tmp_path / "records"
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=output_file, best_of=3,
            effect_records_dir=records,
        )

        def fake_start_worker(worker_id):
            log = supervisor._worker_logs.get(worker_id)
            log.write("".join(f"{line}\n" for line in log_lines).encode("utf-8"))
            log.flush()
            with open(output_file, "a", encoding="utf-8") as out:
                out.write("row\n" * matches)
            # Every match played to its end, a random-seat one (no row) included.
            with open(supervisor._progress_path, "a", encoding="utf-8") as progress:
                progress.write("match\n" * (matches + random_seat_matches))
            if record_lines:
                shard = records / f"{supervisor.run_id}.0-abc.jsonl"
                shard.write_text("{}\n" * record_lines, encoding="utf-8")
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor._start_worker = fake_start_worker
        supervisor.run()
        return supervisor

    def test_the_failure_lines_are_printed_by_kind(self, tmp_path, capsys):
        supervisor = self._run(
            tmp_path, matches=1, record_lines=3,
            log_lines=["Worker: 10 matches generated", "Error generating match: boom"],
        )

        assert supervisor.worker_failures.as_dict() == {"match_error": 1}
        assert "Worker logs over the run: 1 match_error" in capsys.readouterr().out

    def test_a_run_that_played_matches_and_recorded_nothing_collected_nothing(self, tmp_path):
        assert self._run(tmp_path, matches=2, record_lines=0).collected_nothing is True

    def test_records_of_this_run_are_something(self, tmp_path):
        assert self._run(tmp_path, matches=2, record_lines=3).collected_nothing is False

    def test_no_match_played_is_no_verdict(self, tmp_path):
        assert self._run(tmp_path, matches=0, record_lines=0).collected_nothing is False

    def test_a_run_of_random_seat_matches_alone_is_judged_too(self, tmp_path):
        """A random-seat match writes no outcome row (WP7): such a run read
        as no match played, whatever it recorded."""
        supervisor = self._run(tmp_path, matches=0, record_lines=0, random_seat_matches=3)
        assert supervisor.collected_nothing is True

    def test_random_seat_matches_that_threw_are_matches_played(self, tmp_path):
        supervisor = self._run(tmp_path, matches=0, record_lines=0, log_lines=[
            "Error generating match: [random-seat match] boom",
        ])
        assert supervisor.worker_failures.as_dict() == {"random_seat_match_error": 1}
        assert supervisor.collected_nothing is True

    def test_the_run_s_stops_and_matches_reach_its_manifest(self, tmp_path):
        import json

        from price_predictor.infrastructure.run_manifest import run_recorded

        def body():
            self._run(tmp_path, matches=1, record_lines=2, random_seat_matches=2)
            return 0

        run_recorded(tmp_path / "records", body, command="python -m sealed match-outcomes")

        entry = json.loads(
            (tmp_path / "records" / "run.json").read_text(encoding="utf-8"))["invocations"][0]
        assert (entry["matches"], entry["matches_played"]) == (1, 3)
        assert (entry["stop_timeout"], entry["recycle_seconds"]) == (600, 600)
        assert entry["worker_stops"]["survived"] == 0
        assert entry["hook_failures"] == {"connector": {}, "forge": {}}
        assert entry["fork_refusals"] == {}

    def test_the_fork_refusal_tally_is_printed_and_kept_and_is_no_failure(
        self, tmp_path, capsys,
    ):
        """WP5: the worker prints how many forks declined, by kind and reason;
        expected, so they are counted apart from every failure line and leave
        the run's verdict alone."""
        import json

        from price_predictor.infrastructure.run_manifest import run_recorded

        tally = "Effect records: fork refusals so far, expected and no failure: {{{body}}}"
        found = {}

        def body():
            found["supervisor"] = self._run(tmp_path, matches=1, record_lines=2, log_lines=[
                tally.format(body="directed dies=1"),
                tally.format(body="directed dies=3, intervention no choices=2"),
            ])
            return 0

        run_recorded(tmp_path / "records", body, command="python -m sealed match-outcomes")

        supervisor = found["supervisor"]
        assert not supervisor.worker_failures
        assert supervisor.collected_nothing is False
        assert ("Fork refusals over the run (expected, no failure): "
                "3 directed dies, 2 intervention no choices") in capsys.readouterr().out
        entry = json.loads(
            (tmp_path / "records" / "run.json").read_text(encoding="utf-8"))["invocations"][0]
        assert entry["fork_refusals"] == {"directed dies": 3, "intervention no choices": 2}
        assert entry["worker_failures"] == {}

    def test_with_effect_records_each_worker_gets_the_stop_protocol(self, tmp_path):
        from unittest.mock import MagicMock

        from price_predictor.infrastructure.forge_jvm import WorkerStopSettings

        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
            effect_records_dir=tmp_path / "records",
            worker_stop=WorkerStopSettings(timeout=30, recycle_seconds=900),
        )
        supervisor._connector = MagicMock()
        supervisor._connector.start.return_value = FakeProcess(pid=1000)

        supervisor._start_worker(1)

        kwargs = supervisor._connector.start.call_args.kwargs
        logs = tmp_path / "records" / "logs"
        assert kwargs["stop_file"] == logs / f"{supervisor.run_id}.1.stop"
        assert kwargs["recycle_seconds"] == 900
        assert kwargs["progress_file"] == logs / f"{supervisor.run_id}.progress.txt"
        assert supervisor._pool._graceful.settings.timeout == 30

    def test_without_effect_records_the_pool_kills_as_before(self, tmp_path):
        from unittest.mock import MagicMock

        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
        )
        supervisor._connector = MagicMock()
        supervisor._connector.start.return_value = FakeProcess(pid=1000)

        supervisor._start_worker(0)

        kwargs = supervisor._connector.start.call_args.kwargs
        assert (kwargs["stop_file"], kwargs["recycle_seconds"], kwargs["progress_file"]) == (
            None, None, None)
        assert supervisor._pool._graceful is None

    def test_the_pool_scans_the_logs_at_every_status_line(self, tmp_path):
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
            effect_records_dir=tmp_path / "records",
        )
        assert supervisor._pool._on_status is not None

    def test_without_effect_records_nothing_is_scanned(self, tmp_path):
        supervisor = MatchOutcomeSupervisor(
            worker_count=1, output_path=tmp_path / "out.txt", best_of=3,
        )
        assert supervisor._pool._on_status is None
        assert supervisor.collected_nothing is False
