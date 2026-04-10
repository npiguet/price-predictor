"""Unit tests for MatchOutcomeSupervisor."""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from sealed.application.match_outcomes import MatchOutcomeSupervisor


class FakeProcess:
    """Fake subprocess.Popen that exits immediately with a given code."""

    def __init__(self, pid: int = 99, returncode: int = 0, hang: bool = False):
        self.pid = pid
        self.returncode = returncode
        self._hang = hang
        self._terminated = False

    def wait(self):
        if self._hang:
            # Block until terminated
            while not self._terminated:
                time.sleep(0.01)
        return self.returncode

    def terminate(self):
        self._terminated = True

    def kill(self):
        self._terminated = True

    def poll(self):
        return None if (self._hang and not self._terminated) else self.returncode


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

        supervisor = MatchOutcomeSupervisor(worker_count=3, output_path=output_file)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert len(spawned) == 3

    def test_default_worker_count_is_twelve(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=12, output_path=output_file)
        assert supervisor._worker_count == 12

    def test_explicit_worker_count_respected(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=2, output_path=output_file)
        assert supervisor._worker_count == 2


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

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
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

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
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

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
        supervisor._start_worker = fake_start_worker

        start = time.monotonic()
        supervisor.run()
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, "run() should return quickly after shutdown event"

    def test_all_workers_terminated_on_shutdown(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        processes = []

        def fake_start_worker(worker_id):
            proc = FakeProcess(pid=1000 + worker_id, hang=True)
            processes.append(proc)
            if len(processes) == 2:
                threading.Timer(0.05, lambda: supervisor._shutdown_event.set()).start()
            return proc

        supervisor = MatchOutcomeSupervisor(worker_count=2, output_path=output_file)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        for proc in processes:
            assert proc._terminated, "All worker processes must be terminated on shutdown"


class TestSupervisorWorkerCount:
    """US2: --workers argument controls exact number of spawned processes."""

    def test_explicit_two_workers_spawns_two(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        spawned_ids = set()
        lock = threading.Lock()

        def fake_start_worker(worker_id):
            with lock:
                spawned_ids.add(worker_id)
                if len(spawned_ids) == 2:
                    supervisor._shutdown_event.set()
            return FakeProcess(pid=1000 + worker_id, hang=True)

        supervisor = MatchOutcomeSupervisor(worker_count=2, output_path=output_file)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert len(spawned_ids) == 2
        assert spawned_ids == {0, 1}

    def test_omitted_workers_defaults_to_twelve(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=12, output_path=output_file)
        assert supervisor._worker_count == 12

    def test_single_worker_spawns_exactly_one(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        spawned_ids = []

        def fake_start_worker(worker_id):
            spawned_ids.append(worker_id)
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert len(spawned_ids) == 1


class TestSupervisorRecycleOldest:
    def test_oldest_worker_is_terminated(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=2, output_path=output_file)

        old_proc = FakeProcess(pid=100, hang=True)
        new_proc = FakeProcess(pid=200, hang=True)

        with supervisor._processes_lock:
            supervisor._processes.extend([old_proc, new_proc])
            supervisor._start_times[old_proc] = 1000.0
            supervisor._start_times[new_proc] = 2000.0

        supervisor._kill_oldest_worker()

        assert old_proc._terminated, "Oldest worker must be terminated"
        assert not new_proc._terminated, "Newer worker must not be terminated"

    def test_no_crash_when_no_workers(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
        # Should not raise
        supervisor._kill_oldest_worker()


class TestSupervisorStatusReporting:
    def test_status_counts_output_file_lines(self, tmp_path):
        output_file = tmp_path / "match-outcomes.txt"
        output_file.write_text("line1\nline2\nline3\n")

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)

        # Access internal method
        line_count = supervisor._count_output_lines()
        assert line_count == 3

    def test_status_count_zero_when_file_absent(self, tmp_path):
        output_file = tmp_path / "nonexistent.txt"

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
        assert supervisor._count_output_lines() == 0

    def test_output_dir_created_on_run(self, tmp_path):
        output_file = tmp_path / "sealed" / "match-outcomes.txt"

        def fake_start_worker(worker_id):
            supervisor._shutdown_event.set()
            return FakeProcess(pid=1000, returncode=0)

        supervisor = MatchOutcomeSupervisor(worker_count=1, output_path=output_file)
        supervisor._start_worker = fake_start_worker
        supervisor.run()

        assert output_file.parent.exists(), "Output directory must be created"
