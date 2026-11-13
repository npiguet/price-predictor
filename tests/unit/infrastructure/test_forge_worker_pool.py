"""Unit tests for ForgeWorkerPool.

The pool is shared by ``sealed match-outcomes`` and ``draft play-draft-games``,
and the spec requires both to report progress identically — so the status line's
shape is a contract, not an implementation detail.
"""

from __future__ import annotations

import subprocess
import threading
import time
from unittest.mock import patch

import pytest

from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool, WorkerStartError
from tests.unit.infrastructure.real_processes import SLEEPER

KILL = "price_predictor.infrastructure.forge_jvm.kill_process_tree"
BIND = "price_predictor.infrastructure.forge_jvm.bind_to_this_process"


class _FakeProc:
    """Minimal stand-in for subprocess.Popen in pool-state tests."""

    def __init__(self, pid: int, alive: bool = True) -> None:
        self.pid = pid
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


def _pool(output_path, worker_count=12, **kw):
    return ForgeWorkerPool(
        worker_count=worker_count,
        spawn_worker=kw.pop("spawn_worker", lambda i: None),
        output_path=output_path,
        **kw,
    )


class TestStatusLine:
    def test_status_line_shape(self, tmp_path, capsys):
        out = tmp_path / "out.txt"
        out.write_text("a\nb\nc\n", encoding="utf-8")
        pool = _pool(out, worker_count=12)

        pool._report_status(0.0, 120.0, 0, 60.0)

        line = capsys.readouterr().out.strip()
        assert line == "[120s] 3 matches completed | 3.0 matches/min | 0/12 workers alive"

    def test_returns_count_and_time(self, tmp_path):
        out = tmp_path / "out.txt"
        out.write_text("a\nb\n", encoding="utf-8")

        count, when = _pool(out)._report_status(0.0, 60.0, 0, 0.0)

        assert (count, when) == (2, 60.0)

    def test_counts_this_runs_matches_and_returns_the_whole_file(self, tmp_path, capsys):
        out = tmp_path / "out.txt"
        out.write_text("\n" * 105, encoding="utf-8")

        # 100 rows were there when the run began; 5 are this run's
        count, _ = _pool(out)._report_status(0.0, 60.0, 100, 0.0, baseline=100)

        assert capsys.readouterr().out.startswith("[60s] 5 matches completed | 5.0 matches/min")
        assert count == 105

    def test_rate_is_per_minute_over_the_interval(self, tmp_path, capsys):
        out = tmp_path / "out.txt"
        out.write_text("\n" * 10, encoding="utf-8")

        # 6 new lines in 30s -> 12 per minute
        _pool(out)._report_status(0.0, 30.0, 4, 0.0)

        assert "12.0 matches/min" in capsys.readouterr().out

    def test_rate_is_zero_when_interval_is_zero(self, tmp_path, capsys):
        _pool(tmp_path / "out.txt")._report_status(0.0, 0.0, 0, 0.0)
        assert "0.0 matches/min" in capsys.readouterr().out

    def test_alive_count_reflects_live_processes(self, tmp_path, capsys):
        pool = _pool(tmp_path / "out.txt", worker_count=3)
        pool._processes = [_FakeProc(1), _FakeProc(2, alive=False), _FakeProc(3)]

        pool._report_status(0.0, 60.0, 0, 0.0)

        assert "2/3 workers alive" in capsys.readouterr().out


class TestLineCount:
    def test_counts_lines(self, tmp_path):
        out = tmp_path / "out.txt"
        out.write_text("one\ntwo\nthree\n", encoding="utf-8")
        assert _pool(out).output_line_count() == 3

    def test_zero_when_absent(self, tmp_path):
        assert _pool(tmp_path / "missing.txt").output_line_count() == 0


class TestRecycle:
    """Recycling bounds how long one hung match can hold a worker slot."""

    def test_oldest_alive_worker_is_killed(self, tmp_path):
        pool = _pool(tmp_path / "out.txt", worker_count=3)
        oldest, middle, newest = _FakeProc(1), _FakeProc(2), _FakeProc(3)
        pool._start_times = {oldest: 10.0, middle: 20.0, newest: 30.0}

        with patch(KILL) as killer:
            pool._kill_oldest_worker()

        killer.assert_called_once_with(oldest)

    def test_dead_workers_are_not_selected(self, tmp_path):
        pool = _pool(tmp_path / "out.txt", worker_count=2)
        dead, alive = _FakeProc(1, alive=False), _FakeProc(2)
        pool._start_times = {dead: 10.0, alive: 20.0}

        with patch(KILL) as killer:
            pool._kill_oldest_worker()

        killer.assert_called_once_with(alive)

    def test_no_workers_is_a_noop(self, tmp_path):
        with patch(KILL) as killer:
            _pool(tmp_path / "out.txt", worker_count=1)._kill_oldest_worker()

        killer.assert_not_called()


class TestStopCondition:
    """draft play-draft-games stops on a row target; match-outcomes passes None."""

    def test_should_stop_ends_the_run(self, tmp_path, capsys):
        out = tmp_path / "out.txt"
        out.write_text("a\nb\nc\n", encoding="utf-8")
        seen: list[int] = []

        pool = _pool(
            out,
            worker_count=1,
            status_interval=0,
            should_stop=lambda count: seen.append(count) or True,
        )
        with patch(KILL):
            pool._supervisor_loop()

        assert seen == [3]
        assert pool._shutdown_event.is_set()

    def test_no_recycle_when_stopping(self, tmp_path):
        out = tmp_path / "out.txt"
        out.write_text("a\n", encoding="utf-8")
        pool = _pool(
            out, worker_count=1, status_interval=0, should_stop=lambda count: True,
        )
        pool._start_times = {_FakeProc(1): 1.0}

        with patch(KILL) as killer:
            pool._supervisor_loop()

        # Only _terminate_all's kills; the recycle is skipped on the stopping tick.
        assert killer.call_count == 0

    def test_shutdown_request_stops_the_loop(self, tmp_path, capsys):
        pool = _pool(tmp_path / "out.txt", worker_count=2, status_interval=0)
        pool.request_shutdown()

        with patch(KILL):
            pool._supervisor_loop()

        assert "Shutting down, terminating 2 workers..." in capsys.readouterr().out


class _ExitingProc:
    """A worker that exits at once with ``code``, as a JVM failing at startup does."""

    def __init__(self, pid: int, code: int = 1) -> None:
        self.pid = pid
        self.returncode = code

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode


def _run_with_watchdog(pool: ForgeWorkerPool, seconds: float = 5.0) -> float:
    """``pool.run()`` on the main thread (it installs signal handlers), stopped
    by a watchdog if it never ends by itself; returns how long it took."""
    watchdog = threading.Timer(seconds, pool.request_shutdown)
    watchdog.start()
    started = time.monotonic()
    try:
        pool.run()
    finally:
        watchdog.cancel()
    return time.monotonic() - started


class TestAWorkerThatCannotStart:
    """python R2: a missing JAR or ``java`` made ``spawn_worker`` raise, and a
    worker failing at startup exited at once; both were restarted forever, at
    full speed, collecting nothing and naming no reason."""

    def test_a_spawn_that_always_raises_ends_the_run_with_an_error(self, tmp_path):
        calls = []

        def spawn(worker_id):
            calls.append(worker_id)
            raise FileNotFoundError("Connector JAR not found")

        pool = _pool(
            tmp_path / "out.txt", worker_count=2, spawn_worker=spawn,
            max_failed_starts=3, restart_backoff=0.01,
        )

        with patch(KILL), pytest.raises(WorkerStartError, match="Connector JAR not found"):
            elapsed = _run_with_watchdog(pool)
            pytest.fail(f"run() returned after {elapsed:.1f}s instead of raising")
        assert len(calls) <= 2 * 3 + 2

    def test_the_error_comes_within_about_two_seconds(self, tmp_path):
        def spawn(worker_id):
            raise FileNotFoundError("java")

        pool = _pool(
            tmp_path / "out.txt", worker_count=2, spawn_worker=spawn,
            max_failed_starts=3, restart_backoff=0.01,
        )
        started = time.monotonic()
        with patch(KILL), pytest.raises(WorkerStartError):
            _run_with_watchdog(pool)
        assert time.monotonic() - started < 2.5

    def test_quick_exits_in_every_slot_end_the_run_after_n_restarts(self, tmp_path):
        spawned = []

        def spawn(worker_id):
            spawned.append(worker_id)
            return _ExitingProc(1000 + len(spawned))

        pool = _pool(
            tmp_path / "out.txt", worker_count=1, spawn_worker=spawn,
            should_stop=lambda count: False, max_failed_starts=4,
            restart_backoff=0.01,
        )

        with patch(KILL), pytest.raises(WorkerStartError, match="exited"):
            _run_with_watchdog(pool)
        assert len(spawned) == 4

    def test_a_worker_that_ran_a_while_before_exiting_is_no_failed_start(self, tmp_path):
        spawned = []

        def spawn(worker_id):
            spawned.append(worker_id)
            return _ExitingProc(1000 + len(spawned))

        pool = _pool(
            tmp_path / "out.txt", worker_count=1, spawn_worker=spawn,
            max_failed_starts=2, restart_backoff=0.01, quick_exit_seconds=0.0,
        )

        with patch(KILL):
            _run_with_watchdog(pool, seconds=0.3)  # ends by the watchdog, no raise
        assert len(spawned) > 2

    def test_a_recycled_worker_is_no_failed_start(self, tmp_path):
        pool = _pool(tmp_path / "out.txt", worker_count=1)
        young = _FakeProc(1)
        pool._start_times = {young: time.monotonic()}

        with patch(KILL):
            pool._kill_oldest_worker()

        assert pool._failed_start(young, lifetime=0.1) is False
        assert pool._failed_start(_FakeProc(2), lifetime=0.1) is True


class TestNoWorkerOutlivesTheRun:
    """python R4: a worker spawned while the pool shut down was appended after
    ``_terminate_all`` took its snapshot, and was never killed."""

    def test_a_worker_spawned_during_shutdown_is_killed(self, tmp_path):
        from tests.unit.sealed.conftest import FakeProcess

        release = threading.Event()
        spawned: list[FakeProcess] = []

        def spawn(worker_id):
            release.wait(timeout=5)
            proc = FakeProcess(pid=4000 + len(spawned), hang=True)
            spawned.append(proc)
            return proc

        pool = _pool(
            tmp_path / "out.txt", worker_count=1, spawn_worker=spawn,
            should_stop=lambda count: True,
        )
        original = ForgeWorkerPool._terminate_all

        def terminate_then_release(self):
            original(self)
            release.set()

        with patch(KILL, side_effect=lambda proc: proc.kill()), patch.object(
            ForgeWorkerPool, "_terminate_all", terminate_then_release,
        ):
            _run_with_watchdog(pool, seconds=15)

        assert spawned, "the spawn never returned"
        assert all(proc._terminated for proc in spawned)


class TestARunThatRaises:
    """V4 #2 / TP1: ``run`` had no ``finally``, so a supervisor loop that
    raised left every worker running (the monitor threads are daemons)."""

    @pytest.mark.real_process_kill
    def test_every_worker_is_killed_when_the_loop_raises(self, tmp_path):
        started: list[subprocess.Popen] = []
        up = threading.Event()

        def spawn(worker_id):
            proc = subprocess.Popen(SLEEPER)
            started.append(proc)
            up.set()
            return proc

        def should_stop(count):
            if up.wait(timeout=10):
                raise RuntimeError("the stop condition broke")
            return False

        pool = _pool(tmp_path / "out.txt", worker_count=1, spawn_worker=spawn,
                     should_stop=should_stop)
        try:
            with pytest.raises(RuntimeError, match="stop condition broke"):
                _run_with_watchdog(pool, seconds=20)
            assert started, "no worker was started"
            assert all(proc.poll() is not None for proc in started)
        finally:
            for proc in started:
                proc.kill()


class TestSurvivors:
    """V4 #8: a worker the kill could not end was counted as ended; and a
    worker is killed once, a survivor being one still alive at the end."""

    def test_a_worker_the_kill_cannot_end_is_reported(self, tmp_path):
        pool = _pool(tmp_path / "out.txt", worker_count=1)
        pool._processes = [_FakeProc(77)]

        with patch(KILL, return_value=False):
            pool._terminate_all()

        assert pool.survivors == [77]

    def test_a_worker_dead_by_the_end_is_no_survivor(self, tmp_path):
        """The kill timed out, but the process ended before anyone asked."""
        pool = _pool(tmp_path / "out.txt", worker_count=1)
        slow = _FakeProc(77)
        pool._processes = [slow]

        with patch(KILL, return_value=False):
            pool._terminate_all()
        slow._alive = False

        assert pool.survivors == []

    def test_each_worker_is_killed_once_however_often_the_pool_terminates(self, tmp_path):
        """``_supervisor_loop`` and ``run``'s ``finally`` both terminate; a
        survivor's 20 s kill ran twice."""
        pool = _pool(tmp_path / "out.txt", worker_count=1)
        worker = _FakeProc(77)
        pool._processes = [worker]

        with patch(KILL, return_value=False) as killer:
            pool._terminate_all()
            pool._terminate_all()

        assert killer.call_count == 1

    def test_a_worker_spawned_during_shutdown_is_killed_through_the_pool(self, tmp_path):
        """The late-spawn kill ignored the kill's answer: its survivor went
        unrecorded."""
        class Waitable(_FakeProc):
            def wait(self, timeout=None):
                return None

        def spawn(worker_id):
            pool.request_shutdown()
            return Waitable(4000)

        pool = _pool(tmp_path / "out.txt", worker_count=1, spawn_worker=spawn)
        with patch(KILL, return_value=False):
            pool._monitor_worker(0)

        assert pool.survivors == [4000]


class TestEveryWorkerIsBound:
    """V4 #2: every spawned worker is bound to the supervisor's process, so a
    hard-killed supervisor takes them with it."""

    def test_the_monitor_binds_each_worker_it_spawns(self, tmp_path):
        spawned: list[_ExitingProc] = []

        def spawn(worker_id):
            spawned.append(_ExitingProc(5000 + len(spawned), code=0))
            return spawned[-1]

        pool = _pool(tmp_path / "out.txt", worker_count=1, spawn_worker=spawn,
                     max_failed_starts=2, restart_backoff=0.01)
        with patch(KILL), patch(BIND) as bind, pytest.raises(WorkerStartError):
            _run_with_watchdog(pool)

        assert [call.args[0] for call in bind.call_args_list] == spawned
