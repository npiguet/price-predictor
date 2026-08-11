"""Unit tests for ForgeWorkerPool.

The pool is shared by ``sealed match-outcomes`` and ``draft play-draft-games``,
and the spec requires both to report progress identically — so the status line's
shape is a contract, not an implementation detail.
"""

from __future__ import annotations

from unittest.mock import patch

from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool

KILL = "price_predictor.infrastructure.forge_jvm.kill_process_tree"


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
