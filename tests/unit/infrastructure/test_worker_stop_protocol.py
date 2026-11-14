"""The match workers' stop protocol, supervisor side (V4 #1, P1 #1).

The only way the pool ended a worker was ``taskkill /F /T``, mid-game: every
round's workers when its match budget was reached, and one worker a minute
for the recycle, each kill cutting a game. ``ForgeWorkerPool`` under a
``GracefulStop`` asks a worker to stop through its stop file, waits for its
game to end, and kills it only past a timeout; a worker recycles itself at a
game end. The pool here drives real child processes: ``stand_in_worker.py``,
which honours the protocol the way the Java worker does and whose literals
are pinned to the Java sources below.
"""

from __future__ import annotations

import functools
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from price_predictor.infrastructure import forge_jvm
from price_predictor.infrastructure.forge_jvm import (
    RECYCLE_SECONDS_PROPERTY,
    STARTUP_LINE,
    STOP_FILE_PROPERTY,
    ForgeWorkerPool,
    WorkerLogFiles,
    WorkerStartError,
    WorkerStopSettings,
    stop_line_of,
)
from tests.unit.infrastructure import stand_in_worker as stand_in

STAND_IN = Path(stand_in.__file__)

_JAVA = (
    Path(__file__).resolve().parents[3] / "forge-connector" / "src" / "main" / "java"
    / "com" / "pricepredictor" / "connector"
)


@functools.cache
def _java(name: str) -> str:
    return (_JAVA / name).read_text(encoding="utf-8")


# ── the stand-in and the pool read the Java worker's own literals ─────────


@pytest.mark.parametrize(("source", "literal"), [
    ("WorkerStop.java", f'STOP_FILE = "{stand_in.STOP_FILE}"'),
    ("WorkerStop.java", f'RECYCLE_SECONDS = "{stand_in.RECYCLE_SECONDS}"'),
    ("WorkerStop.java", f'"{stand_in.REASON_STOP_FILE}" + stopFile'),
    ("WorkerStop.java", f'"{stand_in.REASON_RECYCLING}" + recycleNanos / 1_000_000_000L + " s"'),
    ("WorkerStop.java", f'"{stand_in.DESCRIBE}"'),
    ("WorkerStop.java", f'stopFile + "{stand_in.DESCRIBE_EXISTS}"'),
    ("WorkerStop.java", f'"{stand_in.DESCRIBE_OR}"'),
    ("WorkerStop.java", f'"{stand_in.DESCRIBE_RUN}" + recycleNanos / 1_000_000_000L + " s"'),
    ("WorkerStop.java", f'"{stand_in.RECYCLE_REFUSAL}"'),
    ("WorkerLoop.java", f'STOPPED = "{stand_in.STOPPED}"'),
    ("WorkerLoop.java", 'STOPPED + " (" + ending.stopped() + "); " + ending.completed() + " "'),
    ("WorkerLoop.java", f'IN_THIS_JVM = "{stand_in.IN_THIS_JVM}"'),
    ("WorkerLoop.java", f'GAMES_TAIL = "{stand_in.GAMES_TAIL}"'),
    ("WorkerLoop.java", "progress.completed() + IN_THIS_JVM + games + GAMES_TAIL"),
    ("MatchWorkerMain.java", f'"{stand_in.PROGRESS_NOUN}", "Error generating match"'),
    ("MatchWorkerMain.java", f'PROGRESS_FILE = "{stand_in.PROGRESS_FILE}"'),
    ("MatchWorkerMain.java", "stop.describe().ifPresent(System.out::println)"),
    ("MatchWorkerMain.java",
     "progress.stream().println(WorkerLoop.stoppedLine(ending, progress, games))"),
    ("MatchWorkerMain.java", "stopGracefully(ending, matches, gamePlayer.gamesPlayed(), "),
    ("GamePlayer.java", "gamesPlayed++;"),
    ("MatchWorkerMain.java", "return ToolMain.exit(ExitCode.OK)"),
    ("ExitCode.java", f"BAD_CONFIG({stand_in.BAD_CONFIG})"),
    ("ExitCode.java", "OK(0)"),
])
def test_the_stand_in_speaks_as_the_java_worker(source, literal):
    """A reworded Java line, or a renamed property, would leave the stand-in
    and the pool agreeing with each other and with no real worker."""
    assert literal in _java(source)


def test_the_pool_reads_the_properties_and_lines_the_worker_writes():
    assert STOP_FILE_PROPERTY == stand_in.STOP_FILE
    assert RECYCLE_SECONDS_PROPERTY == stand_in.RECYCLE_SECONDS
    assert STARTUP_LINE == stand_in.DESCRIBE
    line = (f"{stand_in.STOPPED} ({stand_in.REASON_STOP_FILE}C:\\a (b)\\run.0.stop); "
            f"7 {stand_in.PROGRESS_NOUN}{stand_in.IN_THIS_JVM}12{stand_in.GAMES_TAIL}")
    found = stop_line_of("Worker: 10 matches generated\n" + line + "\n")
    assert found is not None
    assert (found.group("matches"), found.group("games")) == ("7", "12")
    assert found.group("reason") == "stop file C:\\a (b)\\run.0.stop"


def test_the_stand_in_refuses_a_recycle_below_one_second(tmp_path):
    result = subprocess.run(
        [sys.executable, str(STAND_IN), f"-D{stand_in.RECYCLE_SECONDS}=0"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == stand_in.BAD_CONFIG
    assert stand_in.RECYCLE_REFUSAL in result.stderr


# ── a pool of stand-ins ───────────────────────────────────────────────────


class _Run:
    """A pool of stand-in workers under ``tmp_path``: every process it
    started, each slot's log, and the progress file its games append to."""

    def __init__(self, tmp_path: Path, *, workers: int = 2, settings: WorkerStopSettings,
                 target: int | None = None, status_interval: int | None = None,
                 quick_exit_seconds: float | None = None, max_failed_starts: int | None = None,
                 **props: object) -> None:
        self.logs = WorkerLogFiles(tmp_path / "logs", "run")
        self.progress = tmp_path / "logs" / "run.progress.txt"
        self.progress.parent.mkdir(parents=True, exist_ok=True)
        self.progress.write_text("", encoding="utf-8")
        self.graceful = self.logs.graceful_stop(settings)
        self.props = {f"standin.{key.replace('_', '.')}": value for key, value in props.items()}
        self.started: list[subprocess.Popen] = []
        self.pool = ForgeWorkerPool(
            worker_count=workers, spawn_worker=self.spawn, output_path=self.progress,
            should_stop=None if target is None else (lambda count: count >= target),
            graceful=self.graceful, status_interval=status_interval,
            quick_exit_seconds=quick_exit_seconds, max_failed_starts=max_failed_starts,
            restart_backoff=0.01,
        )
        self.pool.STOP_POLL_INTERVAL = 0.05

    def spawn(self, slot: int) -> subprocess.Popen:
        properties = {
            **self.graceful.system_properties(slot),
            "sealed.progress.file": self.progress, **self.props,
        }
        proc = subprocess.Popen(
            [sys.executable, "-u", str(STAND_IN), *(f"-D{k}={v}" for k, v in properties.items())],
            stdout=self.logs.get(slot), stderr=subprocess.STDOUT,
        )
        self.started.append(proc)
        return proc

    def run(self, *, after: float | None = None, action=None, seconds: float = 30.0) -> float:
        """``pool.run()`` on the main thread (it installs signal handlers);
        ``action`` (default: ask the pool to shut down) after ``after``
        seconds; a watchdog kills everything after ``seconds``. How long it took."""
        timers = []
        if after is not None:
            timers.append(threading.Timer(after, action or self.pool.request_shutdown))
        timers.append(threading.Timer(seconds, self._watchdog))
        for timer in timers:
            timer.start()
        began = time.monotonic()
        try:
            self.pool.run()
        finally:
            for timer in timers:
                timer.cancel()
            self.logs.close_all()
        return time.monotonic() - began

    def _watchdog(self) -> None:
        self.pool.request_shutdown()
        self.pool._hurry.set()

    def end(self) -> None:
        for proc in self.started:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)

    def log(self) -> str:
        return "\n".join(
            self.logs.path_for(slot).read_text(encoding="utf-8", errors="replace")
            for slot in range(self.pool._worker_count) if self.logs.path_for(slot).exists()
        )

    def games(self) -> int:
        return len(self.progress.read_text(encoding="utf-8").splitlines())


@pytest.fixture
def runs():
    made: list[_Run] = []

    def make(*args, **kwargs) -> _Run:
        made.append(_Run(*args, **kwargs))
        return made[-1]

    yield make
    for run in made:
        run.end()


def _settings(**overrides) -> WorkerStopSettings:
    values = {"timeout": 20.0, "recycle_seconds": 3600, "init_seconds": 20.0}
    return WorkerStopSettings(**{**values, **overrides})


class TestAGracefulStop:
    def test_a_round_s_end_lets_every_worker_finish_its_game(
        self, tmp_path, runs, killed_process_trees,
    ):
        """The round's match budget reached: each worker finishes the game it
        plays and stops, so every game started was finished and none was cut."""
        run = runs(tmp_path, settings=_settings(), target=4, game_seconds=0.2)

        run.run()

        assert all(proc.poll() == 0 for proc in run.started)
        assert killed_process_trees == []
        stops = run.pool.stops
        assert (stops.stopped, stops.killed, stops.crashed) == (2, 0, 0)
        lines = [stop_line_of(text) for text in run.log().split("\n") if stop_line_of(text)]
        assert len(lines) == 2
        assert all(line.group("reason").startswith("stop file ") for line in lines)
        # Every game the progress file counts is one a stop line counts: none cut.
        assert sum(int(line.group("matches")) for line in lines) == run.games() >= 4

    def test_the_pool_returns_only_once_every_worker_has_ended(self, tmp_path, runs):
        """The job object kills the workers when the supervisor exits, so a
        stop still under way when ``run`` returned would be a kill."""
        run = runs(tmp_path, settings=_settings(), game_seconds=1.5)

        elapsed = run.run(after=0.5)

        assert elapsed >= 1.4, "it returned before the game in progress ended"
        assert all(proc.poll() is not None for proc in run.started)
        assert run.pool.stops.stopped == 2
        assert run.pool.survivors == []

    def test_each_worker_s_stop_file_is_deleted_before_it_starts(self, tmp_path, runs):
        """A stop file left by the last round's stop would stop the next
        round's worker before its first match."""
        run = runs(tmp_path, workers=1, settings=_settings(), target=2, game_seconds=0.1)
        stale = run.logs.stop_path_for(0)
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.touch()

        run.run()

        assert run.games() >= 2


class TestTheTimeout:
    @pytest.mark.real_process_kill
    def test_a_worker_that_does_not_stop_is_killed_and_reported(self, tmp_path, runs, capsys):
        run = runs(tmp_path, settings=_settings(timeout=0.5), behaviour="ignore",
                   game_seconds=0.1)

        elapsed = run.run(after=0.5)

        assert elapsed < 8
        assert all(proc.poll() not in (None, 0) for proc in run.started)
        stops = run.pool.stops
        assert (stops.stopped, stops.killed) == (0, 2)
        assert stop_line_of(run.log()) is None
        assert "did not stop within 0s of the request; its game is cut" in capsys.readouterr().out

    def test_a_second_interrupt_kills_at_once(self, tmp_path, runs):
        run = runs(tmp_path, settings=_settings(timeout=60), behaviour="ignore")

        def interrupt_twice():
            run.pool._handle_signal(signal.SIGINT, None)
            time.sleep(0.5)
            run.pool._handle_signal(signal.SIGINT, None)

        elapsed = run.run(after=0.3, action=interrupt_twice)

        assert elapsed < 10
        assert run.pool.interrupted
        assert run.pool.stops.killed == 2

    def test_a_worker_whose_game_outlasts_its_recycle_is_killed(self, tmp_path, runs, capsys):
        """The backstop the kill-based recycle was: a game that does not end
        is killed ``recycle + timeout`` seconds into its worker's life."""
        run = runs(tmp_path, workers=1, settings=_settings(timeout=0.5, recycle_seconds=1),
                   behaviour="ignore", status_interval=1)

        run.run(after=3.5)

        out = capsys.readouterr().out
        assert "its game has not ended 0s past its 1s recycle time" in out
        assert len(run.started) >= 2, "the killed worker was restarted"
        assert run.pool.stops.killed >= 2


class TestRecycling:
    def test_a_worker_recycles_itself_at_a_game_end_and_is_restarted(
        self, tmp_path, runs, capsys, killed_process_trees,
    ):
        run = runs(tmp_path, settings=_settings(recycle_seconds=1), game_seconds=0.2)

        run.run(after=3.0)

        assert killed_process_trees == []
        assert run.pool.stops.recycled >= 2
        assert len(run.started) > 2
        assert "recycling after 1 s" in run.log()
        assert "recycled itself at a game end" in capsys.readouterr().out

    def test_a_worker_that_recycles_before_its_first_match_failed_to_start(
        self, tmp_path, runs,
    ):
        """A recycle time shorter than Forge's start stops every worker
        before it plays, and a restart on exit 0 would loop for ever."""
        run = runs(tmp_path, workers=1, settings=_settings(recycle_seconds=1),
                   init_seconds=1.2, max_failed_starts=2)

        with pytest.raises(WorkerStartError, match="--recycle-seconds"):
            run.run()


class TestACrash:
    def test_a_worker_that_fails_is_restarted(self, tmp_path, runs):
        run = runs(tmp_path, workers=1, settings=_settings(), crash_after_games=1,
                   quick_exit_seconds=0.0)

        run.run(after=1.5)

        assert len(run.started) >= 2
        assert run.pool.stops.crashed >= 1
        assert run.pool.stops.recycled == 0

    def test_exit_0_without_the_stop_line_is_a_crash(self, tmp_path, runs, capsys):
        run = runs(tmp_path, workers=1, settings=_settings(recycle_seconds=1),
                   behaviour="exit0-without-line", quick_exit_seconds=0.0)

        run.run(after=2.5)

        assert run.pool.stops.recycled == 0
        assert run.pool.stops.crashed >= 1
        assert "Worker 0 exited (code 0), restarting..." in capsys.readouterr().out


class TestCtrlC:
    def test_workers_the_console_interrupt_ended_are_not_waited_for(
        self, tmp_path, runs, killed_process_trees,
    ):
        """A console Ctrl-C reaches every JVM, whose shutdown hook closes its
        shard: the pool waits out no stop timeout for a worker already gone."""
        die = tmp_path / "ctrl-c"
        run = runs(tmp_path, settings=_settings(timeout=60), die_file=die, game_seconds=5)

        def ctrl_c():
            die.touch()
            time.sleep(0.2)
            run.pool._handle_signal(signal.SIGINT, None)

        elapsed = run.run(after=0.5, action=ctrl_c)

        assert elapsed < 10
        assert run.pool.interrupted
        assert killed_process_trees == []
        assert all(proc.poll() == 130 for proc in run.started)
        assert run.pool.stops.killed == 0


class TestAStopDuringForgeStart:
    def test_waits_for_the_start_rather_than_killing_the_worker(
        self, tmp_path, runs, killed_process_trees,
    ):
        """The worker reads a request only once Forge is initialised; the
        stop timeout counted from the request would kill it mid-start."""
        run = runs(tmp_path, settings=_settings(timeout=0.2), init_seconds=1.5)

        elapsed = run.run(after=0.3)

        assert elapsed >= 1.4
        assert killed_process_trees == []
        assert run.pool.stops.stopped == 2
        assert "; 0 matches generated in this JVM" in run.log()

    def test_but_not_for_ever(self, tmp_path, runs):
        run = runs(tmp_path, settings=_settings(timeout=0.2, init_seconds=1.0),
                   init_seconds=60)

        elapsed = run.run(after=0.2)

        assert elapsed < 10
        assert run.pool.stops.killed == 2


def test_without_the_protocol_the_pool_kills_as_before(tmp_path, killed_process_trees):
    """draft's workers have no stop protocol: their pool kills them."""
    progress = tmp_path / "out.txt"
    started: list[subprocess.Popen] = []

    def spawn(slot):
        started.append(subprocess.Popen(
            [sys.executable, str(STAND_IN), "-Dstandin.game.seconds=0.1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
        return started[-1]

    pool = ForgeWorkerPool(worker_count=1, spawn_worker=spawn, output_path=progress)
    timer = threading.Timer(0.5, pool.request_shutdown)
    timer.start()
    try:
        pool.run()
    finally:
        timer.cancel()
        for proc in started:
            proc.kill()
            proc.wait(timeout=10)
    assert killed_process_trees == started


def test_the_stop_properties_name_the_slot_s_stop_file_beside_its_log(tmp_path):
    logs = WorkerLogFiles(tmp_path / "logs", "run-1")
    graceful = logs.graceful_stop(WorkerStopSettings(recycle_seconds=900))

    assert graceful.system_properties(3) == {
        "worker.stop.file": str(tmp_path / "logs" / "run-1.3.stop"),
        "worker.recycle.seconds": "900",
    }
    assert re.fullmatch(r"run-1\.3\.stop", Path(graceful.stop_file_for(3)).name)
    assert forge_jvm.DEFAULT_STOP_TIMEOUT == 600
    assert forge_jvm.DEFAULT_RECYCLE_SECONDS == 600
