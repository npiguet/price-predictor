"""The Forge worker pool: ``ForgeWorkerPool``, the ``--workers`` type and the
error of a pool that could start no worker."""

from __future__ import annotations

import dataclasses
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from price_predictor.infrastructure.job_objects import bind_to_this_process
from price_predictor.infrastructure.jvm_processes import kill_process_tree
from price_predictor.infrastructure.worker_log_mark import LogMark, past_forge_start, stop_line_of
from price_predictor.infrastructure.worker_pool_books import PoolWorker, WorkerBooks
from price_predictor.infrastructure.worker_stops import GracefulStop, WorkerStops

#: The most workers a pool may run: the machine runs at most twelve CPU-busy
#: processes, and a Forge worker is one.
MAX_WORKERS = 12


def worker_count(text: str) -> int:
    """``--workers`` as argparse reads it: 1 to :data:`MAX_WORKERS`. With
    none the pool never starts a worker and never stops; past the cap the
    machine is oversubscribed."""
    import argparse

    value = int(text)
    if not 1 <= value <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(
            f"must be between 1 and {MAX_WORKERS}, got {value}"
        )
    return value


class WorkerStartError(RuntimeError):
    """Every slot of a :class:`ForgeWorkerPool` failed to start a worker
    ``max_failed_starts`` times in a row: the run cannot collect anything."""


#: How much of a slot's log is read back for a worker's stop line: the line
#: is the last its worker prints, with at most the hook-failure report and a
#: few stderr lines after it.
_LOG_TAIL_BYTES = 64 * 1024


def _seconds(value: float) -> str:
    """A duration as a report prints it: ``600s``, ``0.5s``."""
    return f"{value:g}s"


def _or_default(value, default):
    """``value``, or ``default`` when it is None."""
    return default if value is None else value


@dataclasses.dataclass(frozen=True)
class PoolOptions:
    """How a :class:`ForgeWorkerPool` runs, beyond which workers it starts.
    A field left None takes the pool's class constant (or nothing).

    Attributes:
        status_interval: seconds between status lines (default 60).
        should_stop: called with the current line count after each status
            line; returning True ends the run. None means run until signalled.
        on_status: called after each status line (a log scan); what it
            raises is printed and the run goes on.
        graceful: the workers' stop protocol; None kills them instead.
        max_failed_starts, restart_backoff, quick_exit_seconds: the
            failed-start rule (:class:`ForgeWorkerPool`'s docstring).
    """

    status_interval: int | None = None
    should_stop: Callable[[int], bool] | None = None
    on_status: Callable[[], None] | None = None
    graceful: GracefulStop | None = None
    max_failed_starts: int | None = None
    restart_backoff: float | None = None
    quick_exit_seconds: float | None = None


@dataclasses.dataclass(frozen=True)
class LastStatus:
    """Where the last status line left off: when the run started and how
    many lines the output held then (``baseline``), and the line count and
    time of the last line."""

    start_time: float
    count: int
    time: float
    baseline: int = 0


@dataclasses.dataclass(frozen=True)
class _Ended:
    """How a worker ended, as its monitor reads it: its :class:`WorkerStops`
    field, the matches and games of one that printed the stop line, its
    lifetime and whether it failed to start."""

    ending: str
    matches: int | None
    games: int | None
    lifetime: float
    failed: bool


class ForgeWorkerPool:
    """Supervise a pool of Forge JVM workers that append to one output file.

    Spawns ``worker_count`` workers, monitors each in its own thread, restarts
    the ones that end, reports status every ``status_interval`` seconds and
    shuts down cleanly on SIGINT/SIGTERM.

    Workers are recycled, so no JVM lives long (``DEFAULT_RECYCLE_SECONDS``
    says why): every status interval the pool kills the longest-running
    worker (:meth:`_kill_oldest_worker`), and the monitor thread starts a
    fresh one. Beyond that, the pool works in one of two ways:

    - With ``graceful`` (:class:`GracefulStop`, the match workers' stop
      protocol, ``contracts/cli.md`` § Match worker stop protocol), the
      periodic kill cuts the game its worker plays and passes over a worker
      still starting Forge; each worker also stops itself at the first game
      end past its recycle time, printing the stop line and exiting 0, and
      the pool starts a fresh one, unless the run's stop condition already
      holds; a worker ``recycle + timeout`` seconds old, its game not
      ending, is killed and its game cut. To stop the run (its target
      reached, a shutdown, a first Ctrl-C, which reaches the supervisor
      alone: :func:`graceful_popen_options`) the pool creates every running
      worker's stop file and waits for each to finish its game, killing one
      only past the timeout (no periodic kill meanwhile), and kills at once
      one still starting Forge, which has no game to cut; a second Ctrl-C
      kills every worker still finishing its game. ``run`` returns once
      every worker has ended or been killed, since a supervisor that exits
      takes its workers with it (:func:`bind_to_this_process`).
    - Without, the pool kills every worker at the end, mid-game.

    Progress is measured by counting lines in ``output_path`` rather than by
    workers reporting back, so a worker needs no channel to the supervisor.

    A failed start (``spawn_worker`` raising, a worker exiting within
    ``quick_exit_seconds`` of its start without the pool having killed it, or
    one recycling itself before it played a game) is retried after a backoff
    that doubles from ``restart_backoff``; the first retry is immediate, as
    after an ordinary crash. Once every slot has failed ``max_failed_starts``
    starts in a row, ``run`` ends with :class:`WorkerStartError`: a missing
    JAR, a card database that stops every Forge startup or an unreadable
    input file would otherwise be retried forever, collecting nothing.

    Used by ``sealed match-outcomes``, the ``effects`` collectors and ``draft
    play-draft-games`` (whose workers have no stop protocol); the printed
    lines are part of their operator contract and must not drift.
    """

    #: Seconds between status reports, and how often the longest-running
    #: worker is recycled — see
    #: :meth:`_kill_oldest_worker`, which explains what that recycle defends
    #: against. Do not lengthen it to win throughput.
    STATUS_INTERVAL = 60

    #: Consecutive failed starts in every slot that end the run.
    MAX_FAILED_STARTS = 5
    #: The second retry's delay in seconds; each later one doubles it.
    RESTART_BACKOFF = 1.0
    MAX_RESTART_BACKOFF = 30.0
    #: A worker exiting this soon after its start, unkilled, failed to start:
    #: a Forge start takes tens of seconds, and a worker plays until recycled.
    QUICK_EXIT_SECONDS = 120.0
    #: How often a graceful stop looks at the workers it waits for.
    STOP_POLL_SECONDS = 0.05

    def __init__(
        self,
        *,
        worker_count: int,
        spawn_worker: Callable[[int], subprocess.Popen],
        output_path: Path,
        options: PoolOptions | None = None,
    ) -> None:
        """
        Args:
            worker_count: how many workers to keep alive.
            spawn_worker: called with a worker index, returns a started process.
                Under ``options.graceful`` it must give the worker
                ``graceful.system_properties(index)``.
            output_path: the file workers append to; its line count is progress.
            options: the status interval, stop condition, status hook, stop
                protocol and failed-start rule (:class:`PoolOptions`).
        """
        options = options or PoolOptions()
        self._worker_count = worker_count
        self._spawn_worker = spawn_worker
        self._output_path = Path(output_path)
        self._status_interval = _or_default(options.status_interval, self.STATUS_INTERVAL)
        self._should_stop = options.should_stop
        self._on_status = options.on_status
        self._graceful = options.graceful
        self._shutdown_event = threading.Event()
        #: Set by a second interrupt: every worker still asked to stop is
        #: killed at once.
        self._hurry = threading.Event()
        self._interrupted = False
        self._books = WorkerBooks()
        self._max_failed_starts = _or_default(options.max_failed_starts, self.MAX_FAILED_STARTS)
        self._restart_backoff = _or_default(options.restart_backoff, self.RESTART_BACKOFF)
        self._quick_exit_seconds = _or_default(
            options.quick_exit_seconds, self.QUICK_EXIT_SECONDS,
        )
        self._start_failure: str | None = None

    @property
    def interrupted(self) -> bool:
        """True when the run ended on SIGINT/SIGTERM rather than by itself.

        Callers need this to tell a clean finish from a Ctrl-C, which decide
        different exit codes.
        """
        return self._interrupted

    @property
    def survivors(self) -> list[int]:
        """PIDs of the workers the pool killed, recycled or at shutdown, that
        are still running when asked.

        Each may still append to its shard and to the progress file it was
        started with, so a caller counting either as final should say so. A
        kill that timed out on a process that ended later leaves no survivor.
        """
        return self._books.survivors()

    @property
    def stops(self) -> WorkerStops:
        """How the run's workers ended, by how, so far."""
        return self._books.stops_so_far()

    def run(self) -> None:
        """Start all workers and block until shutdown.

        Returns only once every worker has ended or been killed (a graceful
        stop waits for each to finish its game). Raises
        :class:`WorkerStartError` when every slot failed to start a worker
        ``max_failed_starts`` times in a row.
        """
        previous = self._install_signal_handlers()
        try:
            self._run_workers()
        finally:
            self._restore_signal_handlers(previous)

    def _install_signal_handlers(self) -> dict[int, object]:
        """Handle SIGINT and SIGTERM for the run; the handlers they replace."""
        previous = {
            signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in previous:
            signal.signal(signum, self._handle_signal)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: dict[int, object]) -> None:
        """Put back the handlers :meth:`_install_signal_handlers` replaced, so
        a signal after the run reaches its caller and not a finished pool. A
        handler Python did not install (None) cannot be set again and stays."""
        for signum, handler in previous.items():
            if handler is not None:
                signal.signal(signum, handler)

    def _run_workers(self) -> None:
        """The body of :meth:`run`, under its signal handlers."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Starting {self._worker_count} workers...")

        monitor_threads = []
        try:
            for i in range(self._worker_count):
                t = threading.Thread(target=self._monitor_worker, args=(i,), daemon=True)
                t.start()
                monitor_threads.append(t)

            self._supervisor_loop()
        finally:
            # However the loop ended: the monitor threads are daemons, so a
            # raise here would otherwise leave every worker running. Under
            # the protocol this waits out every graceful stop, before the
            # supervisor can exit and its job object kill the workers.
            self._shutdown_event.set()
            self._terminate_all()
            for t in monitor_threads:
                t.join(timeout=10)
            self._remove_stop_files()

        if self._start_failure is not None:
            raise WorkerStartError(
                f"every one of {self._worker_count} worker slot(s) failed to "
                f"start {self._max_failed_starts} time(s) in a row; last: "
                f"{self._start_failure}"
            )
        print("Done.")

    def _remove_stop_files(self) -> None:
        """Under the protocol, once every worker has ended: delete each slot's
        stop file, which the run's end created and nothing reads any more.
        Kept while a worker the kill could not end survives, which may still
        read its own."""
        if self._graceful is None or self.survivors:
            return
        for slot in range(self._worker_count):
            stop_file = Path(self._graceful.stop_file_for(slot))
            try:
                stop_file.unlink(missing_ok=True)
            except OSError as exc:
                print(f"Could not delete worker {slot}'s stop file {stop_file}: {exc}")

    def output_line_count(self) -> int:
        """Count lines in the output file (= number of completed matches)."""
        if not self._output_path.exists():
            return 0
        try:
            with open(self._output_path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def request_shutdown(self) -> None:
        """Ask the pool to stop; the supervisor loop exits at its next tick."""
        self._shutdown_event.set()

    def _monitor_worker(self, worker_id: int) -> None:
        """Monitor one worker, restarting it when it ends until shutdown.

        The restart reuses the same ``worker_id``, because the id names a pool
        slot and not a process. Together with recycling that makes one slot
        hundreds of JVMs over a run, so nothing a worker counts from zero may
        be namespaced by the slot alone.
        """
        first = True
        while not self._shutdown_event.is_set():
            if not first and self._stop_condition_holds():
                # A worker respawned now would only start Forge to be stopped.
                self._shutdown_event.set()
                return
            first = False
            worker = self._spawn(worker_id)
            if worker is None:
                continue
            ended = self._wait_out(worker)
            if self._shutdown_event.is_set():
                return
            self._restart_after(worker, ended)

    def _spawn(self, worker_id: int) -> PoolWorker | None:
        """Spawn a worker in slot ``worker_id``, bind it to this process and
        keep it; None when the spawn raised, which counts as a failed start
        unless the pool shuts down."""
        mark = self._prepare_slot(worker_id)
        try:
            proc = self._spawn_worker(worker_id)
        except Exception as exc:
            if not self._shutdown_event.is_set():
                print(f"Monitor error for worker {worker_id}: {exc}")
                self._note_failed_start(worker_id, f"{type(exc).__name__}: {exc}")
            return None
        started = time.monotonic()
        bind_to_this_process(proc)
        worker = self._books.track(proc, slot=worker_id, started=started, mark=mark)
        # _terminate_all may have taken its snapshot before the worker was
        # kept; a worker spawned while the pool shuts down is killed here, or
        # it outlives the run. It is still starting Forge: no game is cut.
        if self._shutdown_event.is_set():
            self._kill(worker, starting=self._graceful is not None)
        return worker

    def _wait_out(self, worker: PoolWorker) -> _Ended:
        """Wait for ``worker`` to end, read how it ended and drop it."""
        try:
            worker.process.wait()
        except Exception as exc:
            print(f"Monitor error for worker {worker.slot}: {exc}")

        lifetime = time.monotonic() - worker.started
        ending, matches, games = self._ending(worker)
        # A worker that recycled itself played until its recycle time,
        # however short: it failed to start only if it played no game (a
        # best-of-N match spanning the recycle time finishes no match).
        failed = (
            games == 0 if ending == "recycled"
            else self._failed_start(worker.process, lifetime=lifetime)
        )
        self._books.drop(worker, ending)
        return _Ended(ending, matches, games, lifetime, failed)

    def _restart_after(self, worker: PoolWorker, ended: _Ended) -> None:
        """Say why a slot's worker is restarted, and count a failed start or
        clear the slot's count."""
        proc = worker.process
        if ended.ending == "recycled":
            print(
                f"Worker {worker.slot} recycled itself at a game end after "
                f"{ended.lifetime:.0f}s ({ended.matches} matches, {ended.games} games), "
                "restarting..."
            )
        else:
            print(f"Worker {worker.slot} exited (code {proc.returncode}), restarting...")
        if ended.failed:
            self._note_failed_start(worker.slot, self._failure_reason(proc, ended))
            return
        self._books.clear_failed_starts(worker.slot)

    @staticmethod
    def _failure_reason(proc: subprocess.Popen, ended: _Ended) -> str:
        """What a failed start's error says, should it end the run."""
        if ended.ending == "recycled":
            return (
                f"worker recycled itself {ended.lifetime:.0f}s after starting without "
                "playing a game (is --recycle-seconds shorter than a Forge start?)"
            )
        return f"worker exited (code {proc.returncode}) {ended.lifetime:.0f}s after starting"

    def _prepare_slot(self, worker_id: int) -> LogMark | None:
        """Under the protocol, before a slot's worker is spawned: delete its
        stop file, which an earlier worker's stop left there and which would
        stop the new one before its first match, and mark where its log
        stands. None without the protocol."""
        if self._graceful is None:
            return None
        stop_file = Path(self._graceful.stop_file_for(worker_id))
        try:
            stop_file.unlink(missing_ok=True)
        except OSError as exc:
            print(f"Could not delete worker {worker_id}'s stop file {stop_file}: {exc}")
        return LogMark.of(Path(self._graceful.log_path_for(worker_id)))

    def _ending(self, worker: PoolWorker) -> tuple[str, int | None, int | None]:
        """How an ended worker ended (a :class:`WorkerStops` field) and, for
        one that printed the stop line, the matches its JVM finished and the
        games it played.

        Exit 0 with the stop line since its spawn is a stop at a game end:
        ``stopped`` when the pool asked, ``recycled`` when it did not. Exit 0
        without the line, or any other code, is no stop: ``killed_starting``
        or ``killed`` when the pool killed it, ``ended_otherwise`` when the
        pool had asked, else ``crashed``."""
        asked, killed, starting = self._books.kills_of(worker)
        line = self._stop_line(worker)
        if line is not None:
            return (("stopped" if asked else "recycled"), int(line.group("matches")),
                    int(line.group("games")))
        if killed:
            return ("killed_starting" if starting else "killed"), None, None
        return ("ended_otherwise" if asked else "crashed"), None, None

    @staticmethod
    def _stop_line(worker: PoolWorker):
        """The stop line an ended worker printed since its spawn, if it
        exited 0 under the protocol; else None."""
        if getattr(worker.process, "returncode", None) != 0 or worker.mark is None:
            return None
        return stop_line_of(worker.mark.written_since(tail=_LOG_TAIL_BYTES))

    def _failed_start(self, proc: subprocess.Popen, *, lifetime: float) -> bool:
        """Whether a worker that exited after ``lifetime`` seconds failed to
        start: it exited that soon and the pool did not kill it."""
        killed = self._books.was_killed(proc)
        return not killed and lifetime < self._quick_exit_seconds

    def _note_failed_start(self, worker_id: int, reason: str) -> None:
        """Count one failed start of ``worker_id``. Once every slot has failed
        ``max_failed_starts`` in a row the run ends; otherwise wait out the
        backoff."""
        failures, every_slot_failed = self._books.note_failed_start(
            worker_id, limit=self._max_failed_starts, slots=self._worker_count,
        )
        if every_slot_failed:
            self._start_failure = reason
            self._shutdown_event.set()
            return
        if failures > 1:
            delay = min(
                self._restart_backoff * 2 ** (failures - 2), self.MAX_RESTART_BACKOFF,
            )
            self._shutdown_event.wait(timeout=delay)

    def _stop_condition_holds(self) -> bool:
        """Whether the run's stop condition holds now; False without one, or
        when asking it raises (the supervisor loop meets the same raise)."""
        if self._should_stop is None:
            return False
        try:
            return bool(self._should_stop(self.output_line_count()))
        except Exception:  # noqa: BLE001 — the supervisor loop reports it
            return False

    #: How often the stop condition is polled, in seconds. Far shorter than the
    #: status interval: a run with a small target would otherwise overshoot it by
    #: everything the workers finish before the next status line.
    STOP_POLL_INTERVAL = 2

    def _supervisor_loop(self) -> None:
        """Report status every status interval, and at each report, under
        the protocol, kill a worker whose game has not ended long past its
        recycle time, then recycle the oldest worker.

        The stop condition is polled far more often than that, so a small
        ``--n-pairings`` stops near its target instead of at the next status tick.
        """
        start_time = time.monotonic()
        # The output file is append-only and usually holds earlier runs' rows:
        # the status line counts this run's, and the first rate is not inflated
        # by the whole file.
        baseline = self.output_line_count()
        last = LastStatus(start_time, baseline, start_time, baseline)
        next_status = start_time + self._status_interval

        while not self._shutdown_event.is_set():
            if self._stop_polled():
                break
            self._shutdown_event.wait(timeout=self._sleep_slice(next_status))
            if self._shutdown_event.is_set():
                break
            now = time.monotonic()
            if now < next_status:
                continue  # a stop-poll slice, not a status tick
            last = self._status_tick(last, now)
            next_status = now + self._status_interval
            if self._stops_at(last.count):
                break
            self._recycle()

        self._announce_shutdown()
        self._terminate_all()

    def _stop_polled(self) -> bool:
        """:meth:`_stops_at` the output's line count now, the file read only
        when there is a stop condition."""
        return self._should_stop is not None and self._stops_at(self.output_line_count())

    def _stops_at(self, count: int) -> bool:
        """Whether the stop condition holds at line count ``count``; sets
        the shutdown when it does. What the condition raises ends the run."""
        if self._should_stop is None or not self._should_stop(count):
            return False
        self._shutdown_event.set()
        return True

    def _status_tick(self, last: LastStatus, now: float) -> LastStatus:
        """Print the status line and run the status hook; where this line
        left off."""
        count, when = self._report_status(last, now)
        if self._on_status is not None:
            try:
                self._on_status()
            except Exception as exc:  # noqa: BLE001 — a report never stops a run
                print(f"Status report failed: {exc}")
        return LastStatus(last.start_time, count, when, last.baseline)

    def _recycle(self) -> None:
        """A status tick's kills: under the protocol the overdue workers,
        then, always, the oldest."""
        if self._graceful is not None:
            self._kill_overdue_workers()
        self._kill_oldest_worker()

    def _announce_shutdown(self) -> None:
        """Say how the run's workers are ended."""
        if self._graceful is None:
            print(f"Shutting down, terminating {self._worker_count} workers...")
        else:
            print(
                f"Shutting down, asking {self._worker_count} workers to stop at "
                "their game's end..."
            )

    def _sleep_slice(self, next_status: float) -> float:
        """How long to wait next: the stop-poll slice, or the rest of the interval."""
        remaining = next_status - time.monotonic()
        if self._should_stop is None:
            return max(0.0, remaining)
        return max(0.0, min(float(self.STOP_POLL_INTERVAL), remaining))

    def _report_status(self, last: LastStatus, now: float) -> tuple[int, float]:
        """Print one status line and return the file's line count and ``now``.

        ``last.baseline`` is the line count when the run started; the line
        reports the matches completed since, while the returned count stays
        the whole file's, which is what ``should_stop`` is handed.
        """
        elapsed = now - last.start_time
        interval = now - last.time

        count = self.output_line_count()
        delta = count - last.count
        rate = delta / interval * 60 if interval > 0 else 0.0

        alive = self._books.alive_count()

        print(
            f"[{elapsed:.0f}s] {count - last.baseline} matches completed"
            f" | {rate:.1f} matches/min"
            f" | {alive}/{self._worker_count} workers alive"
        )
        return count, now

    def _kill_oldest_worker(self) -> None:
        """Kill the longest-running worker so the monitor thread restarts it
        fresh: the recycle of every pool, each status interval.

        Why: Forge workers sometimes get stuck in infinite or near-infinite loops,
        holding a slot indefinitely, and about one game in 400 crawls (Forge's
        AI taking seconds per decision on a huge board, the game lasting 8 to
        20 minutes). Recycling the oldest one each status interval bounds how
        long any single match can occupy a worker.

        Under the stop protocol the kill cuts the game the worker plays (no
        game-end marker: ``build-corpus`` drops it), the worker counts as
        ``killed``, and three kinds of worker are passed over: one asked to
        stop (a graceful stop is finishing its game), one the pool already
        tried to kill, and one still starting Forge (no startup line in its
        log since its spawn). Killing that last would only start Forge again,
        and twelve JVMs starting at once take minutes: a pool that killed its
        oldest each minute regardless could kill every worker before its
        first game. When every worker is still starting, none is killed; one
        that never finishes starting is killed by :meth:`_kill_overdue_workers`.
        No kill happens once the pool shuts down (:meth:`_stop_all` asks
        every worker to finish its game).

        This is containment, not tuning, and it is one measure with the worker's
        small ``-Xmx`` (``match_worker_connector.WORKER_XMX``): Forge degrades late in a
        JVM's life, and a small heap plus a hard recycle bounds that instead of
        letting one worker consume a run. So restarts are the normal operating
        condition, not an accident: anything a worker counts from zero must be
        namespaced by the JVM lifetime rather than by the worker slot;
        effect-record ids are, via the lifetime token in RecordShardWriter. Do
        not lengthen or remove this interval to win throughput: what it
        contains costs far more than the restarts do.
        """
        if self._graceful is not None:
            self._kill_oldest_playing_worker()
            return
        found = self._books.oldest_alive()
        if found is None:
            return
        oldest, age = found
        self._kill(oldest)
        print(f"Recycled longest-running worker (PID {oldest.process.pid}, age {age:.0f}s)")

    def _kill_oldest_playing_worker(self) -> None:
        """:meth:`_kill_oldest_worker` under the stop protocol: the oldest
        worker past its Forge start, neither asked to stop nor already killed."""
        if self._shutdown_event.is_set():
            return
        candidates = sorted(self._books.recyclable(), key=lambda w: w.started)
        oldest = next((w for w in candidates if past_forge_start(w.mark)), None)
        if oldest is None:
            return
        age = time.monotonic() - oldest.started
        self._kill(oldest)
        print(
            f"Recycled longest-running worker {oldest.slot} (PID {oldest.process.pid}, age "
            f"{age:.0f}s); the game it was playing is cut (no game-end "
            "marker: build-corpus drops it)"
        )

    def _kill_overdue_workers(self) -> None:
        """Kill every worker still running ``recycle + timeout`` seconds after
        its start: its recycle time asked it to stop at its game's end, and
        that game has not ended within the stop timeout. Under the periodic
        recycle (:meth:`_kill_oldest_worker`) a worker rarely lives that long;
        this ends one that does, such as one still starting Forge, which
        that recycle passes over. A game has no time bound, so this cuts a game
        that is merely long: the line says so, the worker counts as
        ``killed`` and its game, without a marker, among the round's cut
        games."""
        settings = self._graceful.settings
        limit = settings.recycle_seconds + settings.timeout
        now = time.monotonic()
        overdue = [
            (w, now - w.started) for w in self._books.recyclable() if now - w.started > limit
        ]
        for worker, age in overdue:
            print(
                f"Killing worker {worker.slot} (PID {worker.process.pid}), {age:.0f}s old: "
                f"its game has not ended {_seconds(settings.timeout)} past its "
                f"{_seconds(settings.recycle_seconds)} recycle time; the game it "
                "was playing is cut (no game-end marker: build-corpus drops it)"
            )
            self._kill(worker)

    def _terminate_all(self) -> None:
        """End every running worker: without the protocol, kill it; with it,
        ask it to stop at its game's end and wait (:meth:`_stop_all`)."""
        workers = self._books.running()
        if self._graceful is None:
            for worker in workers:
                self._kill(worker)
            return
        self._stop_all(workers)

    def _stop_all(self, workers: list[PoolWorker]) -> None:
        """Ask every running worker of ``workers`` to stop at its game's end,
        then wait until each has ended, killing one ``timeout`` seconds after
        the request. A worker whose log shows it still starting Forge (no
        startup line since its spawn) is killed at once, counted as
        ``killed_starting``: its stop file is created first, so one that
        prints its startup line meanwhile stops before its first match, and
        none has a game to cut. One that has already ended is not waited
        for; one asked before keeps its deadline, so a second call waits out
        no timeout twice; a second interrupt kills at once."""
        timeout = self._graceful.settings.timeout
        asked = self._books.ask_to_stop(workers, deadline=time.monotonic() + timeout)
        self._create_stop_files(asked)
        starting = self._kill_starting(asked)
        if len(asked) > len(starting):
            print(
                f"Asked {len(asked) - len(starting)} worker(s) to stop at their game's "
                f"end; each is killed {_seconds(timeout)} after the request"
            )
        self._wait_for_stops(asked)

    def _create_stop_files(self, asked: list[PoolWorker]) -> None:
        """Create each asked worker's stop file; kill one whose file cannot be."""
        for worker in asked:
            stop_file = Path(self._graceful.stop_file_for(worker.slot))
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch()
            except OSError as exc:
                print(f"Could not ask worker PID {worker.process.pid} to stop "
                      f"({stop_file}: {exc}); killing it")
                self._kill(worker)

    def _kill_starting(self, asked: list[PoolWorker]) -> list[PoolWorker]:
        """Kill every asked worker still starting Forge; those."""
        starting = []
        for worker in asked:
            if not past_forge_start(worker.mark):
                starting.append(worker)
                print(f"Killing worker {worker.slot} (PID {worker.process.pid}): "
                      "still starting Forge, no game to cut")
                self._kill(worker, starting=True)
        return starting

    def _wait_for_stops(self, asked: list[PoolWorker]) -> None:
        """Poll the workers asked to stop until none is left running and
        unkilled, killing each past its deadline, or every one on a second
        interrupt."""
        timeout = self._graceful.settings.timeout
        while True:
            waiting = self._books.still_waiting(asked)
            if not waiting:
                return
            now = time.monotonic()
            for worker, deadline in waiting:
                self._kill_if_due(worker, deadline=deadline, now=now, timeout=timeout)
            self._hurry.wait(timeout=self.STOP_POLL_SECONDS)

    def _kill_if_due(
        self, worker: PoolWorker, *, deadline: float, now: float, timeout: float,
    ) -> None:
        """Kill a worker asked to stop on a second interrupt, or once its
        ``deadline`` has passed."""
        pid = worker.process.pid
        if self._hurry.is_set():
            print(f"Killing worker PID {pid}: interrupted again")
            self._kill(worker)
        elif now >= deadline:
            print(
                f"Killing worker PID {pid}: it did not stop within "
                f"{_seconds(timeout)} of the request; its game is cut"
            )
            self._kill(worker)

    def _kill(self, worker: PoolWorker, *, starting: bool = False) -> None:
        """Kill one worker's tree, once: a second call for the same worker
        (the loop's shutdown, then ``run``'s ``finally``) does nothing, so a
        survivor's timed-out kill is not waited out twice. The single kill
        path of the pool. ``starting``: a worker still starting Forge, whose
        kill cuts no game (``killed_starting``)."""
        if self._books.claim_kill(worker, starting=starting):
            kill_process_tree(worker.process)

    def _handle_signal(self, signum, frame) -> None:
        """Signal handler: set shutdown event to stop workers. A signal while
        the pool already shuts down kills every worker still finishing its
        game. Under the protocol the workers run in process groups of their
        own (:func:`graceful_popen_options`), so a console Ctrl-C reaches
        this handler alone and the workers end only as it decides."""
        if self._shutdown_event.is_set():
            self._hurry.set()
        self._interrupted = True
        self._shutdown_event.set()
