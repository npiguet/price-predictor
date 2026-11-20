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
        status_interval: int | None = None,
        should_stop: Callable[[int], bool] | None = None,
        on_status: Callable[[], None] | None = None,
        max_failed_starts: int | None = None,
        restart_backoff: float | None = None,
        quick_exit_seconds: float | None = None,
        graceful: GracefulStop | None = None,
    ) -> None:
        """
        Args:
            worker_count: how many workers to keep alive.
            spawn_worker: called with a worker index, returns a started process.
                Under ``graceful`` it must give the worker
                ``graceful.system_properties(index)``.
            output_path: the file workers append to; its line count is progress.
            status_interval: seconds between status lines (default 60).
            should_stop: called with the current line count after each status
                line; returning True ends the run. None means run until signalled.
            on_status: called after each status line (a log scan); what it
                raises is printed and the run goes on.
            max_failed_starts, restart_backoff, quick_exit_seconds: the
                failed-start rule (class docstring); default the class constants.
            graceful: the workers' stop protocol; None kills them instead.
        """
        self._worker_count = worker_count
        self._spawn_worker = spawn_worker
        self._output_path = Path(output_path)
        self._status_interval = (
            self.STATUS_INTERVAL if status_interval is None else status_interval
        )
        self._should_stop = should_stop
        self._on_status = on_status
        self._graceful = graceful
        self._shutdown_event = threading.Event()
        #: Set by a second interrupt: every worker still asked to stop is
        #: killed at once.
        self._hurry = threading.Event()
        self._interrupted = False
        self._processes: list[subprocess.Popen] = []
        self._start_times: dict[subprocess.Popen, float] = {}
        self._processes_lock = threading.Lock()
        self._max_failed_starts = (
            self.MAX_FAILED_STARTS if max_failed_starts is None else max_failed_starts
        )
        self._restart_backoff = (
            self.RESTART_BACKOFF if restart_backoff is None else restart_backoff
        )
        self._quick_exit_seconds = (
            self.QUICK_EXIT_SECONDS if quick_exit_seconds is None else quick_exit_seconds
        )
        #: Processes the pool killed itself: their exit is no failure.
        self._killed: set[subprocess.Popen] = set()
        #: Of those, the ones killed at a stop while still starting Forge.
        self._killed_starting: set[subprocess.Popen] = set()
        self._failed_starts: dict[int, int] = {}
        self._start_failure: str | None = None
        #: Every worker the pool has killed or tried to, in order: each is
        #: killed once, however often the pool terminates.
        self._kill_attempted: list[subprocess.Popen] = []
        #: Under ``graceful``: each running worker's slot and where its log
        #: stood at its spawn, and when each worker was asked to stop and by
        #: when it must have ended.
        self._slots: dict[subprocess.Popen, int] = {}
        self._log_marks: dict[subprocess.Popen, LogMark] = {}
        self._deadlines: dict[subprocess.Popen, float] = {}
        #: How the workers ended so far.
        self._stops = WorkerStops()

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
        with self._processes_lock:
            attempted = list(self._kill_attempted)
        return [proc.pid for proc in attempted if proc.poll() is None]

    @property
    def stops(self) -> WorkerStops:
        """How the run's workers ended, by how, so far."""
        with self._processes_lock:
            return dataclasses.replace(self._stops)

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
            mark = self._prepare_slot(worker_id)
            try:
                proc = self._spawn_worker(worker_id)
            except Exception as exc:
                if self._shutdown_event.is_set():
                    return
                print(f"Monitor error for worker {worker_id}: {exc}")
                self._note_failed_start(worker_id, f"{type(exc).__name__}: {exc}")
                continue

            started = time.monotonic()
            bind_to_this_process(proc)
            with self._processes_lock:
                self._processes.append(proc)
                self._start_times[proc] = started
                self._slots[proc] = worker_id
                if mark is not None:
                    self._log_marks[proc] = mark
            # _terminate_all may have taken its snapshot before the append; a
            # worker spawned while the pool shuts down is killed here, or it
            # outlives the run. It is still starting Forge: no game is cut.
            if self._shutdown_event.is_set():
                self._kill(proc, starting=self._graceful is not None)

            try:
                proc.wait()
            except Exception as exc:
                print(f"Monitor error for worker {worker_id}: {exc}")

            lifetime = time.monotonic() - started
            ending, matches, games = self._ending(proc)
            # A worker that recycled itself played until its recycle time,
            # however short: it failed to start only if it played no game (a
            # best-of-N match spanning the recycle time finishes no match).
            failed = (
                games == 0 if ending == "recycled"
                else self._failed_start(proc, lifetime=lifetime)
            )
            with self._processes_lock:
                if proc in self._processes:
                    self._processes.remove(proc)
                # A killed worker that has exited is no survivor; holding its
                # Popen for the pool's life would hold a process handle per
                # recycle. One still running stays, to be reported.
                if proc.poll() is not None:
                    self._kill_attempted = [
                        attempted for attempted in self._kill_attempted
                        if attempted is not proc
                    ]
                self._start_times.pop(proc, None)
                self._killed.discard(proc)
                self._killed_starting.discard(proc)
                self._slots.pop(proc, None)
                self._log_marks.pop(proc, None)
                self._deadlines.pop(proc, None)
                if proc.poll() is not None:
                    setattr(self._stops, ending, getattr(self._stops, ending) + 1)

            if self._shutdown_event.is_set():
                return

            if ending == "recycled":
                print(
                    f"Worker {worker_id} recycled itself at a game end after "
                    f"{lifetime:.0f}s ({matches} matches, {games} games), restarting..."
                )
            else:
                print(f"Worker {worker_id} exited (code {proc.returncode}), restarting...")
            if failed:
                self._note_failed_start(
                    worker_id,
                    f"worker recycled itself {lifetime:.0f}s after starting without "
                    "playing a game (is --recycle-seconds shorter than a Forge start?)"
                    if ending == "recycled" else
                    f"worker exited (code {proc.returncode}) {lifetime:.0f}s "
                    "after starting",
                )
            else:
                with self._processes_lock:
                    self._failed_starts[worker_id] = 0

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

    def _ending(self, proc: subprocess.Popen) -> tuple[str, int | None, int | None]:
        """How an ended worker ended (a :class:`WorkerStops` field) and, for
        one that printed the stop line, the matches its JVM finished and the
        games it played.

        Exit 0 with the stop line since its spawn is a stop at a game end:
        ``stopped`` when the pool asked, ``recycled`` when it did not. Exit 0
        without the line, or any other code, is no stop: ``killed_starting``
        or ``killed`` when the pool killed it, ``ended_otherwise`` when the
        pool had asked, else ``crashed``."""
        with self._processes_lock:
            asked = proc in self._deadlines
            killed = proc in self._killed
            starting = proc in self._killed_starting
            mark = self._log_marks.get(proc)
        if getattr(proc, "returncode", None) == 0 and mark is not None:
            line = stop_line_of(mark.written_since(tail=_LOG_TAIL_BYTES))
            if line is not None:
                return (("stopped" if asked else "recycled"), int(line.group("matches")),
                        int(line.group("games")))
        if killed:
            return ("killed_starting" if starting else "killed"), None, None
        return ("ended_otherwise" if asked else "crashed"), None, None

    def _failed_start(self, proc: subprocess.Popen, *, lifetime: float) -> bool:
        """Whether a worker that exited after ``lifetime`` seconds failed to
        start: it exited that soon and the pool did not kill it."""
        with self._processes_lock:
            killed = proc in self._killed
        return not killed and lifetime < self._quick_exit_seconds

    def _note_failed_start(self, worker_id: int, reason: str) -> None:
        """Count one failed start of ``worker_id``. Once every slot has failed
        ``max_failed_starts`` in a row the run ends; otherwise wait out the
        backoff."""
        with self._processes_lock:
            failures = self._failed_starts.get(worker_id, 0) + 1
            self._failed_starts[worker_id] = failures
            every_slot_failed = all(
                self._failed_starts.get(slot, 0) >= self._max_failed_starts
                for slot in range(self._worker_count)
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
        last_count = baseline
        last_time = start_time
        next_status = start_time + self._status_interval

        while not self._shutdown_event.is_set():
            if self._should_stop is not None and self._should_stop(
                self.output_line_count()
            ):
                self._shutdown_event.set()
                break

            timeout = self._sleep_slice(next_status)
            self._shutdown_event.wait(timeout=timeout)
            if self._shutdown_event.is_set():
                break

            now = time.monotonic()
            if now < next_status:
                continue  # a stop-poll slice, not a status tick

            last_count, last_time = self._report_status(
                start_time, now, last_count, last_time, baseline,
            )
            if self._on_status is not None:
                try:
                    self._on_status()
                except Exception as exc:  # noqa: BLE001 — a report never stops a run
                    print(f"Status report failed: {exc}")
            next_status = now + self._status_interval
            if self._should_stop is not None and self._should_stop(last_count):
                self._shutdown_event.set()
                break
            if self._graceful is not None:
                self._kill_overdue_workers()
            self._kill_oldest_worker()

        if self._graceful is None:
            print(f"Shutting down, terminating {self._worker_count} workers...")
        else:
            print(
                f"Shutting down, asking {self._worker_count} workers to stop at "
                "their game's end..."
            )
        self._terminate_all()

    def _sleep_slice(self, next_status: float) -> float:
        """How long to wait next: the stop-poll slice, or the rest of the interval."""
        remaining = next_status - time.monotonic()
        if self._should_stop is None:
            return max(0.0, remaining)
        return max(0.0, min(float(self.STOP_POLL_INTERVAL), remaining))

    def _report_status(
        self,
        start_time: float,
        now: float,
        last_count: int,
        last_time: float,
        baseline: int = 0,
    ) -> tuple[int, float]:
        """Print one status line and return the file's line count and ``now``.

        ``baseline`` is the line count when the run started; the line reports
        the matches completed since, while the returned count stays the whole
        file's, which is what ``should_stop`` is handed.
        """
        elapsed = now - start_time
        interval = now - last_time

        count = self.output_line_count()
        delta = count - last_count
        rate = delta / interval * 60 if interval > 0 else 0.0

        with self._processes_lock:
            alive = sum(1 for p in self._processes if p.poll() is None)

        print(
            f"[{elapsed:.0f}s] {count - baseline} matches completed"
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
        long any single match can occupy a worker. Introduced in commit
        eda4505.

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
        small ``-Xmx`` (see MatchWorkerConnector.start): Forge degrades late in a
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
        with self._processes_lock:
            alive = [(p, t) for p, t in self._start_times.items() if p.poll() is None]
            if not alive:
                return
            oldest_proc, oldest_time = min(alive, key=lambda x: x[1])
            age = time.monotonic() - oldest_time
            self._killed.add(oldest_proc)

        self._kill(oldest_proc)
        print(f"Recycled longest-running worker (PID {oldest_proc.pid}, age {age:.0f}s)")

    def _kill_oldest_playing_worker(self) -> None:
        """:meth:`_kill_oldest_worker` under the stop protocol: the oldest
        worker past its Forge start, neither asked to stop nor already killed."""
        if self._shutdown_event.is_set():
            return
        with self._processes_lock:
            candidates = sorted(
                (
                    (started, proc, self._slots.get(proc, -1), self._log_marks.get(proc))
                    for proc, started in self._start_times.items()
                    if proc.poll() is None and proc not in self._deadlines
                    and not any(proc is attempted for attempted in self._kill_attempted)
                ),
                key=lambda candidate: candidate[0],
            )
        for started, proc, slot, mark in candidates:
            if not past_forge_start(mark):
                continue
            age = time.monotonic() - started
            self._kill(proc)
            print(
                f"Recycled longest-running worker {slot} (PID {proc.pid}, age "
                f"{age:.0f}s); the game it was playing is cut (no game-end "
                "marker: build-corpus drops it)"
            )
            return

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
        with self._processes_lock:
            overdue = [
                (proc, self._slots.get(proc, -1), now - started)
                for proc, started in self._start_times.items()
                if proc.poll() is None and now - started > limit
                and proc not in self._deadlines
                and not any(proc is attempted for attempted in self._kill_attempted)
            ]
        for proc, slot, age in overdue:
            print(
                f"Killing worker {slot} (PID {proc.pid}), {age:.0f}s old: its game "
                f"has not ended {_seconds(settings.timeout)} past its "
                f"{_seconds(settings.recycle_seconds)} recycle time; the game it "
                "was playing is cut (no game-end marker: build-corpus drops it)"
            )
            self._kill(proc)

    def _terminate_all(self) -> None:
        """End every running worker: without the protocol, kill it; with it,
        ask it to stop at its game's end and wait (:meth:`_stop_all`)."""
        with self._processes_lock:
            procs = list(self._processes)

        if self._graceful is None:
            for proc in procs:
                self._kill(proc)
            return
        self._stop_all(procs)

    def _stop_all(self, procs: list[subprocess.Popen]) -> None:
        """Ask every running worker of ``procs`` to stop at its game's end,
        then wait until each has ended, killing one ``timeout`` seconds after
        the request. A worker whose log shows it still starting Forge (no
        startup line since its spawn) is killed at once, counted as
        ``killed_starting``: its stop file is created first, so one that
        prints its startup line meanwhile stops before its first match, and
        none has a game to cut. One that has already ended is not waited
        for; one asked before keeps its deadline, so a second call waits out
        no timeout twice; a second interrupt kills at once."""
        settings = self._graceful.settings
        now = time.monotonic()
        asked: list[tuple[subprocess.Popen, int]] = []
        with self._processes_lock:
            for proc in procs:
                if proc.poll() is not None or proc in self._deadlines:
                    continue
                self._deadlines[proc] = now + settings.timeout
                asked.append((proc, self._slots.get(proc, -1)))
        for proc, slot in asked:
            stop_file = Path(self._graceful.stop_file_for(slot))
            try:
                stop_file.parent.mkdir(parents=True, exist_ok=True)
                stop_file.touch()
            except OSError as exc:
                print(f"Could not ask worker PID {proc.pid} to stop ({stop_file}: {exc}); "
                      "killing it")
                self._kill(proc)
        starting = []
        for proc, slot in asked:
            with self._processes_lock:
                mark = self._log_marks.get(proc)
            if not past_forge_start(mark):
                starting.append(proc)
                print(f"Killing worker {slot} (PID {proc.pid}): still starting Forge, "
                      "no game to cut")
                self._kill(proc, starting=True)
        if len(asked) > len(starting):
            print(
                f"Asked {len(asked) - len(starting)} worker(s) to stop at their game's "
                f"end; each is killed {_seconds(settings.timeout)} after the request"
            )
        while True:
            with self._processes_lock:
                waiting = [
                    (proc, deadline) for proc, deadline in self._deadlines.items()
                    if proc.poll() is None
                    and not any(proc is attempted for attempted in self._kill_attempted)
                ]
            if not waiting:
                return
            now = time.monotonic()
            for proc, deadline in waiting:
                if self._hurry.is_set():
                    print(f"Killing worker PID {proc.pid}: interrupted again")
                    self._kill(proc)
                elif now >= deadline:
                    print(
                        f"Killing worker PID {proc.pid}: it did not stop within "
                        f"{_seconds(settings.timeout)} of the request; its game is cut"
                    )
                    self._kill(proc)
            self._hurry.wait(timeout=self.STOP_POLL_SECONDS)

    def _kill(self, proc: subprocess.Popen, *, starting: bool = False) -> None:
        """Kill one worker's tree, once: a second call for the same worker
        (the loop's shutdown, then ``run``'s ``finally``) does nothing, so a
        survivor's timed-out kill is not waited out twice. The single kill
        path of the pool. ``starting``: a worker still starting Forge, whose
        kill cuts no game (``killed_starting``)."""
        with self._processes_lock:
            if any(proc is attempted for attempted in self._kill_attempted):
                return
            self._kill_attempted.append(proc)
            self._killed.add(proc)
            if starting:
                self._killed_starting.add(proc)
        kill_process_tree(proc)

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
