"""The books a ``ForgeWorkerPool`` keeps: each running worker
(:class:`PoolWorker`), every process it killed or tried to, how its workers
ended and each slot's failed starts, all under one lock (:class:`WorkerBooks`).

Every method of :class:`WorkerBooks` is one critical section: the pool's
threads (the supervisor loop, one monitor per slot, the signal handler) read
and change the books only through them."""

from __future__ import annotations

import dataclasses
import subprocess
import threading
import time

from price_predictor.infrastructure.worker_log_mark import LogMark
from price_predictor.infrastructure.worker_stops import WorkerStops


@dataclasses.dataclass(eq=False)
class PoolWorker:
    """One running worker, kept from its spawn until its monitor sees it
    end. ``process``, ``slot``, ``started`` and ``mark`` are set at the spawn
    and never change; the other fields change, and are read, only under the
    books' lock. Compared by identity.

    Attributes:
        process: the worker's JVM.
        slot: the pool slot it runs in.
        started: when it was spawned (``time.monotonic``).
        mark: under the stop protocol, where its slot's log stood at its spawn.
        deadline: once it is asked to stop, by when it must have ended.
        killed: the pool killed it, so its exit is no failure.
        killed_starting: it was killed at a stop while still starting Forge.
    """

    process: subprocess.Popen
    slot: int
    started: float
    mark: LogMark | None = None
    deadline: float | None = None
    killed: bool = False
    killed_starting: bool = False


class WorkerBooks:
    """A pool's workers and what it did to them, under :attr:`lock`.

    Attributes:
        workers: every running worker, by its process, in spawn order.
        kill_attempted: every worker process the pool killed or tried to, in
            order: each is killed once, however often the pool terminates.
            One still running when its monitor sees it end stays, a survivor.
        stops: how the workers ended so far.
        failed_starts: each slot's failed starts in a row.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.workers: dict[subprocess.Popen, PoolWorker] = {}
        self.kill_attempted: list[subprocess.Popen] = []
        self.stops = WorkerStops()
        self.failed_starts: dict[int, int] = {}

    def track(
        self, proc: subprocess.Popen, *, slot: int, started: float,
        mark: LogMark | None = None,
    ) -> PoolWorker:
        """Keep a spawned worker until its monitor sees it end."""
        worker = PoolWorker(proc, slot=slot, started=started, mark=mark)
        with self.lock:
            self.workers[proc] = worker
        return worker

    def drop(self, worker: PoolWorker, ending: str) -> None:
        """Forget an ended worker and count how it ended (a
        :class:`WorkerStops` field). A killed worker that has exited is no
        survivor; holding its Popen for the pool's life would hold a process
        handle per recycle. One still running stays, to be reported."""
        proc = worker.process
        with self.lock:
            self.workers.pop(proc, None)
            if proc.poll() is not None:
                self.kill_attempted = [
                    attempted for attempted in self.kill_attempted if attempted is not proc
                ]
            # No stop waits for it any more, unless a stop that took its
            # snapshot earlier asks it again.
            worker.deadline = None
            if proc.poll() is not None:
                setattr(self.stops, ending, getattr(self.stops, ending) + 1)

    def running(self) -> list[PoolWorker]:
        """Every running worker, in spawn order."""
        with self.lock:
            return list(self.workers.values())

    def alive_count(self) -> int:
        """How many running workers' processes are alive."""
        with self.lock:
            return sum(1 for p in self.workers if p.poll() is None)

    def survivors(self) -> list[int]:
        """The PIDs of the processes the pool killed or tried to that are
        still running; asked of each outside the lock."""
        with self.lock:
            attempted = list(self.kill_attempted)
        return [proc.pid for proc in attempted if proc.poll() is None]

    def stops_so_far(self) -> WorkerStops:
        """A copy of how the workers ended so far."""
        with self.lock:
            return dataclasses.replace(self.stops)

    def kills_of(self, worker: PoolWorker) -> tuple[bool, bool, bool]:
        """Whether ``worker`` was asked to stop, killed, and killed while
        starting Forge."""
        with self.lock:
            return worker.deadline is not None, worker.killed, worker.killed_starting

    def was_killed(self, proc: subprocess.Popen) -> bool:
        """Whether the pool killed the running worker of ``proc``."""
        with self.lock:
            worker = self.workers.get(proc)
            return worker is not None and worker.killed

    def oldest_alive(self) -> tuple[PoolWorker, float] | None:
        """The oldest worker whose process is alive and its age, marked
        killed; None when no process is alive."""
        with self.lock:
            alive = [w for w in self.workers.values() if w.process.poll() is None]
            if not alive:
                return None
            oldest = min(alive, key=lambda w: w.started)
            age = time.monotonic() - oldest.started
            oldest.killed = True
        return oldest, age

    def recyclable(self) -> list[PoolWorker]:
        """The running workers whose process is alive, neither asked to stop
        nor already killed, in spawn order."""
        with self.lock:
            return [
                w for w in self.workers.values()
                if w.process.poll() is None and w.deadline is None
                and not self._kill_was_attempted(w.process)
            ]

    def ask_to_stop(self, workers: list[PoolWorker], *, deadline: float) -> list[PoolWorker]:
        """Give each worker of ``workers`` alive and not yet asked its
        ``deadline``; those."""
        asked: list[PoolWorker] = []
        with self.lock:
            for worker in workers:
                if worker.process.poll() is not None or worker.deadline is not None:
                    continue
                worker.deadline = deadline
                asked.append(worker)
        return asked

    def still_waiting(self, asked: list[PoolWorker]) -> list[tuple[PoolWorker, float]]:
        """Every worker asked to stop, by an earlier stop or this one
        (``asked``), that is still alive and not yet killed, with its
        deadline, in the order they were asked. A worker its monitor saw end
        before this stop asked it is waited for too."""
        with self.lock:
            earlier = [w for w in self.workers.values()
                       if w.deadline is not None and w not in asked]
            return [
                (w, w.deadline) for w in [*earlier, *asked]
                if w.deadline is not None and w.process.poll() is None
                and not self._kill_was_attempted(w.process)
            ]

    def claim_kill(self, worker: PoolWorker, *, starting: bool) -> bool:
        """Record a kill of ``worker`` (``starting``: while it starts Forge);
        False when its process was killed or tried before."""
        with self.lock:
            if self._kill_was_attempted(worker.process):
                return False
            self.kill_attempted.append(worker.process)
            worker.killed = True
            if starting:
                worker.killed_starting = True
        return True

    def note_failed_start(self, slot: int, *, limit: int, slots: int) -> tuple[int, bool]:
        """Count one failed start of ``slot``: its failures in a row, and
        whether every one of ``slots`` slots has failed ``limit`` in a row."""
        with self.lock:
            failures = self.failed_starts.get(slot, 0) + 1
            self.failed_starts[slot] = failures
            every_slot_failed = all(
                self.failed_starts.get(other, 0) >= limit for other in range(slots)
            )
        return failures, every_slot_failed

    def clear_failed_starts(self, slot: int) -> None:
        """A worker of ``slot`` started: its failures in a row are none."""
        with self.lock:
            self.failed_starts[slot] = 0

    def _kill_was_attempted(self, proc: subprocess.Popen) -> bool:
        """Under the lock: whether the pool has killed ``proc`` or tried to."""
        return any(proc is attempted for attempted in self.kill_attempted)
