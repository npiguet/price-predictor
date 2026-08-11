"""MatchOutcomeSupervisor: spawns and monitors Java worker subprocesses."""

from __future__ import annotations

import subprocess
import threading
import uuid
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool
from sealed.infrastructure.match_worker_connector import (
    DEFAULT_SIDE_B_DECKS_WEIGHT,
    MatchWorkerConnector,
)


class MatchOutcomeSupervisor:
    """Manages a pool of Java MatchWorkerMain subprocesses.

    Spawns worker_count workers, monitors each in a dedicated thread, restarts
    crashed workers, reports status every 60 seconds, and handles clean shutdown
    on SIGINT/SIGTERM.

    Those mechanics live in :class:`ForgeWorkerPool` and are shared with
    ``draft play-draft-games``; this class owns what is sealed-specific — the run
    id and the worker command's side-deck arguments.
    """

    STATUS_INTERVAL = ForgeWorkerPool.STATUS_INTERVAL  # seconds between status reports

    def __init__(
        self,
        worker_count: int,
        output_path: Path,
        best_of: int,
        side_a_decks_path: Path | None = None,
        side_b_decks_path: Path | None = None,
        side_b_decks_weight: int = DEFAULT_SIDE_B_DECKS_WEIGHT,
    ) -> None:
        self._worker_count = worker_count
        self._output_path = output_path
        self._best_of = best_of
        self._side_a_decks_path = side_a_decks_path
        self._side_b_decks_path = side_b_decks_path
        self._side_b_decks_weight = side_b_decks_weight
        self._run_id = str(uuid.uuid4())
        self._connector = MatchWorkerConnector()
        # The lambda re-reads self._start_worker per spawn so tests (and any
        # caller) can patch it after construction.
        self._pool = ForgeWorkerPool(
            worker_count=worker_count,
            spawn_worker=lambda worker_id: self._start_worker(worker_id),
            output_path=output_path,
        )

    @property
    def run_id(self) -> str:
        """UUID generated once at construction, shared across all worker restarts."""
        return self._run_id

    def run(self) -> None:
        """Start all workers and block until shutdown."""
        self._pool.run()

    def _start_worker(self, worker_id: int) -> subprocess.Popen:
        """Start one Java worker subprocess. Worker stdout/stderr are discarded —
        the supervisor's own status reports are the only operator-facing output.
        Forge is verbose enough that capturing per-worker logs (with concurrent
        appenders + AV scanning) becomes a measurable I/O bottleneck on long runs.
        """
        proc = self._connector.start(
            self._output_path,
            run_id=self._run_id,
            best_of=self._best_of,
            side_a_decks_path=self._side_a_decks_path,
            side_b_decks_path=self._side_b_decks_path,
            side_b_decks_weight=self._side_b_decks_weight,
        )
        print(f"Worker {worker_id} started (PID {proc.pid})")
        return proc

    # ── Pool state, exposed under the names this class published before the
    # extraction so callers and tests keep working. ──────────────────────────

    @property
    def _processes(self) -> list[subprocess.Popen]:
        return self._pool._processes

    @property
    def _start_times(self) -> dict[subprocess.Popen, float]:
        return self._pool._start_times

    @property
    def _processes_lock(self) -> threading.Lock:
        return self._pool._processes_lock

    @property
    def _shutdown_event(self) -> threading.Event:
        return self._pool._shutdown_event

    def _kill_oldest_worker(self) -> None:
        self._pool._kill_oldest_worker()

    def _count_output_lines(self) -> int:
        return self._pool.output_line_count()
