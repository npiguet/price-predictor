"""MatchOutcomeSupervisor: spawns and monitors Java worker subprocesses."""

from __future__ import annotations

import subprocess
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool, WorkerLogFiles
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
        effect_records_dir: Path | None = None,
        collection_caps: Mapping[str, object] | None = None,
        exclude_cards_path: Path | None = None,
    ) -> None:
        self._worker_count = worker_count
        self._output_path = output_path
        self._best_of = best_of
        self._side_a_decks_path = side_a_decks_path
        self._side_b_decks_path = side_b_decks_path
        self._side_b_decks_weight = side_b_decks_weight
        # Absent, nothing about this run changes: no shard is opened and the
        # two sealed corpora keep their exact format and content.
        self._effect_records_dir = effect_records_dir
        # Only meaningful alongside the shard directory; the worker falls back
        # to its own defaults for anything absent.
        self._collection_caps = collection_caps
        self._exclude_cards_path = exclude_cards_path
        self._run_id = str(uuid.uuid4())
        self._connector = MatchWorkerConnector()
        # Only constructed when a shard directory is: a plain match-outcomes
        # run keeps _start_worker's original discard-everything behaviour, and
        # never needs anything else, because Forge only calls the five latched
        # effect-record failure reporters (ApiEvents.reportEmitterFailure and
        # its siblings) behind EffectRecordOutcomes.isObserved() -- true only
        # once effect-record collection is on (F4).
        self._worker_logs = (
            WorkerLogFiles(effect_records_dir, self._run_id)
            if effect_records_dir is not None else None
        )
        # The lambda re-reads self._start_worker per spawn so tests (and any
        # caller) can patch it after construction.
        self._pool = ForgeWorkerPool(
            worker_count=worker_count,
            spawn_worker=lambda worker_id: self._start_worker(worker_id),
            output_path=output_path,
        )

    @property
    def run_id(self) -> str:
        """UUID generated once at construction, shared across all worker restarts.

        It namespaces a run, not a process: the pool recycles the
        longest-running worker every status interval and restarts crashed ones,
        so the JVM behind a slot is replaced hundreds of times while this value
        and the slot index both stay put. Whatever a worker counts from zero
        therefore has to name its own JVM lifetime — the effect-record shard
        writer does, with a token it mints for itself.
        """
        return self._run_id

    def run(self) -> None:
        """Start all workers and block until shutdown."""
        try:
            self._pool.run()
        finally:
            if self._worker_logs is not None:
                self._worker_logs.close_all()

    def _start_worker(self, worker_id: int) -> subprocess.Popen:
        """Start one Java worker subprocess.

        Without ``--effect-records``, stdout/stderr are discarded — the
        supervisor's own status reports are the only operator-facing output,
        and Forge is verbose enough that capturing per-worker logs (with
        concurrent appenders + AV scanning) becomes a measurable I/O
        bottleneck on long runs. A run collecting effect records is
        different: it is the one case where a worker can trip one of this
        branch's five latched effect-record failure reporters, so it gets a
        real, size-capped destination instead (F4; see
        ``price_predictor.infrastructure.forge_jvm.WorkerLogFiles``).
        """
        log_file = (
            self._worker_logs.get(worker_id) if self._worker_logs is not None
            else None
        )
        proc = self._connector.start(
            self._output_path,
            run_id=self._run_id,
            best_of=self._best_of,
            log_file=log_file,
            side_a_decks_path=self._side_a_decks_path,
            side_b_decks_path=self._side_b_decks_path,
            side_b_decks_weight=self._side_b_decks_weight,
            effect_records_dir=self._effect_records_dir,
            # The slot, not the process: a respawn passes the same number, so
            # this cannot be what makes a restarted worker's ids distinct.
            worker_index=worker_id,
            collection_caps=self._collection_caps,
            exclude_cards_path=self._exclude_cards_path,
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
