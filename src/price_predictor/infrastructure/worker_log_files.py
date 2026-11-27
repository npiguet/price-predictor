"""The persistent, size-capped log file of each worker slot of a pool."""

from __future__ import annotations

from pathlib import Path
from typing import IO

from price_predictor.infrastructure.worker_stops import GracefulStop, WorkerStopSettings


class WorkerLogFiles:
    """Persistent, size-capped stdout/stderr destinations for a worker pool.

    Built for the two supervisors whose workers can trip one of this repo's
    five latched effect-record failure reporters (``ApiEvents.
    reportEmitterFailure`` and its four siblings) -- lines that must reach an
    operator, because a Forge upgrade that silently breaks an emitter would
    otherwise look identical to a mechanic sealed play never uses.
    ``MatchWorkerConnector.start`` already threads a ``log_file`` through to
    ``subprocess.Popen`` (mirroring ``evaluation_connector.py``'s own
    ``_spawn``); this class is what supplies one, since neither supervisor's
    ``spawn_worker`` callback otherwise carries state across
    :class:`ForgeWorkerPool`'s restarts -- it is called fresh, with only a
    worker index, every time a slot is (re)spawned.

    One file per worker slot, not per JVM lifetime: the pool restarts a slot
    many times over a long run (hundreds of JVM lifetimes over an
    eight-hour collection run, each recycled after ``--recycle-seconds`` --
    see ``ForgeWorkerPool``), and a reporter line from any one
    of them has to survive to the end of the run, so the file is opened once
    and reused across every restart of that slot rather than reopened per
    JVM -- the same "open once, reuse across restarts" shape
    ``evaluation_connector.py`` uses for its own per-split worker logs.

    Reuse is still bounded, because unlike a fixed number of validation
    splits, a collection run has no natural end: reused forever would let
    ordinary per-match chatter (a "N matches generated" line every ten
    matches, a full stack trace on each non-fatal match error) grow without
    limit across hundreds of restarts. Past ``max_bytes`` the current file is
    rotated to ``.1`` (replacing any previous one) and a fresh file opened,
    which bounds one slot to ``2 * max_bytes`` on disk instead of an
    open-ended append.
    """

    #: Twelve workers at two generations of the default cap is 480 MiB at the
    #: extreme -- a sliver of what one collection run's shards already
    #: occupy ("7,000 games is under a gigabyte" of compressed shards, per
    #: specs/023-ability-effect-model/quickstart.md) -- and the ordinary case
    #: (a handful of restarts' worth of startup banners and progress lines)
    #: uses far less.
    DEFAULT_MAX_BYTES = 20 * 1024 * 1024

    def __init__(
        self,
        directory: Path,
        run_id: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._directory = Path(directory)
        self._run_id = run_id
        self._max_bytes = max_bytes
        self._open: dict[int, IO[bytes]] = {}

    def path_for(self, worker_id: int) -> Path:
        """Where this worker's current log lives, whether or not it exists yet.

        ``{run_id}.{worker}.log`` in the directory this class was given,
        which every caller makes its run directory's ``logs/``
        (``.claude/skills/artifact-layout/SKILL.md``). One component short of
        a shard's own ``{run_id}.{worker}-{lifetime}``: the lifetime token is
        minted inside the JVM, so a file opened before spawning it can never
        carry one.
        """
        return self._directory / f"{self._run_id}.{worker_id}.log"

    def stop_path_for(self, worker_id: int) -> Path:
        """The file whose existence asks this slot's worker to stop at its
        game's end (``-Dworker.stop.file``): ``{run_id}.{worker}.stop`` beside
        the slot's log, under the run directory's ``logs/``."""
        return self._directory / f"{self._run_id}.{worker_id}.stop"

    def graceful_stop(self, settings: WorkerStopSettings) -> GracefulStop:
        """The stop protocol for a pool whose workers log here: each slot's
        stop file beside its log, its stop line read from that log."""
        return GracefulStop(
            stop_file_for=self.stop_path_for, log_path_for=self.path_for, settings=settings,
        )

    def get(self, worker_id: int) -> IO[bytes]:
        """The open handle for ``worker_id``, rotating first past the cap.

        The rotation degrades quietly (its own ``close()`` and rename are
        guarded: a log that stays over the cap is better than a worker
        lost). A log that cannot be opened raises :class:`OSError` naming the
        path, which fails the spawn: ``ForgeWorkerPool._monitor_worker`` counts
        it a failed start and backs off, so a slot never runs a worker whose
        startup cannot be seen.
        """
        path = self.path_for(worker_id)
        handle = self._open.get(worker_id)
        if handle is not None:
            try:
                too_big = path.stat().st_size > self._max_bytes
            except OSError:
                too_big = False
            if not too_big:
                return handle
            try:
                handle.close()
            except OSError:
                # Already unusable either way -- close() is documented to
                # release the underlying resource even when the OS call
                # itself fails -- so this is not worth degrading over; fall
                # through to reopen below.
                pass
            # Dropped before rotation is attempted, not after: a handle this
            # method has already closed must never be handed back out of
            # ``_open`` on some later call, which a raise between here and the
            # reassignment below would allow -- close() succeeding (or
            # failing; see above) is what makes the handle unusable, not
            # whether the rename that follows does.
            del self._open[worker_id]
            try:
                rotated = path.with_name(path.name + ".1")
                rotated.unlink(missing_ok=True)
                path.rename(rotated)
            except OSError as exc:
                # A logging failure must degrade quietly, never take down
                # collection (item 5): on Windows especially, a rename can
                # lose to a concurrent reader with the file still open. Falls
                # through to reopen ``path`` below either way: rotated or
                # not, it is where this worker's next lines belong.
                print(f"Could not rotate worker log {path}: {exc}")

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            handle = path.open("ab")
        except OSError as exc:
            # The directory could not be created, or the file could not be
            # opened (permissions, another process holding it). A worker with
            # no log cannot be told from one still starting Forge, so the
            # spawn fails: the pool counts it a failed start and retries
            # after a backoff.
            raise OSError(f"Could not open worker log {path}: {exc}") from exc
        self._open[worker_id] = handle
        return handle

    def close_all(self) -> None:
        """Close every handle opened so far. Safe to call more than once."""
        for handle in self._open.values():
            try:
                handle.close()
            except OSError:
                pass
        self._open.clear()
