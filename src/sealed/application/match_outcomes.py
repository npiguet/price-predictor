"""MatchOutcomeSupervisor: spawns and monitors Java worker subprocesses."""

from __future__ import annotations

import subprocess
import threading
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from price_predictor.domain.worker_failures import WorkerFailure
from price_predictor.domain.worker_status import (
    WorkerEngineRefused,
    render_fork_refusals,
    render_hook_failures,
)
from price_predictor.infrastructure.append_only import has_shard_record
from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool, PoolOptions
from price_predictor.infrastructure.run_manifest import annotate_run
from price_predictor.infrastructure.worker_log_files import WorkerLogFiles
from price_predictor.infrastructure.worker_log_scan import WorkerLogReport
from price_predictor.infrastructure.worker_stops import WorkerStopSettings
from sealed.domain.directed_fork_needs import DirectedForkNeeds
from sealed.infrastructure.effect_collection_flags import (
    COUNT_DROPPED_EVENTS_PROPERTY,
    caps_manifest,
    directed_rule_probability_properties,
    dropped_events_report,
)
from sealed.infrastructure.match_worker_connector import (
    MatchWorkerConnector,
    WorkerDecks,
    WorkerLaunch,
    WorkerRecords,
    WorkerStopProtocol,
)

#: Under the run directory: every worker log.
LOGS_DIR = "logs"


@dataclass(frozen=True)
class MatchRunOptions:
    """What a match-outcomes run adds to its worker count, output file and
    best-of: where the decks come from (``decks``: the side-A and side-B decks
    files, the side-B weight and the excluded cards), and, with
    ``effect_records_dir``, the effect-record collection: its ``collection_caps``
    (the worker falls back to its own defaults for anything absent), the
    ``worker_stop`` settings (the defaults when None) and ``allow_degraded``."""

    decks: WorkerDecks = WorkerDecks()
    effect_records_dir: Path | None = None
    collection_caps: Mapping[str, object] | None = None
    worker_stop: WorkerStopSettings | None = None
    allow_degraded: bool = False


class MatchOutcomeSupervisor:
    """Manages a pool of Java MatchWorkerMain subprocesses.

    Spawns worker_count workers, monitors each in a dedicated thread, restarts
    crashed workers, reports status every 60 seconds, and handles clean shutdown
    on SIGINT/SIGTERM.

    With effect records the workers honour the stop protocol
    (``contracts/cli.md`` § Match worker stop protocol, ``worker_stop``): the
    run's end asks each to stop at its game's end and waits for it, killing
    one only past the stop timeout, and a worker recycles itself at a game end.
    Without, the workers have no log to print their stop line to, and the pool
    recycles and ends them by the kill, as it always did: there is no shard
    whose games a kill could cut.

    With effect records, every log scan also judges the workers' engine from
    their status lines: one not ``patched`` with every hook ends the run by
    the stop protocol and :meth:`run` raises :class:`WorkerEngineRefused`,
    unless ``allow_degraded``; an engine that changes under the run always
    does (``worker_status.EngineVerdict``). Without, the workers keep no log
    and nothing is judged.

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
        options: MatchRunOptions = MatchRunOptions(),
    ) -> None:
        effect_records_dir = options.effect_records_dir
        collection_caps = options.collection_caps
        self._worker_count = worker_count
        self._output_path = output_path
        self._best_of = best_of
        self._decks = options.decks
        # Absent, nothing about this run changes: no shard is opened and the
        # two sealed corpora keep their exact format and content.
        self._effect_records_dir = effect_records_dir
        # Only meaningful alongside the shard directory; the worker falls back
        # to its own defaults for anything absent.
        self._collection_caps = collection_caps
        #: Set by :meth:`steer_directed_forks`; without, every worker keeps the
        #: caps' directed-fork probabilities.
        self._directed_needs: DirectedForkNeeds | None = None
        #: The listed rules' own probabilities the next worker starts with,
        #: replaced whole at each log scan (read by the pool's monitor threads).
        self._directed_probabilities: dict[str, float] = (
            caps_manifest(collection_caps)["directed_rule_probabilities"]
            if collection_caps is not None else {}
        )
        self._run_id = str(uuid.uuid4())
        self._connector = MatchWorkerConnector()
        # Only constructed when a shard directory is: a plain match-outcomes
        # run keeps _start_worker's original discard-everything behaviour, and
        # never needs anything else, because Forge only calls the five latched
        # effect-record failure reporters (ApiEvents.reportEmitterFailure and
        # its siblings) behind EffectRecordOutcomes.isObserved() -- true only
        # once effect-record collection is on (F4).
        # Under the run directory's logs/ (.claude/skills/artifact-layout),
        # scanned for failure lines at every status line and at the end.
        self._worker_logs = (
            WorkerLogFiles(Path(effect_records_dir) / LOGS_DIR, self._run_id)
            if effect_records_dir is not None else None
        )
        self._log_report = (
            WorkerLogReport(
                self._worker_logs, worker_count, allow_degraded=options.allow_degraded)
            if self._worker_logs is not None else None
        )
        self._worker_stop = (
            (options.worker_stop if options.worker_stop is not None else WorkerStopSettings())
            if self._worker_logs is not None else None
        )
        #: Under the run directory's logs/: one line per match any worker
        #: completed, a random-seat match (which writes no outcome row) included.
        self._progress_path = (
            Path(effect_records_dir) / LOGS_DIR / f"{self._run_id}.progress.txt"
            if effect_records_dir is not None else None
        )
        #: Matches this run completed (outcome rows), every match it played to
        #: the end (progress lines), and whether it wrote a record, once it has run.
        self._matches = 0
        self._matches_played = 0
        self._wrote_records = False
        # The lambda re-reads self._start_worker per spawn so tests (and any
        # caller) can patch it after construction.
        self._pool = ForgeWorkerPool(
            worker_count=worker_count,
            spawn_worker=lambda worker_id: self._start_worker(worker_id),
            output_path=output_path,
            options=PoolOptions(
                on_status=self._report_worker_logs if self._log_report is not None else None,
                graceful=(
                    self._worker_logs.graceful_stop(self._worker_stop)
                    if self._worker_stop is not None else None
                ),
            ),
        )

    @property
    def run_id(self) -> str:
        """UUID generated once at construction, shared across all worker restarts.

        It namespaces a run, not a process: the pool recycles its workers
        (the longest-running one killed every status interval and, with
        effect records, each stopping itself at a game end after
        ``--recycle-seconds``) and restarts crashed ones, so the JVM behind a slot is
        replaced hundreds of times while this value
        and the slot index both stay put. Whatever a worker counts from zero
        therefore has to name its own JVM lifetime — the effect-record shard
        writer does, with a token it mints for itself.
        """
        return self._run_id

    @property
    def worker_failures(self):
        """Every failure line the run's workers logged, by kind (none
        without ``--effect-records``)."""
        from price_predictor.domain.worker_failures import WorkerFailures

        return self._log_report.totals if self._log_report is not None else WorkerFailures()

    @property
    def collected_nothing(self) -> bool:
        """Whether a run collecting effect records played matches and wrote
        no record: every match failing to record, which the shards alone
        cannot tell from a short run. A match played is one with an outcome
        row, one the progress file counts (a random-seat match writes no row)
        or a random-seat match that threw, so a run of random-seat matches
        alone is judged too."""
        played = max(self._matches, self._matches_played) + self.worker_failures.counts[
            WorkerFailure.RANDOM_SEAT_MATCH_ERROR.value
        ]
        return self._effect_records_dir is not None and played > 0 and not self._wrote_records

    def steer_directed_forks(self, needs: DirectedForkNeeds | None) -> None:
        """Recompute each rule of ``needs`` at every log scan from the forks
        the workers kept, and start every later worker with the result, as
        ``collect-variants`` does between its rounds (finding 6.1); None
        leaves every worker the caps' probabilities."""
        if needs is None:
            return
        self._directed_needs = needs
        self._directed_probabilities = needs.probabilities()

    def run(self) -> None:
        """Start all workers and block until shutdown, then report the
        workers' failure lines and print the dropped-event table when the run
        counted dropped events (FR-041). Raises :class:`WorkerEngineRefused`
        once the run ended for its workers' engine."""
        baseline = self._pool.output_line_count()
        self._record_caps()
        try:
            self._pool.run()
        finally:
            if self._worker_logs is not None:
                self._worker_logs.close_all()
            self._matches = self._pool.output_line_count() - baseline
            if self._progress_path is not None and self._progress_path.exists():
                with open(self._progress_path, "rb") as progress:
                    self._matches_played = sum(1 for _ in progress)
            if self._effect_records_dir is not None:
                self._finish_effect_records()
        self._raise_engine_refusal()
        if self._counts_dropped_events():
            print(dropped_events_report(self._effect_records_dir, self._run_id))

    def _record_caps(self) -> None:
        """The resolved caps into ``run.json`` before any worker starts, as
        the collectors record theirs, so an interrupted run has them."""
        if self._effect_records_dir is not None and self._collection_caps is not None:
            annotate_run(
                self._effect_records_dir, run_id=self._run_id,
                caps=caps_manifest(self._collection_caps),
            )

    def _report_worker_logs(self) -> None:
        """Print the failure lines the workers logged since the last look."""
        found = self._log_report.scan()
        if found:
            print(f"Worker logs: {found.render()} (under {self._worker_logs_dir})")
        self._review_engine()
        if self._directed_needs is not None:
            self._recompute_directed_probabilities()

    def _recompute_directed_probabilities(self) -> None:
        """The probabilities the kept forks summed at this scan give; the
        rules whose probability moved are printed."""
        before = self._directed_probabilities
        after = self._directed_needs.probabilities(self._log_report.directed_fork_tally())
        moved = [f"{rule} {p:.2f}" for rule, p in after.items() if before.get(rule) != p]
        self._directed_probabilities = after
        if moved:
            print(f"Directed fork probabilities for the next workers: {', '.join(moved)}")

    def _review_engine(self) -> None:
        """Judge the workers' engine: warn once under ``--allow-degraded``; on
        a refusal, say so and ask the pool to stop every worker at its game's
        end, as a first Ctrl-C does."""
        refused_before = self._log_report.engine_refusal is not None
        warning = self._log_report.review_engine()
        if warning is not None:
            print(f"Warning: {warning}")
        refusal = self._log_report.engine_refusal
        if refusal is not None and not refused_before:
            print(f"Stopping the run: {refusal}")
            self._pool.request_shutdown()

    def _raise_engine_refusal(self) -> None:
        """Raise :class:`WorkerEngineRefused` when the run ended for its engine."""
        refusal = self._log_report.engine_refusal if self._log_report is not None else None
        if refusal is not None:
            raise WorkerEngineRefused(
                f"{refusal}. The records written before the stop stay under "
                f"{self._effect_records_dir}"
            )

    @property
    def _worker_logs_dir(self) -> Path:
        return Path(self._effect_records_dir) / LOGS_DIR

    def _finish_effect_records(self) -> None:
        """The run's worker-log totals and record count, printed and kept in
        ``run.json``; an error line when matches recorded nothing."""
        self._report_worker_logs()
        self._wrote_records = has_shard_record(self._effect_records_dir, self._run_id)
        print(f"Worker logs over the run: {self._log_report.totals.render()}")
        hooks = self._log_report.hook_failures()
        if any(hooks.values()):
            print(f"Effect-record hook failures over the run: {render_hook_failures(hooks)}")
        refusals = self._log_report.fork_refusals()
        if refusals:
            print("Fork refusals over the run (expected, no failure): "
                  f"{render_fork_refusals(refusals)}")
        survivors = self._pool.survivors
        stops = self._pool.stops
        print(f"Workers: {stops.render()}, {len(survivors)} outlived the kill")
        annotate_run(
            self._effect_records_dir, run_id=self._run_id,
            match_outcomes=str(self._output_path),
            matches=self._matches,
            matches_played=self._matches_played,
            **self._log_report.manifest_fields(),
            worker_survivors=survivors,
            worker_stops=stops.as_dict(survived=len(survivors)),
            **self._worker_stop.manifest_fields(),
            final_directed_rule_probabilities=self._directed_probabilities,
        )
        if self.collected_nothing:
            print(
                f"Error: {max(self._matches, self._matches_played)} match(es) "
                f"played and no effect record was written under "
                f"{self._effect_records_dir}; the worker logs under "
                f"{self._worker_logs_dir} say why"
            )

    def _counts_dropped_events(self) -> bool:
        """Whether the workers were told to count, read off what they were sent."""
        return (
            self._effect_records_dir is not None
            and self._collection_caps is not None
            and str(self._collection_caps.get(COUNT_DROPPED_EVENTS_PROPERTY)) == "true"
        )

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
        ``price_predictor.infrastructure.worker_log_files.WorkerLogFiles``).
        """
        log_file = (
            self._worker_logs.get(worker_id) if self._worker_logs is not None
            else None
        )
        proc = self._connector.start(launch=self._launch(worker_id), log_file=log_file)
        print(f"Worker {worker_id} started (PID {proc.pid})")
        return proc

    def _launch(self, worker_id: int) -> WorkerLaunch:
        """What the worker of slot ``worker_id`` is started with."""
        return WorkerLaunch(
            self._output_path,
            run_id=self._run_id,
            best_of=self._best_of,
            decks=self._decks,
            records=WorkerRecords(
                directory=self._effect_records_dir,
                # The slot, not the process: a respawn passes the same number, so
                # this cannot be what makes a restarted worker's ids distinct.
                worker_index=worker_id,
                collection_caps=self._worker_caps(),
            ),
            stop=self._stop_protocol(worker_id),
            progress_file=self._progress_path,
        )

    def _stop_protocol(self, worker_id: int) -> WorkerStopProtocol:
        """The stop file and recycle age of slot ``worker_id``'s worker; none
        without the stop protocol."""
        if self._worker_stop is None:
            return WorkerStopProtocol()
        return WorkerStopProtocol(
            stop_file=self._worker_logs.stop_path_for(worker_id),
            recycle_seconds=self._worker_stop.recycle_seconds,
        )

    def _worker_caps(self) -> Mapping[str, object] | None:
        """The caps a worker spawned now starts with: the run's, each listed
        rule's own probability as recomputed at the last log scan."""
        if self._directed_needs is None or self._collection_caps is None:
            return self._collection_caps
        return {**self._collection_caps,
                **directed_rule_probability_properties(self._directed_probabilities)}

    # ── Pool state, exposed under the names this class published before the
    # extraction so callers and tests keep working. ──────────────────────────

    @property
    def _shutdown_event(self) -> threading.Event:
        return self._pool._shutdown_event

    def _kill_oldest_worker(self) -> None:
        self._pool._kill_oldest_worker()

    def _count_output_lines(self) -> int:
        return self._pool.output_line_count()
