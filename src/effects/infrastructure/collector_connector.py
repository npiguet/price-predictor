"""Supervises the records-only worker over coverage and variant decks.

A collector's run directory is its ``--effect-records`` directory
(``.claude/skills/artifact-layout/SKILL.md``): the workers' shards go to its
top level, every worker log and each round's progress file under ``logs/``,
and the decks file the workers read under ``decks/``.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import subprocess
from pathlib import Path

from effects.domain.collection_caps import CollectionCaps
from effects.infrastructure.deck_file import write_deck_file
from price_predictor.domain.worker_failures import WorkerFailures
from price_predictor.domain.worker_status import (
    WorkerEngineRefused,
    render_fork_refusals,
    render_hook_failures,
)
from price_predictor.infrastructure.forge_worker_pool import ForgeWorkerPool, PoolOptions
from price_predictor.infrastructure.run_manifest import annotate_run
from price_predictor.infrastructure.worker_log_files import WorkerLogFiles
from price_predictor.infrastructure.worker_log_scan import WorkerLogReport
from price_predictor.infrastructure.worker_stops import WorkerStops, WorkerStopSettings
from sealed.infrastructure.dropped_events import dropped_events_report
from sealed.infrastructure.match_worker_connector import (
    MatchWorkerConnector,
    WorkerDecks,
    WorkerLaunch,
    WorkerRecords,
    WorkerStopProtocol,
)

logger = logging.getLogger(__name__)

#: One game each: the question is whether a card reached a game at all.
COVERAGE_BEST_OF = 1

#: Under the run directory: every worker log and each round's progress file.
LOGS_DIR = "logs"
#: Under the run directory: the decks file each round's workers read.
DECKS_DIR = "decks"


def caps_document(caps: CollectionCaps) -> dict:
    """The resolved caps as ``run.json`` records them, every field by name."""
    return {
        name: list(value) if isinstance(value, tuple) else value
        for name, value in dataclasses.asdict(caps).items()
    }


@dataclasses.dataclass(frozen=True)
class CollectorRun:
    """What one collector run is: how many workers write to which run
    directory, under which run id and caps; ``variant_scripts`` is the variant
    tree the workers stage; ``stop`` the stop protocol's settings (the
    defaults when None); ``allow_degraded`` lets the run go on over an engine
    that is not whole."""

    worker_count: int
    effect_records: Path
    run_id: str
    caps: CollectionCaps
    variant_scripts: Path | None = None
    stop: WorkerStopSettings | None = None
    allow_degraded: bool = False


@dataclasses.dataclass(frozen=True)
class RoundSpec:
    """One round: its decks file's ``label`` and the ``matches`` it plays;
    ``rules`` gives each steered deck its rule column (``deck_file``)."""

    label: str
    matches: int
    rules: list[str | None] | None = None


class CollectorSupervisor:
    """Runs records-only workers until the caller stops asking for rounds.

    After every round it scans the workers' logs for failure lines, reports
    them, and records the run's totals, with its run id and resolved caps,
    in the running command's entry of ``run.json`` (``annotate_run``).

    Its workers honour the stop protocol (``contracts/cli.md`` § Match worker
    stop protocol): a round ends once its pool has asked every worker to stop
    and each has finished its game (or been killed past ``stop.timeout``),
    so a round's shards are final when ``play_round`` returns; workers
    recycle themselves at a game end after ``stop.recycle_seconds``.

    Every status interval, and at each round's end, it reads the workers'
    status lines: an engine not ``patched`` with every hook stops the round
    by that same protocol and ``play_round`` raises
    :class:`WorkerEngineRefused`, unless ``allow_degraded``; an engine that
    changes under the run always does (``worker_status.EngineVerdict``).
    """

    def __init__(self, run: CollectorRun) -> None:
        worker_count = run.worker_count
        self._worker_count = worker_count
        self._effect_records = Path(run.effect_records)
        self._run_id = run.run_id
        self._caps = run.caps
        self._variant_scripts = run.variant_scripts
        self._connector = MatchWorkerConnector()
        self._pool: ForgeWorkerPool | None = None
        self._interrupted = False
        # A real log destination, or the JVM's latched emitter-failure reports
        # vanish into DEVNULL and look like a mechanic that never occurred.
        self._worker_logs = WorkerLogFiles(self.logs_dir, self._run_id)
        self._log_report = WorkerLogReport(
            self._worker_logs, worker_count, allow_degraded=run.allow_degraded,
            run_dir=self._effect_records)
        #: The failure lines the current round's status-interval scans found.
        self._round_failures = WorkerFailures()
        self._stop = run.stop if run.stop is not None else WorkerStopSettings()
        self._graceful = self._worker_logs.graceful_stop(self._stop)
        #: PIDs of workers a round's kill could not end.
        self.survivors: list[int] = []
        #: How the run's workers ended, over every round.
        self.stops = WorkerStops()
        #: Gen-3 games the run's counting left out for want of a game-end
        #: marker: before the first round, then each round's.
        self._cut_before_rounds = 0
        self._cut_by_round: list[int] = []
        self._rounds = 0
        annotate_run(
            self._effect_records, run_id=self._run_id, caps=caps_document(self._caps),
            **self._stop.manifest_fields(),
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def worker_failures(self) -> WorkerFailures:
        """Every failure line the run's workers logged, by kind."""
        return self._log_report.totals

    @property
    def rounds(self) -> int:
        """Rounds played so far."""
        return self._rounds

    @property
    def logs_dir(self) -> Path:
        return self._effect_records / LOGS_DIR

    def decks_path(self, kind: str) -> Path:
        """The run's own decks file for ``kind`` (``coverage``, ``variant``):
        every worker reloads it at its start, so two runs must never share one."""
        return self._effect_records / DECKS_DIR / f"{kind}-decks.{self._run_id}.txt"

    @property
    def interrupted(self) -> bool:
        """True when some round ended on Ctrl-C rather than at its budget.

        ``ForgeWorkerPool.run`` returns normally on SIGINT, so without this a
        caller would start the next round, which ``play_round`` now refuses.
        The pool restores the previous handlers when its run ends, so a
        Ctrl-C after the round raises ``KeyboardInterrupt`` in the caller.
        Latched when the pool is dropped, because ``stop()`` drops it before
        ``collect_variants`` asks.
        """
        return self._interrupted or (self._pool is not None and self._pool.interrupted)

    def start_worker(self, worker_id: int, decks_file: Path) -> subprocess.Popen:
        """Spawn one records-only worker playing both sides from ``decks_file``:
        with no output file the JVM builds neither sealed writer, so a coverage
        run cannot reach the sealed corpus."""
        log_file = self._worker_logs.get(worker_id)
        process = self._connector.start(
            launch=WorkerLaunch(
                None,
                run_id=self._run_id,
                best_of=COVERAGE_BEST_OF,
                decks=WorkerDecks(side_a=decks_file, side_b=decks_file, decks_only=True),
                records=WorkerRecords(
                    directory=self._effect_records,
                    worker_index=worker_id,
                    collection_caps=self._caps.as_system_properties(),
                    variant_scripts=self._variant_scripts,
                ),
                stop=WorkerStopProtocol(
                    stop_file=self._graceful.stop_file_for(worker_id),
                    recycle_seconds=self._stop.recycle_seconds,
                ),
                progress_file=self.progress_path,
            ),
            log_file=log_file,
        )
        logger.info("Coverage worker %d started (PID %d)", worker_id, process.pid)
        return process

    @property
    def progress_path(self) -> Path:
        """The current round's scratch file its completed matches are counted
        from, never the sealed corpus (FR-052, FR-059). One per round: a
        worker the backstop kill could not end keeps appending to its own
        round's file, never to the next round's count."""
        return self.logs_dir / f"{self._run_id}.round-{self._rounds:04d}.progress.txt"

    def play_round(self, decks: list[list[str]], decks_file: Path, spec: RoundSpec) -> None:
        """Write ``decks`` to ``decks_file``, the file the workers read, and
        play ``spec.matches`` matches from it, then stop and report what the
        workers logged. ``spec.rules`` gives each steered deck its rule column
        (``deck_file``).

        Once the progress file counts ``matches``, the pool asks every worker
        to stop and returns only when each has finished the game it was
        playing (a best-of-one match) or been killed past the stop timeout:
        each worker plays at most one game more, so a round ends at most one
        game's length (bounded by the timeout) after its budget, the progress
        file overshooting it by the matches that ended between two polls
        (``ForgeWorkerPool.STOP_POLL_INTERVAL``) plus at most one per
        worker, and the round's
        shards are closed whole before the caller counts them.

        No deck raises before any worker spawns: every worker is decks-only,
        so no worker could play at all.

        Once an interrupt is latched (:attr:`interrupted`) it returns at once
        without writing, counting or starting anything: a Ctrl-C between two
        rounds ends the run and does not play one more.
        """
        if self.interrupted:
            return
        deck_count = write_deck_file(decks, decks_file, label=spec.label, rules=spec.rules)
        if deck_count == 0:
            raise ValueError(
                f"{decks_file}: a decks-only round cannot run without decks "
                "(no deck to write)"
            )
        self._rounds += 1
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text("", encoding="utf-8")
        self._interrupted = self.interrupted
        self._pool = ForgeWorkerPool(
            worker_count=self._worker_count,
            spawn_worker=functools.partial(self.start_worker, decks_file=Path(decks_file)),
            output_path=self.progress_path,
            options=PoolOptions(
                should_stop=lambda completed: completed >= spec.matches,
                on_status=self._scan_logs,
                graceful=self._graceful,
            ),
        )
        logger.info("Playing %d matches from %d decks", spec.matches, deck_count)
        try:
            self._pool.run()
        finally:
            self._report_round(self._pool)
        refusal = self._log_report.engine_refusal
        if refusal is not None:
            raise WorkerEngineRefused(
                f"{refusal}. The records written before the stop stay under "
                f"{self._effect_records}"
            )

    def _scan_logs(self) -> None:
        """At a status line: read what the logs gained, and stop the round's
        pool by the stop protocol once the workers' engine is refused."""
        played = self._played_a_match()
        self._round_failures.merge(self._log_report.scan())
        self._review_engine(self._pool, played=played)

    def _played_a_match(self) -> bool:
        """Whether a worker of the round completed a match: its progress file
        holds a line."""
        try:
            with open(self.progress_path, "rb") as progress:
                return bool(progress.readline())
        except OSError:
            return False

    def _review_engine(
        self, pool: ForgeWorkerPool, *, played: bool, ended: bool = False,
    ) -> None:
        """Judge the workers' engine (``WorkerLogReport.review_engine``):
        warn once under ``--allow-degraded``; on a refusal, say so and ask
        ``pool`` to stop every worker at its game's end, as a run reaching
        its target does."""
        refused_before = self._log_report.engine_refusal is not None
        warning = self._log_report.review_engine(played=played, ended=ended)
        if warning is not None:
            logger.warning("%s", warning)
        refusal = self._log_report.engine_refusal
        if refusal is not None and not refused_before:
            logger.error("Stopping the run: %s", refusal)
            pool.request_shutdown()

    def _report_round(self, pool: ForgeWorkerPool) -> None:
        """Say what the round's workers logged, how they ended and which
        outlived the kill, and record the run's totals in ``run.json``."""
        failures, self._round_failures = self._round_failures, WorkerFailures()
        failures.merge(self._log_report.scan())
        self._review_engine(pool, played=self._played_a_match(), ended=True)
        if failures:
            logger.warning(
                "Round %d worker logs: %s (under %s)",
                self._rounds, failures.render(), self.logs_dir,
            )
        else:
            logger.info("Round %d worker logs: no failure lines", self._rounds)
        hooks = self._log_report.hook_failures()
        if any(hooks.values()):
            logger.warning("Effect-record hook failures over the run: %s",
                           render_hook_failures(hooks))
        refusals = self._log_report.fork_refusals()
        if refusals:
            logger.info("Fork refusals over the run (expected, no failure): %s",
                        render_fork_refusals(refusals))
        stops = pool.stops
        self.stops.add(stops)
        (logger.warning if stops.killed else logger.info)(
            "Round %d workers: %s", self._rounds, stops.render(),
        )
        survived = [pid for pid in pool.survivors if pid not in self.survivors]
        if survived:
            self.survivors.extend(survived)
            logger.warning(
                "Round %d: %d worker(s) outlived the kill (PID %s); their shards "
                "were counted as they stood and may still grow",
                self._rounds, len(survived), ", ".join(map(str, survived)),
            )
        annotate_run(
            self._effect_records,
            rounds=self._rounds,
            **self._log_report.manifest_fields(),
            worker_survivors=list(self.survivors),
            worker_stops=self.stops.as_dict(survived=len(self.survivors)),
        )

    @property
    def caps(self) -> CollectionCaps:
        """The caps the next worker starts with."""
        return self._caps

    def set_directed_rule_probabilities(self, probabilities: dict[str, float]) -> None:
        """Start the workers of the rounds that follow with ``probabilities``
        as each listed rule's own directed-fork probability, and record them
        in ``run.json`` (the last set stays)."""
        self._caps = dataclasses.replace(
            self._caps, directed_rule_probabilities=dict(probabilities))
        annotate_run(
            self._effect_records, caps=caps_document(self._caps),
            directed_rule_probabilities=dict(probabilities),
        )

    def count_cut_games(self, cut: int) -> None:
        """Note the gen-3 games the last count of shards left out for want of
        a game-end marker (cut by a kill, a crash or Ctrl-C; their records
        count toward no unit, as build-corpus drops them): before the first
        round, or the round just played. Logged, and kept in ``run.json`` as
        ``cut_games`` (``before_rounds``, ``by_round``, ``total``)."""
        if self._rounds == 0:
            self._cut_before_rounds += cut
            if cut:
                logger.info("%d cut game(s) in the shards counted before the first "
                            "round, left out of the counts", cut)
        else:
            self._cut_by_round += [0] * (self._rounds - len(self._cut_by_round))
            self._cut_by_round[self._rounds - 1] += cut
            (logger.warning if cut else logger.info)(
                "Round %d: %d cut game(s) (no game-end marker) left out of the counts",
                self._rounds, cut,
            )
        annotate_run(self._effect_records, cut_games={
            "before_rounds": self._cut_before_rounds,
            "by_round": list(self._cut_by_round),
            "total": self._cut_before_rounds + sum(self._cut_by_round),
        })

    def record(self, **fields) -> None:
        """Keep ``fields``, facts the caller learned about this run, in the
        running command's entry of ``run.json``."""
        annotate_run(self._effect_records, **fields)

    def stop(self) -> None:
        """End the run: flush the worker logs, stop the pool, report the run's
        worker-log failures, and print the dropped-event table summed over
        every round's workers when the run counted dropped events (FR-041)."""
        self._worker_logs.close_all()
        self._interrupted = self.interrupted
        if self._pool is not None:
            # Runs in both collectors' ``finally``: a raise here would hide
            # the run's report and return code.
            self._pool.request_shutdown()
            self._pool = None
        if self._rounds:
            logger.info(
                "Worker logs over %d round(s): %s",
                self._rounds, self.worker_failures.render(),
            )
        if self._caps.count_dropped_events:
            print(dropped_events_report(self._effect_records, self._run_id))
