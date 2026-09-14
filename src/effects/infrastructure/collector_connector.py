"""Supervises the records-only worker over coverage and variant decks.

The third supervisor over ``ForgeWorkerPool`` (after ``sealed
match-outcomes`` and ``draft play-draft-games``), so the pool's restart,
recycling and status-line behaviour come for free and the operator sees the
same output shape from every long-running Forge command in this repo.

What is different here is the worker's mode: no ``output.file`` is passed, so
neither sealed writer is constructed and these matches can only write effect
records. That guard lives in the Java worker rather than here, because that is
where the writers are built — a Python-side check would not bind.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from effects.domain.collection_caps import CollectionCaps
from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool, WorkerLogFiles
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

logger = logging.getLogger(__name__)

#: Coverage matches are one game each: the question is whether a card reached a
#: game at all, and a best-of-three would spend three times the simulation on
#: the same answer.
COVERAGE_BEST_OF = 1


class CollectorSupervisor:
    """Runs records-only workers until the caller stops asking for rounds."""

    def __init__(
        self,
        worker_count: int,
        effect_records: Path,
        *,
        run_id: str | None = None,
        caps: CollectionCaps | None = None,
    ) -> None:
        self._worker_count = worker_count
        self._effect_records = Path(effect_records)
        self._run_id = run_id or str(uuid.uuid4())
        self._caps = caps or CollectionCaps()
        self._connector = MatchWorkerConnector()
        self._pool: ForgeWorkerPool | None = None
        self._decks_file: Path | None = None
        # Every worker here runs with effect-record collection on by
        # construction (records-only mode, see start_worker), so any of the
        # five latched effect-record failure reporters (ApiEvents.
        # reportEmitterFailure and its siblings) can fire on it. Without a
        # real destination they print into DEVNULL and a broken emitter is
        # indistinguishable from a mechanic that simply never occurred (F4).
        self._worker_logs = WorkerLogFiles(self._effect_records, self._run_id)

    @property
    def run_id(self) -> str:
        return self._run_id

    def start_worker(self, worker_id: int) -> subprocess.Popen:
        """Spawn one records-only worker.

        ``output_file=None`` is what puts the worker in records-only mode: it
        constructs neither the match-outcome writer nor the cards-played one, so
        there is no path by which a coverage run reaches the sealed corpus.
        """
        process = self._connector.start(
            None,
            run_id=self._run_id,
            best_of=COVERAGE_BEST_OF,
            log_file=self._worker_logs.get(worker_id),
            effect_records_dir=self._effect_records,
            worker_index=worker_id,
            collection_caps=self._caps.as_system_properties(),
            side_a_decks_path=self._decks_file,
            side_b_decks_path=self._decks_file,
            decks_only=True,
            progress_file=self.progress_path,
        )
        logger.info("Coverage worker %d started (PID %d)", worker_id, process.pid)
        return process

    @property
    def progress_path(self) -> Path:
        """The file a round's completed matches are counted from.

        Under ``output/effects/``, never the sealed corpus: FR-052 and FR-059
        forbid a coverage or variant run reaching ``match-outcomes.txt``, and
        this is a per-run scratch file that nothing downstream reads.
        """
        return self._effect_records / f"{self._run_id}.progress.txt"

    def play_round(self, decks_file: Path, *, matches: int) -> None:
        """Play ``matches`` matches from ``decks_file``, then stop.

        The budget is what makes a round a round: without it the pool runs until
        it is signalled, the caller never recounts coverage, nothing retires,
        and the run cannot finish.

        ``decks_file`` is validated up front rather than merely logged: every
        worker this round spawns runs ``decks_only=True``, which forces both
        sides of every match to sample from this file (see ``start_worker``),
        so a missing or empty file is not a degraded round but one no worker
        can play at all. Raising here, before any worker is spawned, turns
        that into one clear Python-level error instead of an unexplained
        subprocess exit -- ``GeneratedDecksIndex.load`` on an empty file is
        untested territory this deliberately never reaches.
        """
        self._decks_file = Path(decks_file)
        if not self._decks_file.exists():
            raise ValueError(
                f"{self._decks_file}: a decks-only round cannot run without "
                "decks (file does not exist)"
            )
        deck_count = sum(1 for _ in self._decks_file.open(encoding="utf-8"))
        if deck_count == 0:
            raise ValueError(
                f"{self._decks_file}: a decks-only round cannot run without "
                "decks (file is empty)"
            )
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text("", encoding="utf-8")
        self._pool = ForgeWorkerPool(
            worker_count=self._worker_count,
            spawn_worker=self.start_worker,
            output_path=self.progress_path,
            should_stop=lambda completed: completed >= matches,
        )
        logger.info("Playing %d matches from %d decks", matches, deck_count)
        self._pool.run()

    def stop(self) -> None:
        self._worker_logs.close_all()
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
