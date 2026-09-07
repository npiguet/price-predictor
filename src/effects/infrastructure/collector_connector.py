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

from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool
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
    ) -> None:
        self._worker_count = worker_count
        self._effect_records = Path(effect_records)
        self._run_id = run_id or str(uuid.uuid4())
        self._connector = MatchWorkerConnector()
        self._pool: ForgeWorkerPool | None = None

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
            effect_records_dir=self._effect_records,
            worker_index=worker_id,
        )
        logger.info("Coverage worker %d started (PID %d)", worker_id, process.pid)
        return process

    def play_round(
        self, weights: dict[str, float], cards_folder: Path, *, decks: int,
    ) -> None:
        """Play one round's worth of weighted decks.

        The weights name which cards this round is trying to reach; the worker
        builds the decks itself from the converted corpus, because deck building
        needs a live Forge to know what is castable together.
        """
        self._pool = ForgeWorkerPool(
            worker_count=self._worker_count,
            spawn_worker=self.start_worker,
            output_path=self._effect_records,
        )
        logger.info(
            "Playing %d decks over %d weighted cards", decks, len(weights),
        )
        self._pool.run()

    def stop(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
