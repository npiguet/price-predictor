"""Supervises the records-only worker over coverage and variant decks."""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

from effects.domain.collection_caps import CollectionCaps
from price_predictor.infrastructure.forge_jvm import ForgeWorkerPool, WorkerLogFiles
from sealed.infrastructure.match_worker_connector import MatchWorkerConnector

logger = logging.getLogger(__name__)

#: One game each: the question is whether a card reached a game at all.
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
        variant_scripts: Path | None = None,
    ) -> None:
        self._worker_count = worker_count
        self._effect_records = Path(effect_records)
        self._run_id = run_id or str(uuid.uuid4())
        self._caps = caps or CollectionCaps()
        self._variant_scripts = variant_scripts
        self._connector = MatchWorkerConnector()
        self._pool: ForgeWorkerPool | None = None
        self._interrupted = False
        self._decks_file: Path | None = None
        # A real log destination, or the JVM's latched emitter-failure reports
        # vanish into DEVNULL and look like a mechanic that never occurred.
        self._worker_logs = WorkerLogFiles(self._effect_records, self._run_id)

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def interrupted(self) -> bool:
        """True when some round ended on Ctrl-C rather than at its budget.

        ``ForgeWorkerPool.run`` returns normally on SIGINT, so without this a
        caller would start the next round. Latched, because ``stop()`` drops
        the pool before ``collect_variants`` asks.
        """
        return self._interrupted

    def start_worker(self, worker_id: int) -> subprocess.Popen:
        """Spawn one records-only worker: with no output file the JVM builds
        neither sealed writer, so a coverage run cannot reach the sealed corpus."""
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
            variant_scripts=self._variant_scripts,
        )
        logger.info("Coverage worker %d started (PID %d)", worker_id, process.pid)
        return process

    @property
    def progress_path(self) -> Path:
        """The per-run scratch file a round's completed matches are counted
        from, never the sealed corpus (FR-052, FR-059)."""
        return self._effect_records / f"{self._run_id}.progress.txt"

    def play_round(self, decks_file: Path, *, matches: int) -> None:
        """Play ``matches`` matches from ``decks_file``, then stop.

        A missing or empty ``decks_file`` raises before any worker spawns:
        every worker is decks-only, so no worker could play at all.
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
        self._interrupted = self._interrupted or self._pool.interrupted

    def stop(self) -> None:
        self._worker_logs.close_all()
        if self._pool is not None:
            # Runs in both collectors' ``finally``: a raise here would hide
            # the run's report and return code.
            self._pool.request_shutdown()
            self._pool = None
