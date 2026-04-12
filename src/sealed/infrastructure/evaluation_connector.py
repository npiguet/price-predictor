"""Launch and manage Java evaluation processes for scorer evaluation."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
    kill_process_tree,
)


@dataclass
class _ManagedWorker:
    """Tracks one validation worker through its spawn / restart lifecycle."""

    proc: subprocess.Popen
    matches_file: Path
    outcomes_file: Path
    log_file: IO[bytes]
    last_count: int
    last_progress: float
    done: bool = False


def _count_outcomes(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


class EvaluationConnector:
    """Coordinates Java processes for scorer evaluation."""

    def build_forge_decks(self, pools: list[list[str]]) -> list[list[str]]:
        """Build one Forge deck per pool using DeckBuilderMain.

        Sends all pools to a single JVM invocation via stdin (one pool per
        line, pipe-separated card names) and reads back one built deck per
        line from stdout. Avoids N separate JVM startups with Forge init.
        """
        cmd = self._build_deck_builder_command()
        stdin_text = "\n".join("|".join(pool) for pool in pools) + "\n"

        result = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"DeckBuilderMain failed (rc={result.returncode}):\n{result.stderr}"
            )

        decks: list[list[str]] = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line:
                decks.append(line.split("|"))
        return decks

    def outcome_file_path(self, matches_file: Path) -> Path:
        """Derive the outcomes file path from a matches file path."""
        return Path(str(matches_file) + "-outcomes.txt")

    def launch_workers(
        self,
        split_files: list[Path],
        best_of: int = 3,
        idle_timeout_s: float = 300.0,
        poll_interval_s: float = 10.0,
    ) -> list[Path]:
        """Launch Java workers in parallel and wait for completion.

        Watchdog: each worker is polled every ``poll_interval_s`` seconds.
        If a worker has not appended a new outcome line for ``idle_timeout_s``
        seconds, or if it exits with a non-zero code before finishing all
        matches, it is killed and restarted on the same matches file.
        ``ValidationMatchPlayer`` skips already-completed matches based on
        the outcomes file line count, so a restart resumes where the
        previous attempt left off.

        Returns outcome file paths in the same order as ``split_files``.
        """
        outcome_files = [self.outcome_file_path(f) for f in split_files]

        workers = self._spawn_initial_workers(
            split_files, outcome_files, best_of=best_of,
        )
        try:
            self._poll_until_done(
                workers,
                best_of=best_of,
                idle_timeout_s=idle_timeout_s,
                poll_interval_s=poll_interval_s,
            )
        finally:
            for w in workers:
                try:
                    w.log_file.close()
                except Exception:
                    pass

        return outcome_files

    def _spawn(
        self,
        matches_file: Path,
        log_file: IO[bytes],
        best_of: int,
    ) -> subprocess.Popen:
        cmd = self._build_worker_command(matches_file, best_of=best_of)
        return subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file, start_new_session=True,
        )

    def _spawn_initial_workers(
        self,
        split_files: list[Path],
        outcome_files: list[Path],
        *,
        best_of: int,
    ) -> list[_ManagedWorker]:
        now = time.monotonic()
        workers: list[_ManagedWorker] = []
        for matches_file, outcomes_file in zip(split_files, outcome_files):
            log_path = matches_file.with_suffix(".worker.log")
            log_file = log_path.open("ab")
            proc = self._spawn(matches_file, log_file, best_of)
            workers.append(_ManagedWorker(
                proc=proc,
                matches_file=matches_file,
                outcomes_file=outcomes_file,
                log_file=log_file,
                last_count=_count_outcomes(outcomes_file),
                last_progress=now,
            ))
        return workers

    def _poll_until_done(
        self,
        workers: list[_ManagedWorker],
        *,
        best_of: int,
        idle_timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        while True:
            all_done = True
            for w in workers:
                if w.done:
                    continue
                if self._poll_worker(
                    w, best_of=best_of, idle_timeout_s=idle_timeout_s,
                ):
                    all_done = False
            if all_done:
                break
            time.sleep(poll_interval_s)

    def _poll_worker(
        self,
        w: _ManagedWorker,
        *,
        best_of: int,
        idle_timeout_s: float,
    ) -> bool:
        """Advance one worker's state. Returns True if it's still running."""
        rc = w.proc.poll()

        if rc == 0:
            print(f"Worker completed: {w.matches_file.name}")
            w.done = True
            return False

        if rc is not None:
            print(f"Worker restarted (rc={rc}): {w.matches_file.name}")
            self._restart(w, best_of)
            return True

        current_count = _count_outcomes(w.outcomes_file)
        now = time.monotonic()
        if current_count > w.last_count:
            w.last_count = current_count
            w.last_progress = now
        elif now - w.last_progress >= idle_timeout_s:
            print(f"Worker restarted (idle): {w.matches_file.name}")
            kill_process_tree(w.proc)
            self._restart(w, best_of)
        return True

    def _restart(self, w: _ManagedWorker, best_of: int) -> None:
        w.proc = self._spawn(w.matches_file, w.log_file, best_of)
        w.last_count = _count_outcomes(w.outcomes_file)
        w.last_progress = time.monotonic()

    def _build_worker_command(self, matches_file: Path, best_of: int = 3) -> list[str]:
        """Build the Java command for a validation worker."""
        return build_jvm_command(
            main_class="com.pricepredictor.connector.ValidationWorkerMain",
            classpath=build_forge_classpath(),
            system_properties={
                "matches.file": str(matches_file),
                "best.of": str(best_of),
            },
            xmx="1200m",
        )

    def _build_deck_builder_command(self) -> list[str]:
        """Build the Java command for the batch Forge deck builder."""
        return build_jvm_command(
            main_class="com.pricepredictor.connector.DeckBuilderMain",
            classpath=build_forge_classpath(),
            xmx="1200m",
        )
