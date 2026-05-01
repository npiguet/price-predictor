"""MatchWorkerConnector: launches a Java MatchWorkerMain subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import IO

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)


class MatchWorkerConnector:
    """Spawns a single Java MatchWorkerMain process.

    The returned Popen handle can be waited on and terminated by the supervisor.
    """

    def start(
        self,
        output_file: Path,
        run_id: str,
        best_of: int,
        log_file: IO[bytes] | None = None,
        generated_decks_path: Path | None = None,
    ) -> subprocess.Popen:
        """Spawn a MatchWorkerMain Java subprocess and return its Popen handle.

        Args:
            output_file: Path where the worker will append match outcomes.
            run_id: Supervisor-generated UUID stamped onto every match line.
                Required by MatchWorkerMain; passed as ``-Dmatch.run.id``.
            best_of: Number of games per match (e.g. 3, 7, 17). Must be a
                positive odd integer. Passed as ``-Dmatch.best.of``.
            log_file: Open binary file handle for the worker's stdout/stderr.
                When None, output is discarded.
            generated_decks_path: When provided, passed to the Java worker as
                ``-Dgenerated.decks.file=<path>`` so it runs in self-play mode
                instead of the Phase 0 random-pool flow. The method tag for
                each scorer-built deck is read from the deck file's first
                column (``LABEL`` field, written by ``build-decks --label``).

        Returns:
            subprocess.Popen handle for the spawned worker process.

        Raises:
            ValueError: If ``best_of`` is not a positive odd integer.
            FileNotFoundError: If the JAR is not found or java is not on PATH.
        """
        if best_of < 1 or best_of % 2 == 0:
            raise ValueError(
                f"best_of must be a positive odd integer, got: {best_of}"
            )

        system_properties: dict[str, str] = {
            "output.file": str(output_file),
            "match.run.id": run_id,
            "match.best.of": str(best_of),
        }
        if generated_decks_path is not None:
            system_properties["generated.decks.file"] = str(generated_decks_path)

        cmd = build_jvm_command(
            main_class="com.pricepredictor.connector.MatchWorkerMain",
            classpath=build_forge_classpath(),
            system_properties=system_properties,
            xmx="1200m",
        )

        stdio = log_file if log_file is not None else subprocess.DEVNULL
        return subprocess.Popen(
            cmd, stdout=stdio, stderr=stdio, start_new_session=True,
        )
