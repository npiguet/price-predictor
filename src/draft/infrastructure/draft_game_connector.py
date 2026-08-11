"""Launch Java DraftGameWorkerMain subprocesses.

Mirrors ``draft_worker_connector`` and the sealed connectors: the JVM command is
assembled from the shared helpers in ``price_predictor.infrastructure.forge_jvm``
so every worker in the repo is launched the same way.

Workers are autonomous — they are handed a seat table and choose their own
pairings — so this connector passes a population and a few settings, never a
work list.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)

MAIN_CLASS = "com.pricepredictor.connector.DraftGameWorkerMain"
WORKER_XMX = "1200m"  # matches the sealed evaluation worker


class DraftGameConnector:
    """Builds and starts draft game-evaluation workers."""

    def build_command(
        self,
        *,
        seats_file: Path,
        output_file: Path,
        run_id: str,
        best_of: int,
        include_mirrors: bool,
    ) -> list[str]:
        """Assemble the JVM command for one worker.

        The five system properties are the contract in
        ``specs/022-draft-game-evaluation/contracts/seat-table.md``. There is no
        hybrid-fraction property: diversion is applied when the seat table is
        written, and reaches the worker as a seat's label and kind.
        """
        return build_jvm_command(
            main_class=MAIN_CLASS,
            classpath=build_forge_classpath(),
            system_properties={
                "seats.file": str(seats_file),
                "output.file": str(output_file),
                "run.id": run_id,
                "best.of": str(best_of),
                "include.mirrors": "true" if include_mirrors else "false",
            },
            xmx=WORKER_XMX,
        )

    def start(
        self,
        *,
        seats_file: Path,
        output_file: Path,
        run_id: str,
        best_of: int,
        include_mirrors: bool,
    ) -> subprocess.Popen:
        """Start one worker.

        Worker stdout/stderr are discarded, as the sealed match supervisor does:
        the supervisor's own status line is the operator-facing output, and Forge
        is verbose enough that capturing per-worker logs costs measurable I/O on
        long runs.
        """
        cmd = self.build_command(
            seats_file=seats_file,
            output_file=output_file,
            run_id=run_id,
            best_of=best_of,
            include_mirrors=include_mirrors,
        )
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
