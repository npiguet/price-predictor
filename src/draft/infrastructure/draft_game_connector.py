"""Launch Java DraftGameWorkerMain subprocesses.

Mirrors ``draft_worker_connector`` and the sealed connectors: the JVM command is
assembled from the shared helpers in ``price_predictor.infrastructure.forge_classpath``
and ``price_predictor.infrastructure.jvm_processes`` so every worker in the repo
is launched the same way.

Workers are autonomous — they are handed a seat table and choose their own
pairings — so this connector passes a population and a few settings, never a
work list.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from price_predictor.infrastructure.forge_classpath import build_forge_classpath
from price_predictor.infrastructure.jvm_processes import JvmOptions, build_jvm_command

MAIN_CLASS = "com.pricepredictor.connector.DraftGameWorkerMain"
WORKER_XMX = "1200m"  # matches the sealed evaluation worker


@dataclass(frozen=True)
class DraftGameLaunch:
    """Everything a ``DraftGameWorkerMain`` worker is started with."""

    seats_file: Path
    output_file: Path
    run_id: str
    best_of: int
    include_mirrors: bool


class DraftGameConnector:
    """Builds and starts draft game-evaluation workers."""

    def build_command(self, *, launch: DraftGameLaunch) -> list[str]:
        """Assemble the JVM command for one worker.

        The five system properties are the contract in
        ``specs/022-draft-game-evaluation/contracts/seat-table.md``. There is no
        hybrid-fraction property: diversion is applied when the seat table is
        written, and reaches the worker as a seat's label and kind.
        """
        return build_jvm_command(
            main_class=MAIN_CLASS,
            classpath=build_forge_classpath(),
            options=JvmOptions(
                system_properties={
                    "seats.file": str(launch.seats_file),
                    "output.file": str(launch.output_file),
                    "run.id": launch.run_id,
                    "best.of": str(launch.best_of),
                    "include.mirrors": "true" if launch.include_mirrors else "false",
                },
                xmx=WORKER_XMX,
            ),
        )

    def start(self, *, launch: DraftGameLaunch) -> subprocess.Popen:
        """Start one worker.

        Worker stdout/stderr are discarded, as the sealed match supervisor does:
        the supervisor's own status line is the operator-facing output, and Forge
        is verbose enough that capturing per-worker logs costs measurable I/O on
        long runs.
        """
        cmd = self.build_command(launch=launch)
        return subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
