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
        log_file: IO[bytes] | None = None,
    ) -> subprocess.Popen:
        """Spawn a MatchWorkerMain Java subprocess and return its Popen handle.

        Args:
            output_file: Path where the worker will append match outcomes.
            log_file: Open binary file handle for the worker's stdout/stderr.
                When None, output is discarded.

        Returns:
            subprocess.Popen handle for the spawned worker process.

        Raises:
            FileNotFoundError: If the JAR is not found or java is not on PATH.
        """
        cmd = build_jvm_command(
            main_class="com.pricepredictor.connector.MatchWorkerMain",
            classpath=build_forge_classpath(),
            system_properties={"output.file": str(output_file)},
            xmx="1200m",
        )

        stdio = log_file if log_file is not None else subprocess.DEVNULL
        # start_new_session puts the child in its own process group on POSIX
        # so kill_process_tree's os.killpg can take down the whole tree.
        return subprocess.Popen(
            cmd, stdout=stdio, stderr=stdio, start_new_session=True,
        )
