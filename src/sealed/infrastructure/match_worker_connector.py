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
        generated_decks_path: Path | None = None,
    ) -> subprocess.Popen:
        """Spawn a MatchWorkerMain Java subprocess and return its Popen handle.

        Args:
            output_file: Path where the worker will append match outcomes.
            log_file: Open binary file handle for the worker's stdout/stderr.
                When None, output is discarded.
            generated_decks_path: When provided, passed to the Java worker as
                ``-Dgenerated.decks.file=<path>`` so it runs in self-play mode
                instead of the Phase 0 random-pool flow.

        Returns:
            subprocess.Popen handle for the spawned worker process.

        Raises:
            FileNotFoundError: If the JAR is not found or java is not on PATH.
        """
        system_properties: dict[str, str] = {"output.file": str(output_file)}
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
