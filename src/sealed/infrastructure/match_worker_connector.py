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
        log_file: IO[bytes] | None = None,
        generated_decks_path: Path | None = None,
        self_play_label: str | None = None,
    ) -> subprocess.Popen:
        """Spawn a MatchWorkerMain Java subprocess and return its Popen handle.

        Args:
            output_file: Path where the worker will append match outcomes.
            run_id: Supervisor-generated UUID stamped onto every match line.
                Required by MatchWorkerMain; passed as ``-Dmatch.run.id``.
            log_file: Open binary file handle for the worker's stdout/stderr.
                When None, output is discarded.
            generated_decks_path: When provided, passed to the Java worker as
                ``-Dgenerated.decks.file=<path>`` so it runs in self-play mode
                instead of the Phase 0 random-pool flow.
            self_play_label: Method tag written on scorer-built decks in self-play
                mode. Required iff ``generated_decks_path`` is provided; must be
                absent otherwise. Passed as ``-Dself.play.label``.

        Returns:
            subprocess.Popen handle for the spawned worker process.

        Raises:
            ValueError: If the self_play_label / generated_decks_path XOR rule is
                violated (the Java worker also enforces this as a defense in depth).
            FileNotFoundError: If the JAR is not found or java is not on PATH.
        """
        if generated_decks_path is not None and not self_play_label:
            raise ValueError(
                "self_play_label is required when generated_decks_path is provided"
            )
        if generated_decks_path is None and self_play_label is not None:
            raise ValueError(
                "self_play_label is forbidden when generated_decks_path is not provided"
            )

        system_properties: dict[str, str] = {
            "output.file": str(output_file),
            "match.run.id": run_id,
        }
        if generated_decks_path is not None:
            system_properties["generated.decks.file"] = str(generated_decks_path)
            system_properties["self.play.label"] = self_play_label  # type: ignore[assignment]

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
