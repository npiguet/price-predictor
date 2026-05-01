"""MatchWorkerConnector: launches a Java MatchWorkerMain subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import IO

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)

# Default weight of the file-sampled side-B method relative to the 4 Forge
# methods (combined weight 10). Single source of truth: the supervisor
# default, the connector default, the CLI fallback, and the CLI help text
# all reference this value.
DEFAULT_SIDE_B_DECKS_WEIGHT = 4


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
        side_a_decks_path: Path | None = None,
        side_b_decks_path: Path | None = None,
        side_b_decks_weight: int = DEFAULT_SIDE_B_DECKS_WEIGHT,
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
            side_a_decks_path: When provided, the Java worker samples deck A
                from this generated-decks file instead of using the 4 Forge
                methods on side A. Passed as ``-Dside.a.decks.file=<path>``.
            side_b_decks_path: When provided, the Java worker rolls deck B
                between the 4 Forge methods (weights 4:3:2:1) and sampling
                from this file (weight ``side_b_decks_weight``). When omitted,
                deck B uses Forge methods only. Passed as
                ``-Dside.b.decks.file=<path>``.
            side_b_decks_weight: Weight of the file-sampled side-B method
                relative to the Forge methods. Ignored when
                ``side_b_decks_path`` is None. Passed as
                ``-Dside.b.decks.weight=<int>``.

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
        if side_a_decks_path is not None:
            system_properties["side.a.decks.file"] = str(side_a_decks_path)
        if side_b_decks_path is not None:
            system_properties["side.b.decks.file"] = str(side_b_decks_path)
            system_properties["side.b.decks.weight"] = str(side_b_decks_weight)

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
