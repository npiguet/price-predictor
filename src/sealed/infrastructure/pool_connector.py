"""PoolConnector: subprocess call to the forge-connector JAR for pool generation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from price_predictor.infrastructure.forge_jvm import (
    build_forge_classpath,
    build_jvm_command,
)


class PoolConnector:
    """Invokes the forge-connector JAR to generate sealed pools."""

    def generate(self, set_code: str, pool_count: int, pools_path: Path) -> int:
        """Generate sealed pools by invoking PoolMain via the connector JAR.

        Args:
            set_code: MTG set code (e.g. ``RVR``).
            pool_count: Number of sealed pools to generate.
            pools_path: Directory where pools.txt will be written.

        Returns:
            Process exit code (0 = success).

        Raises:
            FileNotFoundError: If the JAR is not found or java is not on PATH.
            RuntimeError: If the subprocess exits with a non-zero code.
        """
        cmd = build_jvm_command(
            main_class="com.pricepredictor.connector.PoolMain",
            classpath=build_forge_classpath(),
            main_args=[
                "--set", set_code,
                "--size", str(pool_count),
                "--pools-path", str(pools_path),
            ],
        )

        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError("Java not found. Ensure java is on PATH.") from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"PoolMain exited with code {result.returncode} for set={set_code}"
            )

        return result.returncode
