"""PoolConnector: subprocess call to the forge-connector JAR for pool generation."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path


class PoolConnector:
    """Invokes the forge-connector JAR to generate sealed pools.

    Uses the same classpath resolution logic as the existing convert command
    in price_predictor/infrastructure/cli.py.
    """

    def generate(self, set_code: str, pool_count: int, pools_path: Path) -> int:
        """Generate sealed pools by invoking PoolMain via the connector JAR.

        Args:
            set_code: MTG set code (e.g. "RVR").
            pool_count: Number of sealed pools to generate.
            pools_path: Directory where pools.txt will be written.

        Returns:
            Process exit code (0 = success).

        Raises:
            FileNotFoundError: If the JAR is not found or java is not on PATH.
            RuntimeError: If the subprocess exits with a non-zero code.
        """
        jar_path = self._resolve_jar_path()

        project_root = jar_path.parent.parent.parent
        forge_dir = project_root.parent / "forge"
        forge_game_jar = forge_dir / "forge-game" / "target" / "forge-game-2.0.10-SNAPSHOT.jar"
        forge_core_jar = forge_dir / "forge-core" / "target" / "forge-core-2.0.10-SNAPSHOT.jar"
        forge_deps = forge_dir / "forge-game" / "target" / "dependency" / "*"

        sep = ";" if platform.system() == "Windows" else ":"
        classpath = sep.join([
            str(jar_path),
            str(forge_game_jar),
            str(forge_core_jar),
            str(forge_deps),
        ])

        cmd = [
            "java", "-cp", classpath,
            "com.pricepredictor.connector.PoolMain",
            "--set", set_code,
            "--size", str(pool_count),
            "--pools-path", str(pools_path),
        ]

        try:
            result = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            raise FileNotFoundError("Java not found. Ensure java is on PATH.")

        if result.returncode != 0:
            raise RuntimeError(
                f"PoolMain exited with code {result.returncode} for set={set_code}"
            )

        return result.returncode

    def _resolve_jar_path(self) -> Path:
        """Resolve the forge-connector JAR path relative to this file."""
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        jar = (
            project_root / "forge-connector" / "target"
            / "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
        )
        if not jar.exists():
            raise FileNotFoundError(
                f"Connector JAR not found at {jar}\n"
                "Build it first: cd forge-connector && mvn package -DskipTests"
            )
        return jar
