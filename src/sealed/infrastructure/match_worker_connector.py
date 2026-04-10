"""MatchWorkerConnector: launches a Java MatchWorkerMain subprocess."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


class MatchWorkerConnector:
    """Builds the Java subprocess command and spawns a MatchWorkerMain worker.

    The returned Popen handle can be waited on and terminated by the supervisor.
    """

    def start(self, output_file: Path) -> subprocess.Popen:
        """Spawn a MatchWorkerMain Java subprocess and return its Popen handle.

        Args:
            output_file: Path where the worker will append match outcomes.

        Returns:
            subprocess.Popen handle for the spawned worker process.

        Raises:
            FileNotFoundError: If the JAR is not found or java is not on PATH.
        """
        jar_path = self._resolve_jar_path()

        project_root = jar_path.parent.parent.parent
        forge_dir = project_root.parent / "forge"

        forge_game_jar = forge_dir / "forge-game" / "target" / "forge-game-2.0.10-SNAPSHOT.jar"
        forge_core_jar = forge_dir / "forge-core" / "target" / "forge-core-2.0.10-SNAPSHOT.jar"
        forge_gui_jar = forge_dir / "forge-gui" / "target" / "forge-gui-2.0.10-SNAPSHOT.jar"
        forge_ai_jar = forge_dir / "forge-ai" / "target" / "forge-ai-2.0.10-SNAPSHOT.jar"

        sep = ";" if platform.system() == "Windows" else ":"
        classpath = sep.join([
            str(jar_path),
            str(forge_game_jar),
            str(forge_core_jar),
            str(forge_gui_jar),
            str(forge_ai_jar),
        ])

        cmd = [
            "java",
            f"-Doutput.file={output_file}",
            "-Xmx1200m",
            "-cp", classpath,
            "com.pricepredictor.connector.MatchWorkerMain",
        ]

        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _resolve_jar_path(self) -> Path:
        """Resolve the forge-connector fat JAR path relative to this file."""
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
