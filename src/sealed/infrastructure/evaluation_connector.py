"""Launch and manage Java ValidationWorkerMain processes for evaluation."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


class EvaluationConnector:
    """Coordinates Java validation workers for scorer evaluation."""

    def split_matches_file(self, matches_file: Path, n_workers: int, work_dir: Path) -> list[Path]:
        """Split a matches file into N worker-specific files."""
        lines = matches_file.read_text(encoding="utf-8").strip().splitlines()
        split_files = []
        chunk_size = len(lines) // n_workers
        remainder = len(lines) % n_workers

        offset = 0
        for i in range(n_workers):
            size = chunk_size + (1 if i < remainder else 0)
            worker_file = work_dir / f"validation-matches-{i}.txt"
            worker_file.write_text(
                "\n".join(lines[offset:offset + size]) + "\n",
                encoding="utf-8",
            )
            split_files.append(worker_file)
            offset += size

        return split_files

    def outcome_file_path(self, matches_file: Path) -> Path:
        """Derive the outcomes file path from a matches file path."""
        return Path(str(matches_file) + "-outcomes.txt")

    def launch_workers(self, split_files: list[Path], max_retries: int = 1) -> list[Path]:
        """Launch Java workers and wait for completion. Returns outcome file paths."""
        outcome_files = []

        for matches_file in split_files:
            outcome_file = self.outcome_file_path(matches_file)
            outcome_files.append(outcome_file)

            retries = 0
            while retries <= max_retries:
                cmd = self._build_worker_command(matches_file)
                try:
                    proc = subprocess.Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    proc.wait()
                    if proc.returncode == 0:
                        break
                    print(f"Worker failed for {matches_file} (rc={proc.returncode}), "
                          f"retry {retries + 1}/{max_retries + 1}")
                except Exception as e:
                    print(f"Worker error for {matches_file}: {e}")
                retries += 1

        return outcome_files

    def _build_worker_command(self, matches_file: Path) -> list[str]:
        """Build the Java command for a validation worker."""
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

        return [
            "java",
            f"-Dmatches.file={matches_file}",
            "-Xmx1200m",
            "-cp", classpath,
            "com.pricepredictor.connector.ValidationWorkerMain",
        ]

    def _resolve_jar_path(self) -> Path:
        """Resolve the forge-connector fat JAR path."""
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
