"""Launch and manage Java evaluation processes for scorer evaluation."""

from __future__ import annotations

import platform
import subprocess
from pathlib import Path


class EvaluationConnector:
    """Coordinates Java processes for scorer evaluation."""

    def build_forge_decks(self, pools: list[list[str]]) -> list[list[str]]:
        """Build one Forge deck per pool using DeckBuilderMain.

        Sends all pools to a single JVM invocation via stdin (one pool per line,
        pipe-separated card names) and reads back one built deck per line from
        stdout. This avoids N separate JVM startups with Forge initialization.

        Args:
            pools: list of pools, each pool is a list of card names.

        Returns:
            list of decks, each deck is a list of 40 card names.
        """
        cmd = self._build_deck_builder_command()
        stdin_text = "\n".join("|".join(pool) for pool in pools) + "\n"

        result = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"DeckBuilderMain failed (rc={result.returncode}):\n{result.stderr}"
            )

        decks = []
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line:
                decks.append(line.split("|"))
        return decks

    def outcome_file_path(self, matches_file: Path) -> Path:
        """Derive the outcomes file path from a matches file path."""
        return Path(str(matches_file) + "-outcomes.txt")

    def launch_workers(
        self,
        split_files: list[Path],
        best_of: int = 3,
        max_retries: int = 1,
    ) -> list[Path]:
        """Launch Java workers in parallel and wait for completion.

        Returns outcome file paths in the same order as split_files.
        """
        outcome_files = [self.outcome_file_path(f) for f in split_files]

        # Start all workers in parallel
        processes = []
        for matches_file in split_files:
            cmd = self._build_worker_command(matches_file, best_of=best_of)
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            processes.append((proc, matches_file))

        # Wait for all and retry failures
        for proc, matches_file in processes:
            proc.wait()
            if proc.returncode != 0:
                print(
                    f"Worker failed for {matches_file} (rc={proc.returncode}), "
                    f"retrying..."
                )
                for attempt in range(max_retries):
                    cmd = self._build_worker_command(matches_file, best_of=best_of)
                    try:
                        retry_proc = subprocess.Popen(
                            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        )
                        retry_proc.wait()
                        if retry_proc.returncode == 0:
                            break
                        print(
                            f"Worker retry {attempt + 1}/{max_retries} failed "
                            f"for {matches_file} (rc={retry_proc.returncode})"
                        )
                    except Exception as e:
                        print(f"Worker retry error for {matches_file}: {e}")

        return outcome_files

    def _build_worker_command(self, matches_file: Path, best_of: int = 3) -> list[str]:
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
            f"-Dbest.of={best_of}",
            "-Xmx1200m",
            "-cp", classpath,
            "com.pricepredictor.connector.ValidationWorkerMain",
        ]

    def _build_deck_builder_command(self) -> list[str]:
        """Build the Java command for the batch Forge deck builder."""
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
            "-Xmx1200m",
            "-cp", classpath,
            "com.pricepredictor.connector.DeckBuilderMain",
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
