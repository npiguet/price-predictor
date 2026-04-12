"""Launch and manage Java evaluation processes for scorer evaluation."""

from __future__ import annotations

import platform
import subprocess
import time
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
        idle_timeout_s: float = 300.0,
        poll_interval_s: float = 10.0,
    ) -> list[Path]:
        """Launch Java workers in parallel and wait for completion.

        Watchdog: each worker is polled every ``poll_interval_s`` seconds.
        If a worker has not appended a new outcome line for ``idle_timeout_s``
        seconds, or if it exits with a non-zero code before finishing all
        matches, it is killed and restarted on the same matches file.
        ``ValidationMatchPlayer`` skips already-completed matches based on
        the outcomes file line count, so a restart resumes where the
        previous attempt left off. There is no restart cap.

        Returns outcome file paths in the same order as split_files.
        """
        outcome_files = [self.outcome_file_path(f) for f in split_files]

        def count_outcomes(path: Path) -> int:
            if not path.exists():
                return 0
            try:
                with path.open("r", encoding="utf-8") as f:
                    return sum(1 for line in f if line.strip())
            except OSError:
                return 0

        def spawn(matches_file: Path, log_file):
            cmd = self._build_worker_command(matches_file, best_of=best_of)
            # start_new_session puts the child in its own process group on
            # POSIX (no-op on Windows) so os.killpg can take down the whole
            # tree if Java spawns subprocesses.
            return subprocess.Popen(
                cmd, stdout=log_file, stderr=log_file, start_new_session=True,
            )

        def kill_proc(proc: subprocess.Popen) -> None:
            if proc.poll() is not None:
                return
            if platform.system() == "Windows":
                # taskkill /T walks the parent-child PID tree and /F forces
                # termination, killing the JVM and any child processes.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                )
            else:
                import os
                import signal
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass

        now = time.monotonic()
        workers: list[dict] = []
        for matches_file, outcomes_file in zip(split_files, outcome_files):
            log_path = matches_file.with_suffix(".worker.log")
            log_file = log_path.open("ab")
            proc = spawn(matches_file, log_file)
            workers.append({
                "proc": proc,
                "matches_file": matches_file,
                "outcomes_file": outcomes_file,
                "log_file": log_file,
                "last_count": count_outcomes(outcomes_file),
                "last_progress": now,
                "done": False,
            })

        try:
            while True:
                all_done = True
                for w in workers:
                    if w["done"]:
                        continue
                    proc = w["proc"]
                    rc = proc.poll()

                    if rc == 0:
                        print(f"Worker completed: {w['matches_file'].name}")
                        w["done"] = True
                        continue

                    if rc is not None:
                        print(f"Worker restarted (rc={rc}): {w['matches_file'].name}")
                        w["proc"] = spawn(w["matches_file"], w["log_file"])
                        w["last_count"] = count_outcomes(w["outcomes_file"])
                        w["last_progress"] = time.monotonic()
                        all_done = False
                        continue

                    current_count = count_outcomes(w["outcomes_file"])
                    now2 = time.monotonic()
                    if current_count > w["last_count"]:
                        w["last_count"] = current_count
                        w["last_progress"] = now2
                    elif now2 - w["last_progress"] >= idle_timeout_s:
                        print(f"Worker restarted (idle): {w['matches_file'].name}")
                        kill_proc(proc)
                        w["proc"] = spawn(w["matches_file"], w["log_file"])
                        w["last_count"] = count_outcomes(w["outcomes_file"])
                        w["last_progress"] = time.monotonic()
                    all_done = False

                if all_done:
                    break
                time.sleep(poll_interval_s)
        finally:
            for w in workers:
                try:
                    w["log_file"].close()
                except Exception:
                    pass

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
