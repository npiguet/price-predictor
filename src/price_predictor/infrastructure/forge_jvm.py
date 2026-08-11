"""Shared helpers for invoking forge-connector and Forge JVM main classes.

Centralises the four classpath/JAR/JVM-launching duplications that previously
lived in `run_convert` and the three sealed connectors, plus the
Windows-vs-POSIX process-tree termination helper.
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

CONNECTOR_JAR_NAME = "forge-connector-1.0.0-SNAPSHOT-jar-with-dependencies.jar"
FORGE_VERSION = "2.0.13-SNAPSHOT"


def project_root() -> Path:
    """Locate the price-predictor project root from this file."""
    return Path(__file__).resolve().parents[3]


def forge_dir() -> Path:
    """Return the sibling Forge checkout directory (../forge)."""
    return project_root().parent / "forge"


def forge_module_jar(module: str) -> Path:
    """Return the path to a Forge module JAR (e.g. ``forge-game``)."""
    return forge_dir() / module / "target" / f"{module}-{FORGE_VERSION}.jar"


def resolve_connector_jar() -> Path:
    """Return the forge-connector fat JAR path. Raises if not built yet."""
    jar = project_root() / "forge-connector" / "target" / CONNECTOR_JAR_NAME
    if not jar.exists():
        raise FileNotFoundError(
            f"Connector JAR not found at {jar}\n"
            "Build it first: cd forge-connector && mvn package -DskipTests"
        )
    return jar


def build_forge_classpath(
    *,
    include_full_runtime: bool = True,
    include_dependency_glob: bool = False,
) -> str:
    """Build the classpath needed to run a forge-connector main class.

    Args:
        include_full_runtime: include forge-gui and forge-ai in addition to
            forge-game and forge-core. Required by every entry point that
            calls ``ForgeEnvironmentInitializer.initialize()`` (which uses
            ``GuiBase`` / ``GuiHeadless`` / ``FModel``) — i.e. all of them,
            including ``ConvertMain``.
        include_dependency_glob: append ``forge-game/target/dependency/*``
            (the maven-dependency-plugin output) — required by ``ConvertMain``.
    """
    parts: list[str] = [
        str(resolve_connector_jar()),
        str(forge_module_jar("forge-game")),
        str(forge_module_jar("forge-core")),
    ]
    if include_full_runtime:
        parts.append(str(forge_module_jar("forge-gui")))
        parts.append(str(forge_module_jar("forge-ai")))
    if include_dependency_glob:
        parts.append(str(forge_dir() / "forge-game" / "target" / "dependency" / "*"))
    return os.pathsep.join(parts)


def build_jvm_command(
    *,
    main_class: str,
    classpath: str,
    system_properties: dict[str, str] | None = None,
    xmx: str | None = None,
    main_args: list[str] | None = None,
) -> list[str]:
    """Construct a ``java`` subprocess command list."""
    cmd: list[str] = ["java"]
    if system_properties:
        for key, value in system_properties.items():
            cmd.append(f"-D{key}={value}")
    if xmx:
        cmd.append(f"-Xmx{xmx}")
    cmd.extend(["-cp", classpath, main_class])
    if main_args:
        cmd.extend(main_args)
    return cmd


def run_forge_worker(
    main_class: str,
    *,
    main_args: list[str] | None = None,
    xmx: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    """Run a one-shot forge-connector main class and return the result.

    Raises FileNotFoundError if ``java`` is not on PATH.
    """
    cmd = build_jvm_command(
        main_class=main_class,
        classpath=build_forge_classpath(),
        main_args=main_args,
        xmx=xmx,
    )
    capture = input_text is not None
    try:
        return subprocess.run(
            cmd,
            input=input_text,
            capture_output=capture,
            text=capture,
            encoding="utf-8" if capture else None,
            check=False,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Java not found. Ensure java is on PATH."
        ) from exc


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a subprocess and its entire child tree.

    On Windows, ``proc.terminate()`` only kills the parent — child JVM
    processes survive as orphans. ``taskkill /F /T`` walks the PID tree
    and forces termination. On POSIX, ``os.killpg`` does the same when
    the process was launched with ``start_new_session=True``.
    """
    if proc.poll() is not None:
        return
    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


class ForgeWorkerPool:
    """Supervise a pool of Forge JVM workers that append to one output file.

    Spawns ``worker_count`` workers, monitors each in its own thread, restarts
    the ones that die, reports status every ``status_interval`` seconds, recycles
    the longest-running worker on each of those ticks, and shuts down cleanly on
    SIGINT/SIGTERM.

    Recycling is a liveness measure, not a throughput one: Forge workers
    sometimes hang in near-infinite loops, and killing the oldest each tick
    bounds how long one can occupy a slot.

    Progress is measured by counting lines in ``output_path`` rather than by
    workers reporting back, so a worker needs no channel to the supervisor.

    Used by ``sealed match-outcomes`` and ``draft play-draft-games``; the printed
    lines are part of both commands' operator contract and must not drift.
    """

    STATUS_INTERVAL = 60  # seconds between status reports

    def __init__(
        self,
        *,
        worker_count: int,
        spawn_worker: Callable[[int], subprocess.Popen],
        output_path: Path,
        status_interval: int | None = None,
        should_stop: Callable[[int], bool] | None = None,
    ) -> None:
        """
        Args:
            worker_count: how many workers to keep alive.
            spawn_worker: called with a worker index, returns a started process.
            output_path: the file workers append to; its line count is progress.
            status_interval: seconds between status lines (default 60).
            should_stop: called with the current line count after each status
                line; returning True ends the run. None means run until signalled.
        """
        self._worker_count = worker_count
        self._spawn_worker = spawn_worker
        self._output_path = Path(output_path)
        self._status_interval = (
            self.STATUS_INTERVAL if status_interval is None else status_interval
        )
        self._should_stop = should_stop
        self._shutdown_event = threading.Event()
        self._interrupted = False
        self._processes: list[subprocess.Popen] = []
        self._start_times: dict[subprocess.Popen, float] = {}
        self._processes_lock = threading.Lock()

    @property
    def interrupted(self) -> bool:
        """True when the run ended on SIGINT/SIGTERM rather than by itself.

        Callers need this to tell a clean finish from a Ctrl-C, which decide
        different exit codes.
        """
        return self._interrupted

    def run(self) -> None:
        """Start all workers and block until shutdown."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print(f"Starting {self._worker_count} workers...")

        monitor_threads = []
        for i in range(self._worker_count):
            t = threading.Thread(target=self._monitor_worker, args=(i,), daemon=True)
            t.start()
            monitor_threads.append(t)

        self._supervisor_loop()

        for t in monitor_threads:
            t.join(timeout=10)

        print("Done.")

    def output_line_count(self) -> int:
        """Count lines in the output file (= number of completed matches)."""
        if not self._output_path.exists():
            return 0
        try:
            with open(self._output_path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    def request_shutdown(self) -> None:
        """Ask the pool to stop; the supervisor loop exits at its next tick."""
        self._shutdown_event.set()

    def _monitor_worker(self, worker_id: int) -> None:
        """Monitor one worker, restarting it on crash until shutdown."""
        while not self._shutdown_event.is_set():
            try:
                proc = self._spawn_worker(worker_id)
                with self._processes_lock:
                    self._processes.append(proc)
                    self._start_times[proc] = time.monotonic()

                proc.wait()

                with self._processes_lock:
                    if proc in self._processes:
                        self._processes.remove(proc)
                    self._start_times.pop(proc, None)

                if self._shutdown_event.is_set():
                    return

                print(f"Worker {worker_id} exited (code {proc.returncode}), restarting...")

            except Exception as exc:
                if not self._shutdown_event.is_set():
                    print(f"Monitor error for worker {worker_id}: {exc}")

    #: How often the stop condition is polled, in seconds. Far shorter than the
    #: status interval: a run with a small target would otherwise overshoot it by
    #: everything the workers finish before the next status line.
    STOP_POLL_INTERVAL = 2

    def _supervisor_loop(self) -> None:
        """Report status and recycle the oldest worker every status interval.

        The stop condition is polled far more often than that, so a small
        ``--n-pairings`` stops near its target instead of at the next status tick.
        """
        start_time = time.monotonic()
        last_count = 0
        last_time = start_time
        next_status = start_time + self._status_interval

        while not self._shutdown_event.is_set():
            if self._should_stop is not None and self._should_stop(
                self.output_line_count()
            ):
                self._shutdown_event.set()
                break

            timeout = self._sleep_slice(next_status)
            self._shutdown_event.wait(timeout=timeout)
            if self._shutdown_event.is_set():
                break

            now = time.monotonic()
            if now < next_status:
                continue  # a stop-poll slice, not a status tick

            last_count, last_time = self._report_status(
                start_time, now, last_count, last_time,
            )
            next_status = now + self._status_interval
            if self._should_stop is not None and self._should_stop(last_count):
                self._shutdown_event.set()
                break
            self._kill_oldest_worker()

        self._terminate_all()
        print(f"Shutting down, terminating {self._worker_count} workers...")

    def _sleep_slice(self, next_status: float) -> float:
        """How long to wait next: the stop-poll slice, or the rest of the interval."""
        remaining = next_status - time.monotonic()
        if self._should_stop is None:
            return max(0.0, remaining)
        return max(0.0, min(float(self.STOP_POLL_INTERVAL), remaining))

    def _report_status(
        self,
        start_time: float,
        now: float,
        last_count: int,
        last_time: float,
    ) -> tuple[int, float]:
        elapsed = now - start_time
        interval = now - last_time

        count = self.output_line_count()
        delta = count - last_count
        rate = delta / interval * 60 if interval > 0 else 0.0

        with self._processes_lock:
            alive = sum(1 for p in self._processes if p.poll() is None)

        print(
            f"[{elapsed:.0f}s] {count} matches completed"
            f" | {rate:.1f} matches/min"
            f" | {alive}/{self._worker_count} workers alive"
        )
        return count, now

    def _kill_oldest_worker(self) -> None:
        """Kill the longest-running worker so the monitor thread restarts it fresh.

        Why: Forge workers sometimes get stuck in infinite or near-infinite loops,
        holding a slot indefinitely. Recycling the oldest one each status interval
        bounds how long any single match can occupy a worker. Introduced in commit
        eda4505.
        """
        with self._processes_lock:
            alive = [(p, t) for p, t in self._start_times.items() if p.poll() is None]
            if not alive:
                return
            oldest_proc, oldest_time = min(alive, key=lambda x: x[1])
            age = time.monotonic() - oldest_time

        kill_process_tree(oldest_proc)
        print(f"Recycled longest-running worker (PID {oldest_proc.pid}, age {age:.0f}s)")

    def _terminate_all(self) -> None:
        """Terminate all running worker processes."""
        with self._processes_lock:
            procs = list(self._processes)

        for proc in procs:
            kill_process_tree(proc)

    def _handle_signal(self, signum, frame) -> None:
        """Signal handler: set shutdown event to stop workers."""
        self._interrupted = True
        self._shutdown_event.set()
