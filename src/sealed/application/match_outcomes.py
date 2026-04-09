"""MatchOutcomeSupervisor: spawns and monitors Java worker subprocesses."""

from __future__ import annotations

import signal
import subprocess
import threading
import time
from pathlib import Path

from sealed.infrastructure.match_worker_connector import MatchWorkerConnector


class MatchOutcomeSupervisor:
    """Manages a pool of Java MatchWorkerMain subprocesses.

    Spawns worker_count workers, monitors each in a dedicated thread, restarts
    crashed workers, reports status every 60 seconds, and handles clean shutdown
    on SIGINT/SIGTERM.
    """

    STATUS_INTERVAL = 60  # seconds between status reports

    def __init__(self, worker_count: int, output_path: Path) -> None:
        self._worker_count = worker_count
        self._output_path = output_path
        self._shutdown_event = threading.Event()
        self._processes: list[subprocess.Popen] = []
        self._processes_lock = threading.Lock()
        self._connector = MatchWorkerConnector()

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

        self._status_reporter_loop()

        # Wait for all monitor threads to exit
        for t in monitor_threads:
            t.join(timeout=10)

        print("Done.")

    def _monitor_worker(self, worker_id: int) -> None:
        """Monitor one worker, restarting it on crash until shutdown."""
        while not self._shutdown_event.is_set():
            try:
                proc = self._start_worker(worker_id)
                with self._processes_lock:
                    self._processes.append(proc)

                proc.wait()

                with self._processes_lock:
                    if proc in self._processes:
                        self._processes.remove(proc)

                if self._shutdown_event.is_set():
                    return

                print(f"Worker {worker_id} exited (code {proc.returncode}), restarting...")

            except Exception as exc:
                if not self._shutdown_event.is_set():
                    print(f"Monitor error for worker {worker_id}: {exc}")

    def _start_worker(self, worker_id: int) -> subprocess.Popen:
        """Start one Java worker subprocess."""
        proc = self._connector.start(self._output_path)
        print(f"Worker {worker_id} started (PID {proc.pid})")
        return proc

    def _status_reporter_loop(self) -> None:
        """Print status every STATUS_INTERVAL seconds until shutdown."""
        start_time = time.monotonic()
        last_count = 0
        last_time = start_time

        while not self._shutdown_event.is_set():
            self._shutdown_event.wait(timeout=self.STATUS_INTERVAL)
            if self._shutdown_event.is_set():
                break

            now = time.monotonic()
            elapsed = now - start_time
            interval = now - last_time

            count = self._count_output_lines()
            delta = count - last_count
            rate = delta / interval * 60 if interval > 0 else 0.0

            with self._processes_lock:
                alive = sum(1 for p in self._processes if p.poll() is None)

            print(
                f"[{elapsed:.0f}s] {count} matches completed"
                f" | {rate:.1f} matches/min"
                f" | {alive}/{self._worker_count} workers alive"
            )

            last_count = count
            last_time = now

        self._terminate_all()
        print(f"Shutting down, terminating {self._worker_count} workers...")

    def _terminate_all(self) -> None:
        """Terminate all running worker processes."""
        with self._processes_lock:
            procs = list(self._processes)

        for proc in procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _handle_signal(self, signum, frame) -> None:
        """Signal handler: set shutdown event to stop workers."""
        self._shutdown_event.set()

    def _count_output_lines(self) -> int:
        """Count lines in the output file (= number of completed matches)."""
        if not self._output_path.exists():
            return 0
        try:
            with open(self._output_path, "rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0
