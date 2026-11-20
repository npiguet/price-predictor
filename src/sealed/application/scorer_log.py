"""The ``train-scorer`` progress log: one timestamped line per call."""

from __future__ import annotations

from datetime import datetime


def _log(message: str) -> None:
    """Print a timestamped progress line to stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
