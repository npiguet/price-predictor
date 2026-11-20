"""The ``train-encoder`` modules' shared pieces: the FR-022 hardcoded
constants and the timestamped progress logger every stage of the command
writes through.
"""

from __future__ import annotations

from datetime import datetime

# ── Hardcoded constants (FR-022) ────────────────────────────────────────
D_MODEL: int = 256
FF_DIM: int = 1024
VAL_FRACTION: float = 0.2
RANDOM_SEED: int = 42


def _log(message: str) -> None:
    """Print a timestamped progress line to stdout."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
