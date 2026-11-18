"""The one progress line every stage of a probing run prints.

Nothing here does work at import time.
"""

from __future__ import annotations

from datetime import datetime


def progress(message: str) -> None:
    """One timestamped line, flushed: a probing run takes the better part of an
    hour, and without these nothing says whether it is alive or where it is."""
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)
