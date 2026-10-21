"""The frozen probe set: enumerated once per curated corpus, read by every run (FR-076).

``run.py --freeze-probe-set`` writes it; every probing run, and every one-off
probe that reads its items, reads it back through :func:`load_probe_set`,
which refuses a set frozen against another corpus. Its ``digest`` is the
sha256 of its canonical JSON (sorted keys, no whitespace) with ``digest``
itself left out, and every scorecard records it.

Nothing here does work at import time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_digest(payload: dict) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace), ``digest`` excluded."""
    body = {k: v for k, v in payload.items() if k != "digest"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def probe_set_path(directory: Path, corpus_digest: str) -> Path:
    """Where the set frozen against the corpus of ``corpus_digest`` lives."""
    return Path(directory) / f"knowledge-probes-set-{corpus_digest[:12]}.json"


def load_probe_set(path: Path, corpus_digest: str) -> dict:
    """The frozen set for this corpus; refuses a missing or mismatched one."""
    if not path.exists():
        raise SystemExit(
            f"no frozen probe set at {path}: run with --freeze-probe-set first "
            "(once per curated corpus)")
    probe_set = json.loads(path.read_text(encoding="utf-8"))
    if probe_set.get("corpus_digest") != corpus_digest:
        raise SystemExit(
            f"{path} was frozen against corpus digest "
            f"{probe_set.get('corpus_digest')}, not this corpus's {corpus_digest}")
    if probe_set.get("digest") != canonical_digest(probe_set):
        raise SystemExit(f"{path} does not match its own digest; it was edited")
    return probe_set
