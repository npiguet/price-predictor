"""What ``build-corpus`` reads from outside the records to decide or record
with: the commit its code runs from, and the keyword-definition table, read
once for every use (the closed keyword list, the display-name misfire check
and the manifest's digest).

Torch-free at import, as every corpus-side module.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


def code_version() -> str:
    """``git rev-parse HEAD`` of the checkout this code runs from; "" outside one."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True,
            text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


@dataclass(frozen=True, slots=True)
class KeywordTable:
    """The keyword-definition file as a build read it: its display names, in
    the file's order, and the SHA-256 of its bytes, which the manifest records
    as ``keyword_definitions_digest``."""

    names: tuple[str, ...]
    digest: str


def read_keyword_table(path: Path | None) -> KeywordTable | None:
    """The table at ``path``, its bytes read once; None without a path or a
    file there, which leaves the overlay keywords alone and checks no line."""
    if path is None or not Path(path).exists():
        return None
    from effects.infrastructure.tokenizer_loading import parse_keyword_definitions

    raw = Path(path).read_bytes()
    definitions = parse_keyword_definitions(raw.decode("utf-8"))
    return KeywordTable(names=tuple(definitions), digest=hashlib.sha256(raw).hexdigest())
