"""One hash over JSON-shaped data, insensitive to how its mappings were built.

Shared because a corpus manifest's digest is pinned by every checkpoint
trained against it, so no caller may drift from it by a separator.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_digest(data: Any) -> bytes:
    """16 bytes of blake2b over ``data`` rendered as canonical JSON."""
    return hashlib.blake2b(
        json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        digest_size=16,
    ).digest()
