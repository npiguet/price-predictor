"""One hash over JSON-shaped data, insensitive to how its mappings were built.

Sorted keys and fixed separators, so two values that compare equal hash equally
whatever order their dicts were filled in. Shared because a corpus manifest's
digest is pinned by every checkpoint trained against it: the rule lives in one
place so that no caller can drift from it by a separator.
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
