"""One hash over JSON-shaped data, insensitive to how its mappings were built.

Shared because a corpus manifest's digest is pinned by every checkpoint
trained against it, so no caller may drift from it by a separator. Two
variants over the same canonical rendering: the 16-byte blake2b the manifest
has always used, and the sha256 hex digest the frozen probe and selection sets
carry. Neither may change, since digests already on disk are compared against.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """``data`` as canonical JSON: sorted keys, no whitespace."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def canonical_digest(data: Any) -> bytes:
    """16 bytes of blake2b over ``data`` rendered as canonical JSON."""
    return hashlib.blake2b(canonical_json(data).encode("utf-8"), digest_size=16).digest()


def canonical_sha256(data: Any) -> str:
    """The sha256 hex digest of ``data`` rendered as canonical JSON."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()
