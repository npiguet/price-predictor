"""The JSON results files the probe scripts write."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plain(obj, digits: int | None = None):
    """``obj`` as plain JSON values: numpy scalars and arrays unwrapped, sets
    sorted, paths as strings, keys as strings, and NaN or infinity as null.
    ``digits`` rounds every float."""
    if isinstance(obj, dict):
        return {str(k): plain(v, digits) for k, v in obj.items()}
    if isinstance(obj, (set, frozenset)):
        return sorted(str(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return plain(obj.tolist(), digits)
    if isinstance(obj, (list, tuple)):
        return [plain(v, digits) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not np.isfinite(f):
            return None
        return f if digits is None else round(f, digits)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj, *, digits: int | None = None, indent: int = 2,
               sort_keys: bool = False) -> None:
    """Write ``plain(obj)`` to ``path``, creating its directory; never a bare NaN."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(plain(obj, digits), indent=indent, sort_keys=sort_keys, allow_nan=False)
    path.write_text(text, encoding="utf-8")
