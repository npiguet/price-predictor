"""Writing a file so that a kill or a refused replace leaves the old one whole.

Torch-free, so the domain layers that write a file read it too."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    """Have ``write`` fill a temporary file beside ``path`` and move it onto
    ``path``: a kill or a failed move leaves the old ``path`` whole. A move
    that fails (a file another process holds open, on Windows) raises an
    ``OSError`` naming ``path`` and removes the temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    try:
        write(temporary)
        try:
            os.replace(temporary, path)
        except OSError as error:
            raise OSError(f"Cannot replace {path}: {error}") from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
