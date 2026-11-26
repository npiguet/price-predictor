"""Shared helpers for saving/loading torch checkpoint files with dataclass configs."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import torch

from price_predictor.infrastructure.atomic_write import write_atomically

T = TypeVar("T")


def copy_atomically(source: Path, target: Path) -> None:
    """Copy ``source`` to ``target`` through :func:`write_atomically`."""
    write_atomically(target, lambda temporary: shutil.copyfile(source, temporary))


def save_checkpoint(path: Path, payload: dict[str, Any], config: Any) -> None:
    """Write ``{**payload, "config": asdict(config)}`` via ``torch.save``,
    atomically."""
    write_atomically(
        path, lambda temporary: torch.save({**payload, "config": asdict(config)}, temporary),
    )


def load_checkpoint(
    path: Path,
    config_cls: type[T],
    *,
    weights_only: bool = True,
    safe_globals: list[type] | None = None,
) -> tuple[dict[str, Any], T]:
    """Load a checkpoint and return ``(payload_without_config, config)``.

    Handles both legacy checkpoints (config stored as a dataclass instance)
    and current ones (config stored as a dict via ``asdict``).
    """
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": weights_only}
    if weights_only and safe_globals:
        ctx = torch.serialization.safe_globals(safe_globals)
    else:
        ctx = None

    if ctx is not None:
        with ctx:
            raw = torch.load(path, **kwargs)
    else:
        raw = torch.load(path, **kwargs)

    raw_config = raw.pop("config")
    config = (
        raw_config
        if isinstance(raw_config, config_cls)
        else config_cls(**raw_config)
    )
    return raw, config
