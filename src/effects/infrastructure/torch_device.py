"""The device every effects command runs its model on, and the host memory
a copy to it starts from."""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pinned(value):
    """``value`` with every tensor in it copied to page-locked host memory, so
    a ``non_blocking`` copy does not hold the host."""
    if isinstance(value, torch.Tensor):
        return value.pin_memory()
    if isinstance(value, dict):
        return {key: pinned(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(pinned(item) for item in value)
    return value
