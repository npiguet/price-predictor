"""The device every effects command runs its model on."""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
