"""Save and load picker model checkpoints (.pt files).

Thin mirror of ``ScorerStore``: the same ``(model_state_dict,
optimizer_state_dict, epoch, best_val_*, config, train_config)`` payload, with
the metric renamed ``best_val_reward`` to reflect the picker's REINFORCE
val-reward semantics. Picker checkpoints carry picker weights only — no scorer
or encoder weights (FR-038).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.optim

from price_predictor.infrastructure.torch_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from sealed.domain.picker_model import PickerConfig


@dataclass(frozen=True)
class LoadedPickerCheckpoint:
    """Typed record of everything a picker .pt file stores."""

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    epoch: int
    best_val_reward: float
    config: PickerConfig
    train_config: dict[str, Any] | None = None


class PickerStore:
    """Persists picker model checkpoints using torch.save/load."""

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_reward: float,
        config: PickerConfig,
        path: Path,
        *,
        train_config: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_reward": best_val_reward,
        }
        if train_config is not None:
            payload["train_config"] = train_config
        save_checkpoint(path, payload, config)

    def load_checkpoint(self, path: Path) -> LoadedPickerCheckpoint:
        payload, config = load_checkpoint(
            path, PickerConfig, weights_only=False,
        )
        return LoadedPickerCheckpoint(
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload.get("optimizer_state_dict", {}),
            epoch=payload["epoch"],
            best_val_reward=payload.get("best_val_reward", float("-inf")),
            config=config,
            train_config=payload.get("train_config"),
        )
