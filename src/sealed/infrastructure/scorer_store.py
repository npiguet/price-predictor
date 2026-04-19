"""Save and load scorer model checkpoints (.pt files)."""

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
from sealed.domain.scorer_model import ScorerConfig


@dataclass(frozen=True)
class LoadedScorerCheckpoint:
    """Typed record of everything a scorer .pt file stores."""

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    epoch: int
    best_val_accuracy: float
    config: ScorerConfig


class ScorerStore:
    """Persists scorer model checkpoints using torch.save/load."""

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_accuracy: float,
        config: ScorerConfig,
        path: Path,
    ) -> None:
        save_checkpoint(
            path,
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
            },
            config,
        )

    def load_checkpoint(self, path: Path) -> LoadedScorerCheckpoint:
        payload, config = load_checkpoint(
            path, ScorerConfig, weights_only=False,
        )
        return LoadedScorerCheckpoint(
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload.get("optimizer_state_dict", {}),
            epoch=payload["epoch"],
            best_val_accuracy=payload.get("best_val_accuracy", -1.0),
            config=config,
        )
