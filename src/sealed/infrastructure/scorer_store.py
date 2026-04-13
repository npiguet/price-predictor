"""Save and load scorer model checkpoints (.pt files)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

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
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
                "config": asdict(config),
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> LoadedScorerCheckpoint:
        raw = torch.load(path, weights_only=False)
        config = _parse_config(raw["config"])
        return LoadedScorerCheckpoint(
            model_state_dict=raw["model_state_dict"],
            optimizer_state_dict=raw.get("optimizer_state_dict", {}),
            epoch=raw["epoch"],
            best_val_accuracy=raw.get("best_val_accuracy", -1.0),
            config=config,
        )


def _parse_config(raw: Any) -> ScorerConfig:
    if isinstance(raw, ScorerConfig):
        return raw
    if isinstance(raw, dict):
        return ScorerConfig(**raw)
    raise ValueError(f"Unexpected checkpoint config type: {type(raw).__name__}")
