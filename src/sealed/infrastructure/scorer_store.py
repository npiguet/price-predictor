"""Save and load scorer model checkpoints (.pt files)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from sealed.domain.scorer_model import ScorerConfig


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

    def load_checkpoint(self, path: Path) -> dict:
        checkpoint = torch.load(path, weights_only=False)
        raw = checkpoint["config"]
        if isinstance(raw, dict):
            checkpoint["config"] = ScorerConfig(**raw)
        elif not isinstance(raw, ScorerConfig):
            raise ValueError(f"Unexpected checkpoint config type: {type(raw).__name__}")
        return checkpoint
