"""Save and load draft-agent checkpoints (.pt files).

Mirrors ``sealed/infrastructure/picker_store.py``: ``model_state_dict`` +
``config`` + ``epoch`` + ``best_val_loss`` + optimizer state + training
metadata. The draft agent additionally stores the critic-target
standardization ``mean``/``std`` so inference can de-standardize the critic
back to raw scorer-score space (FR-032, FR-040, FR-041), plus the LR-plateau
``lr_decay_count`` so a resumed run continues annealing where it left off. No
encoder/scorer weights — agent weights only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
import torch.optim

from draft.domain.draft_agent_model import DraftAgentConfig
from price_predictor.infrastructure.torch_checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class LoadedDraftAgentCheckpoint:
    """Typed record of everything a draft-agent .pt file stores."""

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    epoch: int
    best_val_loss: float
    config: DraftAgentConfig
    critic_mean: float
    critic_std: float
    train_config: dict[str, Any] | None = None
    lr_decay_count: int = 0


class DraftAgentStore:
    """Persists draft-agent checkpoints using torch.save/load."""

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        best_val_loss: float,
        config: DraftAgentConfig,
        path: Path,
        *,
        critic_mean: float,
        critic_std: float,
        train_config: dict[str, Any] | None = None,
        lr_decay_count: int = 0,
    ) -> None:
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "critic_mean": critic_mean,
            "critic_std": critic_std,
            "lr_decay_count": lr_decay_count,
        }
        if train_config is not None:
            payload["train_config"] = train_config
        save_checkpoint(path, payload, config)

    def load_checkpoint(self, path: Path) -> LoadedDraftAgentCheckpoint:
        payload, config = load_checkpoint(
            path, DraftAgentConfig, weights_only=False,
        )
        return LoadedDraftAgentCheckpoint(
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload.get("optimizer_state_dict", {}),
            epoch=payload["epoch"],
            best_val_loss=payload.get("best_val_loss", float("inf")),
            config=config,
            critic_mean=payload.get("critic_mean", 0.0),
            critic_std=payload.get("critic_std", 1.0),
            train_config=payload.get("train_config"),
            lr_decay_count=payload.get("lr_decay_count", 0),
        )
