"""PoolModelStore: atomic save/load of training checkpoints."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


@dataclass
class CheckpointData:
    pool_transformer_state_dict: dict
    optimizer_state_dict: dict
    training_state: Any  # TrainingState instance


class PoolModelStore:
    def save(
        self,
        path: Path,
        model: Any,
        optimizer: Any,
        training_state: Any,
    ) -> None:
        """Atomically save checkpoint to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pool_transformer_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_state": training_state,
        }
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.pt")
        os.close(fd)
        try:
            torch.save(payload, tmp)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load(self, path: Path) -> CheckpointData:
        """Load checkpoint. Raises FileNotFoundError if path is absent."""
        if not path.exists():
            raise FileNotFoundError(str(path))
        payload = torch.load(path, weights_only=False)
        return CheckpointData(
            pool_transformer_state_dict=payload["pool_transformer_state_dict"],
            optimizer_state_dict=payload["optimizer_state_dict"],
            training_state=payload["training_state"],
        )

    def save_timestamped(
        self,
        base_path: Path,
        model: Any,
        optimizer: Any,
        training_state: Any,
    ) -> Path:
        """Save timestamped checkpoint under base_path.parent/checkpoints/{timestamp}.pt.

        Uses dashes in time portion to keep filenames valid on Windows.
        """
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        dest = base_path.parent / "checkpoints" / f"{ts}.pt"
        self.save(dest, model, optimizer, training_state)
        return dest
