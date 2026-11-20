"""The ``train-picker`` run's values: the models it trains and scores with,
where it saves, what it trains on, where in training a validation falls and
the best validation so far.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from sealed.application.picker_pools import PreparedPool
from sealed.domain.picker_model import PickerModel
from sealed.domain.scorer_model import SetTransformerScorer
from sealed.infrastructure.picker_store import PickerStore

if TYPE_CHECKING:
    from sealed.application.train_picker import TrainPickerConfig


@dataclass(frozen=True)
class TrainingModels:
    """The picker and its optimizer, the frozen training scorer, the optional
    auditor, and the frozen reference picker of the KL penalty."""

    picker: PickerModel
    scorer: SetTransformerScorer
    auditor: SetTransformerScorer | None
    reference: PickerModel | None
    optimizer: torch.optim.Optimizer


@dataclass(frozen=True)
class CheckpointTargets:
    """Where every validation saves: the rolling ``latest`` and the run's
    best, through ``store``, each carrying the flattened ``train_config``."""

    store: PickerStore
    latest_path: Path
    best_path: Path
    train_config: dict[str, Any]


@dataclass(frozen=True)
class TrainingRun:
    """One training run: its models, config and device, its train and
    validation pools, its checkpoints, the shuffle's generator and the epoch
    it starts at."""

    models: TrainingModels
    config: TrainPickerConfig
    device: torch.device
    train_pools: list[PreparedPool]
    val_pools: list[PreparedPool]
    checkpoints: CheckpointTargets
    rng: random.Random
    start_epoch: int


@dataclass(frozen=True)
class EvalPoint:
    """Where in training a validation falls: its epoch, and its step of the
    epoch's ``n_steps``."""

    epoch: int
    step: int
    n_steps: int


@dataclass
class BestSoFar:
    """The best validation reward so far, its epoch, and the validations
    since it."""

    reward: float
    epoch: int
    evals_since: int = 0

    def record(self, reward: float, epoch: int) -> bool:
        """Count one validation's reward; True when it is a new best."""
        new_best = reward > self.reward
        if new_best:
            self.reward = reward
            self.epoch = epoch
            self.evals_since = 0
        else:
            self.evals_since += 1
        return new_best
