"""The gen-2 RL epoch loop: a GAE refresh from the current critic, training steps
over shuffled length buckets with their progress lines, and a validate, decay
and checkpoint every mini-epoch until the epochs run out or early stop fires.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from draft.application.draft_lr_schedule import _maybe_decay, _PlateauLR
from draft.application.draft_training_common import length_bucketed_batches
from draft.application.imitation_batches import BatchSource
from draft.application.rl_advantage_refresh import (
    AdvantageRefresh,
    CriticPass,
    _precompute_advantages,
)
from draft.application.rl_batches import LossSettings, PolicyPair, _collate, _compute_loss
from draft.application.rl_coef_schedule import _CoefSchedule
from draft.application.rl_examples import RANDOM_SEED, RLExample
from draft.application.rl_log import _STEP_LOG_INTERVAL, _fmt_dur, _log
from draft.application.rl_validation import _validate, _ValReport
from draft.infrastructure.draft_agent_store import DraftAgentStore

if TYPE_CHECKING:
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig


@dataclass(frozen=True)
class RLLearner:
    """The model and its KL reference, the optimizer, the LR schedule and plateau
    controller that step it, and the entropy and KL coefficient schedules."""

    models: PolicyPair
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    plateau: _PlateauLR
    entropy_sched: _CoefSchedule
    kl_sched: _CoefSchedule

    def loss_settings(self, config: TrainDraftAgentRLConfig) -> LossSettings:
        """The loss weights at this point of the run."""
        return LossSettings(config, self.entropy_sched.coef, self.kl_sched.coef)


@dataclass(frozen=True)
class RLTrainingData:
    """The two splits (validation sorted by length), the learner trajectories the
    GAE refresh walks, what batches are built from, and the steps in one epoch."""

    train: list[RLExample]
    val: list[RLExample]
    trajectories: list[list[RLExample]]
    source: BatchSource
    n_steps: int


@dataclass(frozen=True)
class RLCheckpointTargets:
    """Where the run's checkpoints go: the store, ``latest.pt``, the best
    checkpoint's path, and the config and generation recorded in each."""

    store: DraftAgentStore
    latest_path: Path
    best_path: Path
    train_config: dict[str, Any]
    generation: int


@dataclass
class RLTrainingProgress:
    """The run's best validation loss, where it was reached, and the evals since."""

    start_epoch: int
    best_val_loss: float
    best_at: tuple[int, int]
    evals_since_best: int = 0

    def record(self, loss: float, epoch: int, step: int) -> bool:
        """Count one eval's ``loss``; True when it is a new best."""
        new_best = loss < self.best_val_loss
        if new_best:
            self.best_val_loss = loss
            self.best_at = (epoch, step)
            self.evals_since_best = 0
        else:
            self.evals_since_best += 1
        return new_best


class _StepLog:
    """One epoch's progress lines: one every ``_STEP_LOG_INTERVAL`` seconds, never
    on the epoch's last step."""

    def __init__(self, n_batches: int) -> None:
        self._n_batches = n_batches
        self._epoch_start = time.monotonic()
        self._last_log = self._epoch_start

    def after_step(self, epoch: int, step: int, total: torch.Tensor) -> None:
        now = time.monotonic()
        if now - self._last_log >= _STEP_LOG_INTERVAL and step < self._n_batches:
            nb = self._n_batches
            overall = step / (now - self._epoch_start)
            eta = (nb - step) / overall if overall > 0 else 0.0
            _log(
                f"  epoch {epoch} step {step}/{nb} "
                f"({step / nb * 100:.0f}%) total={float(total.detach()):.4f} "
                f"ETA {_fmt_dur(eta)}"
            )
            self._last_log = now


def _rl_metadata(config: TrainDraftAgentRLConfig, generation: int) -> dict[str, Any]:
    """What makes a checkpoint self-describing for the next RL cycle."""
    return {
        "generation": generation,
        "reference_checkpoint": str(config.resume or config.checkpoint),
        "algorithm": "reinforce+gae",
        "gae_lambda": config.gae_lambda,
        "kl_coef": config.kl_coef,
        "entropy_coef": config.entropy_coef,
        "value_weight": config.value_weight,
        "rollout_temperature": float(config.rollout_temperature or 1.0),
    }


class RLTrainingLoop:
    """Runs the epochs of one gen-2 RL run."""

    def __init__(
        self,
        config: TrainDraftAgentRLConfig,
        learner: RLLearner,
        data: RLTrainingData,
        targets: RLCheckpointTargets,
    ) -> None:
        self._config = config
        self._learner = learner
        self._data = data
        self._targets = targets
        self._rng = random.Random(RANDOM_SEED)
        self._evals_per_epoch = max(1, config.evals_per_epoch)
        self._eval_interval = max(1, data.n_steps // self._evals_per_epoch)
        self._refresh = AdvantageRefresh(
            CriticPass(data.source, config.batch_size), config.gae_lambda,
        )

    def run(self, progress: RLTrainingProgress) -> None:
        """Train from ``progress.start_epoch`` for up to ``config.epochs`` epochs."""
        config = self._config
        _log(
            f"Training: up to {config.epochs} epochs, {self._data.n_steps} steps/epoch; "
            f"eval every {self._eval_interval} steps (~{self._evals_per_epoch}/epoch); "
            f"patience {config.patience}"
        )
        for epoch in range(progress.start_epoch, progress.start_epoch + config.epochs):
            if self._run_epoch(epoch, progress):
                break

    def _run_epoch(self, epoch: int, progress: RLTrainingProgress) -> bool:
        """A GAE refresh, then one pass over the training split; True when early
        stop fires."""
        model = self._learner.models.model
        # Refresh GAE advantages from the current critic (research D3).
        _precompute_advantages(model, self._data.trajectories, self._refresh, epoch=epoch)
        model.train()
        batches = length_bucketed_batches(self._data.train, self._config.batch_size, self._rng)
        n_batches = len(batches)
        step_log = _StepLog(n_batches)
        for step, chunk in enumerate(batches, start=1):
            total = self._train_step(chunk)
            step_log.after_step(epoch, step, total)
            if self._eval_due(step, n_batches) and self._evaluate(epoch, step, progress):
                return True
        return False

    def _eval_due(self, step: int, n_batches: int) -> bool:
        return step % self._eval_interval == 0 or step == n_batches

    def _train_step(self, chunk: list[RLExample]) -> torch.Tensor:
        """One optimizer step; the batch's total loss."""
        learner = self._learner
        batch = _collate(chunk, self._data.source)
        out = _compute_loss(learner.models, batch, learner.loss_settings(self._config))
        learner.optimizer.zero_grad()
        out.total.backward()
        for group in learner.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], self._config.max_grad_norm)
        learner.optimizer.step()
        learner.scheduler.step()
        return out.total

    def _evaluate(self, epoch: int, step: int, progress: RLTrainingProgress) -> bool:
        """Validate, update the coefficients, decay the LR on a plateau and save
        latest/best; True when early stop is reached."""
        learner = self._learner
        report = _validate(
            learner.models, self._data.val, self._data.source,
            learner.loss_settings(self._config),
        )
        self._log_eval(epoch, step, report)
        learner.entropy_sched.update(-report.loss)
        learner.kl_sched.update(-report.loss)
        new_best = progress.record(report.loss, epoch, step)
        self._apply_plateau(progress)
        self._save(self._targets.latest_path, epoch, progress)
        if new_best:
            self._save(self._targets.best_path, epoch, progress)
        return self._early_stop(progress)

    def _log_eval(self, epoch: int, step: int, report: _ValReport) -> None:
        learner = self._learner
        _log(
            f"  eval epoch {epoch} step {step}/{self._data.n_steps} | "
            f"val_loss={report.loss:.4f} policy={report.policy:.4f} "
            f"value={report.value:.4f} entropy={report.entropy:.4f} "
            f"kl={report.kl:.4f} | ent_coef={learner.entropy_sched.coef:.4g} "
            f"kl_coef={learner.kl_sched.coef:.4g}"
        )

    def _apply_plateau(self, progress: RLTrainingProgress) -> None:
        plateau = self._learner.plateau
        new_lr = _maybe_decay(progress.evals_since_best, self._config.lr_decay_patience, plateau)
        if new_lr is not None:
            _log(f"  LR decay #{plateau.decay_count} -> {new_lr:.2e}")
            progress.evals_since_best = 0

    def _save(self, path: Path, epoch: int, progress: RLTrainingProgress) -> None:
        learner = self._learner
        model = learner.models.model
        source = self._data.source
        targets = self._targets
        targets.store.save_checkpoint(
            model, learner.optimizer, epoch, progress.best_val_loss, model.config, path,
            critic_mean=source.mean, critic_std=source.std,
            train_config=targets.train_config,
            lr_decay_count=learner.plateau.decay_count,
            rl_metadata=_rl_metadata(self._config, targets.generation),
        )

    def _early_stop(self, progress: RLTrainingProgress) -> bool:
        stop = progress.evals_since_best >= self._config.patience
        if stop:
            _log(
                f"Early stop: {progress.evals_since_best} evals without val "
                f"improvement (--patience={self._config.patience})"
            )
        return stop
