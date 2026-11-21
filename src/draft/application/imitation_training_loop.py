"""The gen-1 epoch loop: training steps over shuffled length buckets, the
within-epoch progress lines, and a validate-and-checkpoint every mini-epoch until
the epochs run out or early stop fires.
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
from draft.application.imitation_batches import BatchSource, _collate, _compute_loss
from draft.application.imitation_examples import RANDOM_SEED, DraftExample
from draft.application.imitation_log import _fmt_dur, _log
from draft.application.imitation_validation import _validate, _ValReport
from draft.domain.draft_agent_model import DraftAgentModel
from draft.infrastructure.draft_agent_store import DraftAgentStore

if TYPE_CHECKING:
    from draft.application.train_draft_agent import TrainDraftAgentConfig

_STEP_LOG_INTERVAL = 15.0  # seconds between within-epoch progress lines


@dataclass(frozen=True)
class Learner:
    """The model with the optimizer, the LR schedule and the plateau controller
    that step it."""

    model: DraftAgentModel
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR
    plateau: _PlateauLR


@dataclass(frozen=True)
class TrainingData:
    """The two splits (validation sorted by length), what their batches are built
    from, and the optimizer steps in one epoch."""

    train: list[DraftExample]
    val: list[DraftExample]
    source: BatchSource
    n_steps: int


@dataclass(frozen=True)
class CheckpointTargets:
    """Where the run's checkpoints go: the store, ``latest.pt``, the best
    checkpoint's path and the config recorded in each."""

    store: DraftAgentStore
    latest_path: Path
    best_path: Path
    train_config: dict[str, Any]


@dataclass
class TrainingProgress:
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


@dataclass
class _LossWindow:
    """Training losses summed over a run of steps."""

    imitation: float = 0.0
    critic: float = 0.0
    n: int = 0

    def add(self, imitation: float, critic: float) -> None:
        self.imitation += imitation
        self.critic += critic
        self.n += 1

    def means(self) -> tuple[float, float]:
        """The mean imitation and critic losses; 0.0 over no steps."""
        imitation = self.imitation / self.n if self.n else 0.0
        critic = self.critic / self.n if self.n else 0.0
        return imitation, critic

    def drain(self) -> tuple[float, float]:
        """The means, then an empty window."""
        means = self.means()
        self.imitation = self.critic = 0.0
        self.n = 0
        return means


@dataclass(frozen=True)
class _EvalPoint:
    """Where an eval falls and the mean training losses of the mini-epoch before it."""

    epoch: int
    step: int
    train_imit: float
    train_crit: float


class _StepLog:
    """One epoch's progress lines: one every ``_STEP_LOG_INTERVAL`` seconds, never
    on the epoch's last step."""

    def __init__(self, config: TrainDraftAgentConfig, n_batches: int) -> None:
        self._config = config
        self._n_batches = n_batches
        self._epoch_start = time.monotonic()
        self._last_log = self._epoch_start
        self._last_log_step = 0
        self._window = _LossWindow()

    def after_step(self, epoch: int, step: int, imit_v: float, crit_v: float) -> None:
        self._window.add(imit_v, crit_v)
        now = time.monotonic()
        if now - self._last_log >= _STEP_LOG_INTERVAL and step < self._n_batches:
            self._log_line(epoch, step, now)

    def _log_line(self, epoch: int, step: int, now: float) -> None:
        nb = self._n_batches
        # Windowed rate = true recent training speed; ETA uses the
        # cumulative rate (which includes the eval/checkpoint pauses)
        # so it reflects real end-to-end wall time.
        rate = (step - self._last_log_step) / (now - self._last_log)
        overall = step / (now - self._epoch_start)
        eta = (nb - step) / overall if overall > 0 else 0.0
        ti, tc = self._window.means()
        tl = self._config.imitation_weight * ti + self._config.critic_weight * tc
        _log(
            f"  epoch {epoch} step {step}/{nb} ({step / nb * 100:.0f}%) "
            f"train_loss={tl:.4f} train_imit={ti:.4f} train_crit={tc:.4f} "
            f"{rate:.1f} steps/s ETA {_fmt_dur(eta)}"
        )
        self._last_log = now
        self._last_log_step = step
        self._window.drain()


class ImitationTrainingLoop:
    """Runs the epochs of one gen-1 training run."""

    def __init__(
        self,
        config: TrainDraftAgentConfig,
        learner: Learner,
        data: TrainingData,
        targets: CheckpointTargets,
    ) -> None:
        self._config = config
        self._learner = learner
        self._data = data
        self._targets = targets
        self._rng = random.Random(RANDOM_SEED)
        self._evals_per_epoch = max(1, config.evals_per_epoch)
        self._eval_interval = max(1, data.n_steps // self._evals_per_epoch)

    def run(self, progress: TrainingProgress) -> None:
        """Train from ``progress.start_epoch`` for up to ``config.epochs`` epochs."""
        config = self._config
        _log(
            f"Training: up to {config.epochs} epochs, {self._data.n_steps} steps/epoch; "
            f"eval+checkpoint every {self._eval_interval} steps (~{self._evals_per_epoch} "
            f"mini-epochs/epoch); patience {config.patience} mini-epochs; "
            f"on {self._data.source.device}"
        )
        for epoch in range(progress.start_epoch, progress.start_epoch + config.epochs):
            if self._run_epoch(epoch, progress):
                break

    def _run_epoch(self, epoch: int, progress: TrainingProgress) -> bool:
        """One pass over the training split; True when early stop fires."""
        self._learner.model.train()
        batches = length_bucketed_batches(self._data.train, self._config.batch_size, self._rng)
        n_batches = len(batches)
        step_log = _StepLog(self._config, n_batches)
        since_eval = _LossWindow()  # train loss since the last eval (mini-epoch)
        for step, chunk in enumerate(batches, start=1):
            imit_v, crit_v = self._train_step(chunk)
            since_eval.add(imit_v, crit_v)
            step_log.after_step(epoch, step, imit_v, crit_v)
            if self._eval_due(step, n_batches) and self._evaluate(
                _eval_point(epoch, step, since_eval), progress,
            ):
                return True
        return False

    def _eval_due(self, step: int, n_batches: int) -> bool:
        return step % self._eval_interval == 0 or step == n_batches

    def _train_step(self, chunk: list[DraftExample]) -> tuple[float, float]:
        """One optimizer step; the batch's imitation and critic losses."""
        config = self._config
        learner = self._learner
        batch = _collate(chunk, self._data.source)
        losses, _, _ = _compute_loss(
            learner.model, batch, config.imitation_weight, config.critic_weight,
        )
        learner.optimizer.zero_grad()
        losses.total.backward()
        for group in learner.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"], config.max_grad_norm)
        learner.optimizer.step()
        learner.scheduler.step()
        return float(losses.imitation.detach()), float(losses.critic.detach())

    def _evaluate(self, point: _EvalPoint, progress: TrainingProgress) -> bool:
        """Validate + save latest/best; return True if early-stop is reached.

        ``point`` carries the mean training losses over the mini-epoch just
        finished, logged beside the val losses so the train-vs-val gap is
        visible on one line.
        """
        data = self._data
        report = _validate(self._learner.model, data.val, data.source, self._config)
        self._log_eval(point, report)
        new_best = progress.record(report.loss, point.epoch, point.step)
        self._apply_plateau(progress)
        self._save(self._targets.latest_path, point.epoch, progress)
        if new_best:
            self._save(self._targets.best_path, point.epoch, progress)
        self._learner.model.train()  # _validate switched to eval mode
        return self._early_stop(progress)

    def _log_eval(self, point: _EvalPoint, report: _ValReport) -> None:
        config = self._config
        mse_str = ", ".join(
            f"p{pk}={v:.4f}" for pk, v in sorted(report.per_pack_mse.items())
        )
        train_loss = (
            config.imitation_weight * point.train_imit
            + config.critic_weight * point.train_crit
        )
        # Show the live LR only when annealing is on (keeps default output
        # byte-identical to before).
        lr_str = (
            f" lr={self._learner.optimizer.param_groups[0]['lr']:.2e}"
            if config.lr_decay_patience is not None else ""
        )
        _log(
            f"  eval epoch {point.epoch} step {point.step}/{self._data.n_steps} | "
            f"train_loss={train_loss:.4f} train_imit={point.train_imit:.4f} "
            f"train_crit={point.train_crit:.4f} | val_loss={report.loss:.4f} "
            f"val_imit={report.imitation:.4f} val_crit={report.critic:.4f} "
            f"top1={report.top1:.3f} top3={report.top3:.3f} "
            f"per_pack_mse[{mse_str}]{lr_str}"
        )

    def _apply_plateau(self, progress: TrainingProgress) -> None:
        # Plateau decay: divide the LR and reset the counter so the new LR
        # gets a fresh patience window (an implicit cooldown). With
        # lr_decay_patience < patience this pre-empts early stop until the
        # floor is reached, so early stop only fires at min_lr.
        plateau = self._learner.plateau
        new_lr = _maybe_decay(progress.evals_since_best, self._config.lr_decay_patience, plateau)
        if new_lr is not None:
            _log(
                f"  LR decay #{plateau.decay_count} -> {new_lr:.2e} after "
                f"{self._config.lr_decay_patience} mini-epochs without improvement"
            )
            progress.evals_since_best = 0

    def _save(self, path: Path, epoch: int, progress: TrainingProgress) -> None:
        learner = self._learner
        source = self._data.source
        self._targets.store.save_checkpoint(
            learner.model, learner.optimizer, epoch, progress.best_val_loss,
            learner.model.config, path,
            critic_mean=source.mean, critic_std=source.std,
            train_config=self._targets.train_config,
            lr_decay_count=learner.plateau.decay_count,
        )

    def _early_stop(self, progress: TrainingProgress) -> bool:
        stop = progress.evals_since_best >= self._config.patience
        if stop:
            _log(
                f"Early stop: {progress.evals_since_best} mini-epochs without "
                f"val improvement (--patience={self._config.patience})"
            )
        return stop


def _eval_point(epoch: int, step: int, since_eval: _LossWindow) -> _EvalPoint:
    """The eval at ``step``, with the mini-epoch's mean training losses; empties
    ``since_eval`` for the next mini-epoch."""
    train_imit, train_crit = since_eval.drain()
    return _EvalPoint(epoch, step, train_imit, train_crit)
