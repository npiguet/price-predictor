"""The ``train-picker`` training loop: the epochs of one run, each step's
loss means, a validation every ``evals_per_epoch``-th of an epoch, the
checkpoints it saves, the entropy-coefficient schedule and the early stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sealed.application.picker_pools import _shuffle_train
from sealed.application.picker_training_run import BestSoFar, EvalPoint, TrainingRun
from sealed.application.picker_training_step import _train_one_step
from sealed.application.picker_validation import (
    ValidationContext,
    ValidationReport,
    _validate,
)


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


class _EntropySchedule:
    """Val-reward-driven entropy-coefficient decay (FR-016).

    Held constant at ``initial`` until val reward has improved monotonically for
    ``decay_after`` consecutive epochs; thereafter multiplied by ``factor`` at
    the end of every epoch that fails to improve on the previous best.
    """

    def __init__(self, initial: float, decay_after: int, factor: float = 0.9) -> None:
        self.coef = initial
        self.decay_after = decay_after
        self.factor = factor
        self.armed = False
        self.streak = 0
        self.prev: float | None = None
        self.best = float("-inf")

    def update(self, val_reward: float) -> float:
        if self.prev is not None and val_reward > self.prev:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.decay_after:
            self.armed = True
        improved_best = val_reward > self.best
        if self.armed and not improved_best:
            self.coef *= self.factor
        self.prev = val_reward
        self.best = max(self.best, val_reward)
        return self.coef


def _should_stop(epochs_since_best: int, patience: int) -> bool:
    return epochs_since_best >= patience


@dataclass
class _EpochStats:
    policy_loss: float = 0.0
    entropy_loss: float = 0.0
    aux_loss: float = 0.0
    kl: float = 0.0
    n: int = 0

    def add(self, policy: float, entropy: float, aux: float, kl: float) -> None:
        self.policy_loss += policy
        self.entropy_loss += entropy
        self.aux_loss += aux
        self.kl += kl
        self.n += 1

    def mean(self) -> tuple[float, float, float, float]:
        d = max(self.n, 1)
        return (
            self.policy_loss / d, self.entropy_loss / d,
            self.aux_loss / d, self.kl / d,
        )


def _log_epoch(at: EvalPoint, stats: _EpochStats, report: ValidationReport) -> None:
    policy, entropy, aux, kl = stats.mean()
    parts = [
        f"epoch={at.epoch}",
        f"step={at.step}/{at.n_steps}",
        f"policy_loss={policy:.4f}",
        f"entropy_loss={entropy:.4f}",
        f"aux_loss={aux:.4f}",
        f"val_reward={report.mean_reward:.4f}",
    ]
    if kl != 0.0:
        parts.append(f"kl={kl:.4f}")
    if report.audit_corr is not None and not math.isnan(report.audit_corr):
        parts.append(f"audit_corr={report.audit_corr:.4f}")
    d = report.distrib
    hist = "[" + ", ".join(f"{x:.1f}" for x in d.cmc_hist) + "]"
    parts.append(
        f"| colors_mean={d.colors_mean:.2f} creatures_mean={d.creatures_mean:.2f} "
        f"type_creature_share={d.type_creature_share:.2f} cmc_hist={hist}"
    )
    _log("  ".join(parts))


class PickerTrainingLoop:
    """The epochs of one training run: a gradient step per batch of pools, and
    a validation, its checkpoints and the early-stop test at every interval."""

    def __init__(self, run: TrainingRun, best: BestSoFar) -> None:
        config = run.config
        self.run = run
        self.best = best
        self.schedule = _EntropySchedule(config.entropy_coef, config.entropy_decay_after)
        self.stats = _EpochStats()
        self.validation = ValidationContext(
            run.models.scorer, run.models.auditor, config.batch_size, run.device,
        )
        # Validate `evals_per_epoch` times per epoch (a "mini-epoch" is one such
        # interval). With the default 1 this collapses to one eval at epoch end,
        # i.e. the original behavior. Checkpoints and early-stop act on every
        # eval, so patience is denominated in mini-epochs (FR-019, FR-020).
        self.n_steps = math.ceil(len(run.train_pools) / config.batch_size)
        self.evals_per_epoch = max(1, config.evals_per_epoch)
        self.eval_interval = max(1, self.n_steps // self.evals_per_epoch)

    def train(self) -> int:
        """Train until the epoch budget or an early stop; return the last epoch."""
        config = self.run.config
        _log(
            f"Starting training loop ({config.epochs} epochs, {self.n_steps} steps/epoch, "
            f"validating every {self.eval_interval} steps = ~{self.evals_per_epoch}x/epoch; "
            f"patience={config.patience} evals) on {self.run.device}"
        )
        start_epoch = self.run.start_epoch
        last_epoch = start_epoch
        for epoch in range(start_epoch, start_epoch + config.epochs):
            last_epoch = epoch
            if self._train_epoch(epoch):
                break
        return last_epoch

    def _train_epoch(self, epoch: int) -> bool:
        """One epoch over the shuffled train pools; True when it stopped early."""
        run = self.run
        batch_size = run.config.batch_size
        shuffled = _shuffle_train(run.train_pools, run.rng)
        run.models.picker.train()
        self.stats = _EpochStats()
        for step, start in enumerate(range(0, len(shuffled), batch_size), start=1):
            group = shuffled[start:start + batch_size]
            policy, entropy, aux, kl = _train_one_step(run, group, self.schedule.coef)
            self.stats.add(policy, entropy, aux, kl)

            if step % self.eval_interval != 0 and step != self.n_steps:
                continue
            if self._evaluate(EvalPoint(epoch, step, self.n_steps)):
                return True
            run.models.picker.train()  # _validate switched the picker to eval mode
        return False

    def _evaluate(self, at: EvalPoint) -> bool:
        """Validate, log, save the checkpoints; True when the run stops early."""
        report = _validate(self.run.models.picker, self.run.val_pools, self.validation)
        self.schedule.update(report.mean_reward)
        _log_epoch(at, self.stats, report)
        self.stats = _EpochStats()
        new_best = self.best.record(report.mean_reward, at.epoch)
        self._save(at.epoch, new_best)
        patience = self.run.config.patience
        if not _should_stop(self.best.evals_since, patience):
            return False
        _log(
            f"Early stop: {self.best.evals_since} evals without val-reward "
            f"improvement (--patience={patience})"
        )
        return True

    def _save(self, epoch: int, new_best: bool) -> None:
        models, targets = self.run.models, self.run.checkpoints
        picker = models.picker
        targets.store.save_checkpoint(
            picker, models.optimizer, epoch, self.best.reward, picker.config,
            targets.latest_path, train_config=targets.train_config,
        )
        if new_best:
            targets.store.save_checkpoint(
                picker, models.optimizer, epoch, self.best.reward, picker.config,
                targets.best_path, train_config=targets.train_config,
            )
