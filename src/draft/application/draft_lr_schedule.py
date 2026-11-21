"""The learning-rate control the three draft-agent trainers share: linear warmup
then constant, an optional plateau decay folded into it, and the re-applied LR
of a resumed optimizer.
"""

from __future__ import annotations

import math

import torch


def _reapply_resume_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """Force ``optimizer`` onto ``lr`` after loading a resumed state dict.

    ``load_state_dict`` restores both the saved ``lr`` and the stale
    ``initial_lr`` a previous run's ``LambdaLR`` baked into each param group.
    Setting ``lr`` alone is not enough: a freshly-built ``LambdaLR`` reads its
    ``base_lr`` from ``initial_lr`` (via ``setdefault``, which won't overwrite an
    existing value), then clobbers ``lr`` with ``base_lr × λ`` on the first
    ``step()`` — silently defeating a ``--lr`` override. Popping ``initial_lr``
    lets the new scheduler recapture its base from this ``lr``.
    """
    for group in optimizer.param_groups:
        group["lr"] = lr
        group.pop("initial_lr", None)


class _PlateauLR:
    """ReduceLROnPlateau-style multiplier folded into the warmup scheduler.

    Holds a discrete ``decay_count``; the effective LR is
    ``base_lr * factor ** decay_count`` (times the warmup ramp). The eval loop
    calls :meth:`decay` after ``lr_decay_patience`` mini-epochs without a new
    best val_loss, bounded below by ``min_lr``. ``decay_count`` is persisted in
    the checkpoint so a resumed run continues annealing where it left off.
    """

    def __init__(
        self, base_lr: float, factor: float, min_lr: float, decay_count: int = 0,
    ) -> None:
        self.base_lr = base_lr
        self.factor = factor
        self.min_lr = min_lr
        self.decay_count = decay_count

    def multiplier(self) -> float:
        return self.factor ** self.decay_count

    def current_lr(self) -> float:
        return self.base_lr * self.multiplier()

    def can_decay(self) -> bool:
        """True if one more decay stays at/above ``min_lr``."""
        return self.base_lr * self.factor ** (self.decay_count + 1) >= self.min_lr

    def decay(self) -> float:
        self.decay_count += 1
        return self.current_lr()


def _maybe_decay(
    evals_since_best: int, patience: int | None, controller: _PlateauLR,
) -> float | None:
    """Decay the LR if stuck for ``patience`` evals and the floor allows it.

    Returns the new LR on decay, else ``None``. ``patience is None`` (the
    feature switch) disables annealing entirely.
    """
    if patience is None:
        return None
    if evals_since_best >= patience and controller.can_decay():
        return controller.decay()
    return None


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float,
    controller: _PlateauLR | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_frac`` of total steps, then constant.

    When a plateau ``controller`` is given, the per-step multiplier also folds
    in its (live) decay factor, so a decay recorded mid-run takes effect on the
    next batch.
    """
    warmup_steps = max(1, int(math.ceil(total_steps * warmup_frac)))

    def schedule(step: int) -> float:
        warm = float(step) / float(warmup_steps) if step < warmup_steps else 1.0
        return warm * (controller.multiplier() if controller is not None else 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
