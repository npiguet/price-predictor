"""The val-driven decay of the RL loss's entropy and KL coefficients.
"""

from __future__ import annotations


class _CoefSchedule:
    """Val-driven coefficient decay (entropy / KL), copied from picker_training_loop.

    Held at ``initial`` until the val metric improves for ``decay_after``
    consecutive evals, then multiplied by ``factor`` on each eval that fails to
    beat the running best. The "metric" is higher-is-better; the trainer passes
    ``-val_loss`` so a falling loss counts as improvement.
    """

    def __init__(self, initial: float, decay_after: int, factor: float = 0.9) -> None:
        self.coef = initial
        self.decay_after = decay_after
        self.factor = factor
        self.armed = False
        self.streak = 0
        self.prev: float | None = None
        self.best = float("-inf")

    def update(self, metric: float) -> float:
        if self.prev is not None and metric > self.prev:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.decay_after:
            self.armed = True
        improved_best = metric > self.best
        if self.armed and not improved_best:
            self.coef *= self.factor
        self.prev = metric
        self.best = max(self.best, metric)
        return self.coef
