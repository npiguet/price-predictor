"""The ``train-picker`` training loop's pieces: the entropy-coefficient
schedule, the early-stop test, the per-interval loss means and the line each
validation logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sealed.application.picker_validation import ValidationReport


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


def _log_epoch(
    epoch: int, step: int, n_steps: int, stats: _EpochStats,
    report: ValidationReport,
) -> None:
    policy, entropy, aux, kl = stats.mean()
    parts = [
        f"epoch={epoch}",
        f"step={step}/{n_steps}",
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
