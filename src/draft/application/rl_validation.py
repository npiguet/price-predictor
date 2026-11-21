"""The held-out RL objective, the best-checkpoint metric (research D9).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from draft.application.imitation_batches import BatchSource
from draft.application.rl_batches import LossSettings, PolicyPair, _collate, _compute_loss
from draft.application.rl_examples import RLExample


@dataclass
class _ValReport:
    loss: float
    policy: float
    value: float
    entropy: float
    kl: float


def _validate(
    models: PolicyPair, val: list[RLExample], source: BatchSource, settings: LossSettings,
) -> _ValReport:
    """Held-out RL objective (the best-checkpoint metric, research D9)."""
    model = models.model
    batch_size = settings.config.batch_size
    model.eval()
    if not val:
        return _ValReport(0.0, 0.0, 0.0, 0.0, 0.0)
    tot = pol = val_l = ent = kl = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(val), batch_size):
            chunk = val[s:s + batch_size]
            batch = _collate(chunk, source)
            out = _compute_loss(models, batch, settings)
            bn = len(chunk)
            tot += float(out.total) * bn
            pol += float(out.policy) * bn
            val_l += float(out.value) * bn
            ent += float(out.entropy) * bn
            kl += float(out.kl) * bn
            n += bn
    model.train()
    return _ValReport(tot / n, pol / n, val_l / n, ent / n, kl / n)
