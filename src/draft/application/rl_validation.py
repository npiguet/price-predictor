"""The held-out RL objective, the best-checkpoint metric (research D9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from draft.application.rl_batches import _collate, _compute_loss
from draft.application.rl_examples import RLExample
from draft.domain.draft_agent_model import DraftAgentModel

if TYPE_CHECKING:
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig



@dataclass
class _ValReport:
    loss: float
    policy: float
    value: float
    entropy: float
    kl: float


def _validate(
    model: DraftAgentModel,
    reference: DraftAgentModel,
    val: list[RLExample],
    table: np.ndarray,
    mean: float,
    std: float,
    config: TrainDraftAgentRLConfig,
    entropy_coef: float,
    kl_coef: float,
    device: torch.device,
) -> _ValReport:
    """Held-out RL objective (the best-checkpoint metric, research D9)."""
    model.eval()
    if not val:
        return _ValReport(0.0, 0.0, 0.0, 0.0, 0.0)
    tot = pol = val_l = ent = kl = 0.0
    n = 0
    with torch.no_grad():
        for s in range(0, len(val), config.batch_size):
            chunk = val[s:s + config.batch_size]
            batch = _collate(chunk, table, mean, std, device)
            out = _compute_loss(
                model, reference, batch, config, entropy_coef, kl_coef,
            )
            bn = len(chunk)
            tot += float(out.total) * bn
            pol += float(out.policy) * bn
            val_l += float(out.value) * bn
            ent += float(out.entropy) * bn
            kl += float(out.kl) * bn
            n += bn
    model.train()
    return _ValReport(tot / n, pol / n, val_l / n, ent / n, kl / n)
