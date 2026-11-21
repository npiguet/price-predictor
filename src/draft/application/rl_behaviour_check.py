"""The behaviour-anomaly check (research D6): the reference policy's log-probability
of the corpus's learner picks over a seeded sample, a gross corpus↔checkpoint
mispairing probe.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from draft.application.rl_batches import _collate, _forward_logits_critic
from draft.application.rl_examples import RANDOM_SEED, RLExample
from draft.domain.draft_agent_model import DraftAgentModel
from price_predictor.infrastructure.torch_training import masked_log_softmax

if TYPE_CHECKING:
    from draft.application.train_draft_agent_rl import TrainDraftAgentRLConfig


_PROB_FLOOR = 1e-4  # behaviour prob below this counts as an anomaly (D6).
_ANOMALY_FRACTION = 0.05  # warn if more than this fraction of learner picks anomalous.
_ANOMALY_SAMPLE_FRACTION = 0.01  # behaviour check scans only a 1% subset (D6 is a
_ANOMALY_SAMPLE_FLOOR = 2000      # gross-mispairing probe, not a full-corpus pass)…


@dataclass
class _AnomalySummary:
    n_learner: int        # learner picks actually evaluated (the sampled subset)
    n_total: int          # total learner picks available
    frac_below_floor: float
    mean_behaviour_logprob: float


def _behaviour_anomaly_summary(
    reference: DraftAgentModel,
    examples: list[RLExample],
    table: np.ndarray,
    mean: float,
    std: float,
    config: TrainDraftAgentRLConfig,
    device: torch.device,
) -> _AnomalySummary:
    """Recompute log π_ref,T(a_t) over a learner-pick subset; summarize anomalies (D6).

    Scans only a seeded 1% sample (floored at ``_ANOMALY_SAMPLE_FLOOR``) — this is
    a gross checkpoint/temperature-mispairing probe, not a quantity that needs the
    whole corpus, so it stays a fast pre-training sanity check on large corpora.
    """
    temp = float(config.rollout_temperature or 1.0)
    learner_all = [ex for ex in examples if ex.learner_active]
    n_total = len(learner_all)
    if n_total == 0:
        return _AnomalySummary(0, 0, 0.0, float("nan"))
    cap = min(n_total, max(_ANOMALY_SAMPLE_FLOOR, int(n_total * _ANOMALY_SAMPLE_FRACTION)))
    learner = (
        random.Random(RANDOM_SEED).sample(learner_all, cap)
        if cap < n_total else learner_all
    )
    order = sorted(range(len(learner)), key=lambda i: learner[i].n_tokens)
    below = 0
    logp_sum = 0.0
    reference.eval()
    with torch.no_grad():
        for s in range(0, len(order), config.batch_size):
            idxs = order[s:s + config.batch_size]
            chunk = [learner[i] for i in idxs]
            batch = _collate(chunk, table, mean, std, device)
            ref_logits, _ = _forward_logits_critic(reference, batch)
            logp = masked_log_softmax(ref_logits / temp, batch.pack_mask)
            idx = batch.action_token.clamp(min=0).unsqueeze(1)
            logp_action = logp.gather(1, idx).squeeze(1).cpu().numpy()
            logp_sum += float(logp_action.sum())
            below += int((np.exp(logp_action) < _PROB_FLOOR).sum())
    n = len(learner)
    return _AnomalySummary(n, n_total, below / n, logp_sum / n)
