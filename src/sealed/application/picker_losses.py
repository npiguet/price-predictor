"""The ``train-picker`` loss: the policy term of either objective, the entropy
bonus and the auxiliary pool-quality term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

_ADVANTAGE_STD_EPS = 1e-6  # floor for the per-pool std in GRPO advantage norm.


@dataclass
class _Losses:
    policy_loss: torch.Tensor
    entropy_loss: torch.Tensor
    aux_loss: torch.Tensor
    total: torch.Tensor
    baseline: torch.Tensor   # (B,)
    advantage: torch.Tensor  # (B, S), detached


def _compute_losses(
    rewards: torch.Tensor,
    log_prob: torch.Tensor,
    entropy: torch.Tensor,
    aux_pred: torch.Tensor,
    entropy_coef: float,
    aux_weight: float,
    normalize_advantage: bool = False,
    objective: str = "reinforce",
    topk: int = 16,
) -> _Losses:
    """Assemble the picker training loss (spec § 3.3, § 3.4, FR-039).

    ``rewards`` and ``log_prob`` are ``(B, S)``; ``entropy`` and ``aux_pred``
    are ``(B,)``. The per-pool baseline is the mean reward; the aux target is
    the detached per-pool mean reward (FR-015). The entropy and aux terms are
    identical across objectives; only the policy-loss term differs.

    ``objective="reinforce"`` (default): advantage-weighted policy gradient
    with a detached per-pool baseline (FR-014). When ``normalize_advantage``
    is set, the centered reward is divided by the per-pool reward std
    (GRPO-style group normalization) — within-pool reward variance is small
    (the sampled decks share most cards), so raw advantages are tiny and the
    gradient weak; rescaling to unit variance gives a consistent step. A
    degenerate pool (all samples equal) keeps a ~0 advantage.

    ``objective="topk"`` (FR-039): reward-ranked / best-of-N. Per pool, keep
    the ``topk`` highest-reward sampled decks and maximize their log-prob by
    plain max-likelihood (no baseline/advantage term). Depends only on the
    ranking, not the (tiny) advantage magnitudes, and has no high-variance
    negative-gradient term.
    """
    baseline = rewards.mean(dim=1)
    centered = rewards - baseline.unsqueeze(1)
    if normalize_advantage:
        std = rewards.std(dim=1, unbiased=False, keepdim=True)
        centered = centered / (std + _ADVANTAGE_STD_EPS)
    advantage = centered.detach()
    if objective == "topk":
        k = min(topk, rewards.shape[1])
        topk_idx = rewards.topk(k, dim=1).indices  # (B, k)
        selected_log_prob = log_prob.gather(1, topk_idx)
        policy_loss = -selected_log_prob.mean()
    else:
        policy_loss = -(advantage * log_prob).mean()
    entropy_loss = -entropy_coef * entropy.mean()
    aux_loss = F.mse_loss(aux_pred, baseline.detach())
    total = policy_loss + entropy_loss + aux_weight * aux_loss
    return _Losses(policy_loss, entropy_loss, aux_loss, total, baseline, advantage)
