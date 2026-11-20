"""The ``train-picker`` gradient step: sample decks from the picker, score
them with the frozen scorer, and step the picker on the loss.
"""

from __future__ import annotations

import torch

from price_predictor.infrastructure.torch_training import (
    clip_per_group,
    kl_divergence,
    policy_entropy,
)
from sealed.application.picker_losses import LossInputs, LossSpec, _compute_losses
from sealed.application.picker_pools import PreparedPool, _collate
from sealed.application.picker_sampling import (
    SampledDecks,
    SamplingSpec,
    _sample_decks,
    _sampled_log_prob,
)
from sealed.application.picker_scoring import _score_sampled
from sealed.application.picker_training_run import TrainingRun


def _loss_spec(run: TrainingRun, entropy_coef: float) -> LossSpec:
    config = run.config
    return LossSpec(
        entropy_coef, config.aux_weight, config.normalize_advantage,
        config.objective, config.topk,
    )


def _add_kl(
    run: TrainingRun, total: torch.Tensor, logits: torch.Tensor,
    pool: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, float]:
    """The total plus the KL penalty to the reference picker, and the KL, when
    the run has a reference; else the total unchanged and 0.0."""
    reference = run.models.reference
    if reference is None:
        return total, 0.0
    pool_cards, pool_mask = pool
    with torch.no_grad():
        ref_logits, _ = reference(pool_cards, pool_mask)
    kl = kl_divergence(logits, ref_logits, pool_mask).mean()
    total = total + run.config.kl_coef * kl
    kl_value = float(kl.detach())
    return total, kl_value


def _train_one_step(
    run: TrainingRun, group: list[PreparedPool], entropy_coef: float,
) -> tuple[float, float, float, float]:
    """One gradient step over a batch of pools. Returns the detached
    (policy_loss, entropy_loss, aux_loss, kl) for logging."""
    config, device = run.config, run.device
    picker, scorer, optimizer = run.models.picker, run.models.scorer, run.models.optimizer
    cards, mask, land = _collate(group)
    pool_cards = torch.from_numpy(cards).to(device)
    pool_mask = torch.from_numpy(mask).to(device)
    is_land_mask = torch.from_numpy(land).to(device)

    logits, aux_pred = picker(pool_cards, pool_mask)

    pick_indices, picked_mask = _sample_decks(
        logits.detach(), is_land_mask, pool_mask,
        SamplingSpec(config.n_samples, config.temperature),
    )
    decks = SampledDecks(pick_indices, picked_mask, config.n_samples)

    norm_pool = scorer.normalize_features(pool_cards)
    rewards = _score_sampled(scorer, norm_pool, decks, device)  # (B, S)
    log_prob = _sampled_log_prob(logits, pool_mask, decks, config.temperature)

    entropy = policy_entropy(logits, pool_mask)
    losses = _compute_losses(
        LossInputs(rewards, log_prob, entropy, aux_pred), _loss_spec(run, entropy_coef),
    )
    total, kl_value = _add_kl(run, losses.total, logits, (pool_cards, pool_mask))

    optimizer.zero_grad()
    total.backward()
    clip_per_group(optimizer, max_norm=config.max_grad_norm)
    optimizer.step()

    return (
        float(losses.policy_loss.detach()),
        float(losses.entropy_loss.detach()),
        float(losses.aux_loss.detach()), kl_value,
    )
