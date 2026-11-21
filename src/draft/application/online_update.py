"""The gen-3 single-term GRPO loss and the one shuffled pass a round takes over
its own fresh batch (FR-010, FR-011).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from draft.application.draft_training_common import length_bucketed_batches
from draft.application.online_examples import _Batch, _collate, _forward
from draft.application.online_round_loader import RoundBatch
from draft.domain.draft_agent_model import DraftAgentModel
from price_predictor.infrastructure.torch_training import clip_per_group, masked_log_softmax

# --------------------------------------------------------------------------- #
# Loss (data-model §5, FR-010)
# --------------------------------------------------------------------------- #

def _compute_loss(
    model: DraftAgentModel, batch: _Batch, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``-mean(A · logπ_T(a|s))`` over the batch. Returns ``(loss, policy_logits)``.

    The single term: no critic/value regression, no GAE, no KL anchor, no
    entropy bonus. The critic head is forwarded (the model returns it) and
    discarded, so it receives no gradient and is carried through the run
    unchanged (FR-027).
    """
    policy_logits, _critic = _forward(model, batch)
    logp = masked_log_softmax(policy_logits / temperature, batch.pack_mask)
    taken = logp.gather(1, batch.action_token.unsqueeze(1)).squeeze(1)
    loss = -(batch.advantage * taken).mean()
    return loss, policy_logits


# --------------------------------------------------------------------------- #
# The one-pass update (FR-011, research D8)
# --------------------------------------------------------------------------- #

@dataclass
class UpdateResult:
    """What one round's pass over its own fresh batch produced."""

    policy_loss: float = 0.0
    grad_norm: float = 0.0
    steps: int = 0


def run_update_pass(
    model: DraftAgentModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    batch: RoundBatch,
    *,
    temperature: float,
    batch_size: int,
    max_grad_norm: float,
    device: torch.device,
    rng: random.Random,
) -> UpdateResult:
    """One shuffled, length-bucketed pass over the round's learner picks.

    Exactly one epoch — the batch is then discarded and never revisited
    (FR-011). The mild within-pass off-policyness (later minibatches see
    already-stepped weights) is the accepted trade-off for getting enough
    gradient steps out of a round; ``KL(prev‖new)`` in the movement diagnostics
    exists to size it.
    """
    result = UpdateResult()
    if not batch.examples:
        return result

    model.train()
    minibatches = length_bucketed_batches(batch.examples, batch_size, rng)
    loss_sum = 0.0
    norm_sum = 0.0
    for chunk in minibatches:
        collated = _collate(chunk, batch.table, device)
        loss, _ = _compute_loss(model, collated, temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Pre-clip norms are the movement signal; post-clip ones are bounded by
        # max_grad_norm and carry no shape information.
        norms = clip_per_group(optimizer, max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach())
        norm_sum += max(norms.values()) if norms else 0.0
        result.steps += 1

    if result.steps:
        result.policy_loss = loss_sum / result.steps
        result.grad_norm = norm_sum / result.steps
    return result
