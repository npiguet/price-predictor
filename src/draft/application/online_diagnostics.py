"""The post-update no-grad sweep: exploration and movement figures of a round
(data-model §6, research D9).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from draft.application.online_examples import OnlineExample, _collate, _forward
from draft.domain.draft_agent_model import DraftAgentModel
from price_predictor.infrastructure.torch_training import (
    kl_divergence,
    masked_log_softmax,
    policy_entropy,
)

# --------------------------------------------------------------------------- #
# Diagnostics (data-model §6, research D9)
# --------------------------------------------------------------------------- #

@dataclass
class SweepDiagnostics:
    """Exploration + movement figures from the one post-update no-grad pass."""

    entropy: float = 0.0
    perplexity: float = 1.0
    off_argmax_rate: float = 0.0
    mean_logp: float = 0.0
    kl_prev_new: float = 0.0
    kl_init_new: float = 0.0


def diagnostics_sweep(
    prev_model: DraftAgentModel,
    model: DraftAgentModel,
    examples: list[OnlineExample],
    table: np.ndarray,
    *,
    batch_size: int,
    temperature: float,
    device: torch.device,
    init_model: DraftAgentModel | None = None,
) -> SweepDiagnostics:
    """One batched ``no_grad`` sweep over the round's picks forwarding each model.

    ``prev_model`` holds πₖ — the exact policy that generated the round — so the
    entropy, perplexity and off-argmax rate are exact rather than blurred across
    a pass whose weights were moving. The same sweep yields ``KL(πₖ ‖ πₖ₊₁)``,
    which no pre-update measurement could.

    ``init_model`` holds π₀, the run's warm start, and adds ``KL(π₀ ‖ πₖ₊₁)`` —
    how far the policy has travelled in total. It answers a different question
    from the per-round KL and neither implies the other: small steps taken in a
    consistent direction accumulate, so a per-round KL that rounds to zero does
    **not** establish that the policy is stationary (research D17). Omitted, that
    figure is reported as 0.

    **Every model must be in ``eval()``**: with dropout active these figures
    measure noise rather than the policy. The caller owns mode, since it also
    owns the update that precedes this call.
    """
    n = len(examples)
    if n == 0:
        return SweepDiagnostics()

    entropy_sum = 0.0
    logp_sum = 0.0
    kl_sum = 0.0
    kl_init_sum = 0.0
    off_argmax = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            chunk = examples[start:start + batch_size]
            batch = _collate(chunk, table, device)
            prev_logits, _ = _forward(prev_model, batch)
            new_logits, _ = _forward(model, batch)
            prev_scaled = prev_logits / temperature
            new_scaled = new_logits / temperature

            entropy_sum += float(policy_entropy(prev_scaled, batch.pack_mask).sum())
            logp = masked_log_softmax(prev_scaled, batch.pack_mask)
            actions = batch.action_token.unsqueeze(1)
            logp_sum += float(logp.gather(1, actions).squeeze(1).sum())
            argmax = prev_scaled.masked_fill(
                ~batch.pack_mask, float("-inf"),
            ).argmax(dim=-1)
            off_argmax += int((argmax != batch.action_token).sum())
            kl_sum += float(
                kl_divergence(prev_scaled, new_scaled, batch.pack_mask).sum()
            )
            if init_model is not None:
                init_logits, _ = _forward(init_model, batch)
                kl_init_sum += float(
                    kl_divergence(
                        init_logits / temperature, new_scaled, batch.pack_mask,
                    ).sum()
                )

    entropy = entropy_sum / n
    return SweepDiagnostics(
        entropy=entropy,
        perplexity=math.exp(entropy),
        off_argmax_rate=off_argmax / n,
        mean_logp=logp_sum / n,
        kl_prev_new=kl_sum / n,
        kl_init_new=kl_init_sum / n,
    )
