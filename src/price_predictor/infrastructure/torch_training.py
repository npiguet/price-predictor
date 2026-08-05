"""Shared torch training primitives: masked policy distributions + gradient clipping.

The shared home for the small torch helpers every policy-gradient trainer in the
repository needs, sitting beside :mod:`price_predictor.infrastructure.torch_checkpoint`.
Living in ``price_predictor`` is what makes them importable by both ``sealed``
(picker, scorer) and ``draft`` (agent, RL, online GRPO) without inverting the
``draft`` → ``sealed`` → ``price_predictor`` dependency direction.

The masked helpers all share one subtlety worth stating once: at masked positions
the log-probability is ``-inf`` and the probability is ``0``, so the ``p · logp``
product is ``0 · -inf = NaN``. ``torch.where`` would hide the NaN from the forward
value but autograd still backpropagates NaN through the discarded branch. Every
helper below therefore zeroes the ``-inf`` terms *before* the product, keeping both
the value and the gradient finite (the contribution is genuinely 0 there).
"""

from __future__ import annotations

import torch


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over the ``True`` positions of ``mask``; ``-inf`` elsewhere."""
    return torch.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


def policy_entropy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-row entropy of the masked softmax, shape ``(B,)``. NaN-gradient-guarded."""
    logp = masked_log_softmax(logits, mask)
    p = logp.exp()
    safe_logp = logp.masked_fill(~mask, 0.0)
    return -(p * safe_logp).sum(dim=-1)


def kl_divergence(
    logits: torch.Tensor, ref_logits: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    """Per-row ``KL(π ‖ π_ref)`` over the masked positions, shape ``(B,)``.

    Same ``0 · -inf`` trap as :func:`policy_entropy`: at masked positions the
    log-ratio is ``-inf - -inf = NaN``, so it is zeroed before weighting by ``p``
    (which is 0 there anyway). Callers wanting a scalar penalty take ``.mean()``.
    """
    logp = masked_log_softmax(logits, mask)
    logq = masked_log_softmax(ref_logits, mask)
    p = logp.exp()
    log_ratio = (logp - logq).masked_fill(~mask, 0.0)
    return (p * log_ratio).sum(dim=-1)


def clip_per_group(
    optimizer: torch.optim.Optimizer, *, max_norm: float,
) -> dict[str, float]:
    """Clip each parameter group at ``max_norm``; return the **pre-clip** L2 norms.

    Keyed by each group's ``"name"`` (``"group"`` when unnamed). The pre-clip
    values are the diagnostic signal — post-clip norms are bounded by ``max_norm``
    and carry no shape information.
    """
    norms: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = group.get("name", "group")
        pre_clip = torch.nn.utils.clip_grad_norm_(group["params"], max_norm=max_norm)
        norms[name] = float(pre_clip)
    return norms
