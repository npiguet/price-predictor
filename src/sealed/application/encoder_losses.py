"""The ``train-encoder`` regression loss: the per-batch, per-head
sum-to-1 weighted MSE over the nine heads (FR-017).
"""

from __future__ import annotations

import torch

from sealed.domain.encoder_model import COLOR_ORDER

# Per FR-017, the four signed heads sum unweighted into L_reg; the five
# color_lift heads sum with a 1/5 prefactor. _HEAD_LOSS_WEIGHT[h] gives
# the prefactor used when combining head h's weighted-MSE term into L_reg.
_HEAD_LOSS_WEIGHT: tuple[float, ...] = (
    1.0, 1.0, 1.0, 1.0,           # score_play / score_draw / played_rate / cast_lift
    0.2, 0.2, 0.2, 0.2, 0.2,      # color_lift_W..G
)


# ── Per-batch per-head sum-to-1 weighted MSE (FR-017, Decision D-11) ───


def _stack_head_preds(predictions: dict[str, torch.Tensor]) -> torch.Tensor:
    """Stack the 9 regression-head predictions into a ``(B, 9)`` tensor in
    the canonical head order (``score_play``, ``score_draw``,
    ``played_rate``, ``cast_lift``, then ``color_lift_{W,U,B,R,G}``) so it
    aligns with the ``labels`` / ``weights`` / ``head_mask`` tensors.
    """
    cols: list[torch.Tensor] = [
        predictions["score_play"],
        predictions["score_draw"],
        predictions["played_rate"],
        predictions["cast_lift"],
    ]
    color_lift = predictions["color_lift"]  # (B, 5)
    for k in range(len(COLOR_ORDER)):
        cols.append(color_lift[:, k])
    return torch.stack(cols, dim=1)  # (B, 9)


def _per_batch_weighted_mse(
    predictions: dict[str, torch.Tensor],
    labels: torch.Tensor,
    weights: torch.Tensor,
    head_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute ``L_reg`` for one batch.

    Each of the 9 heads gets a per-batch sum-to-1 weighted MSE term;
    the four signed-head terms sum unweighted, and the five color-lift
    terms sum with the FR-017 ``(1/5)`` prefactor. A head with zero
    total sample weight in the batch contributes exactly zero — its
    numerator is also exactly zero (every contributing card's weight is
    zero), and ``torch.where`` selects the explicit ``0`` branch, so
    nothing leaks through the (clamped, never-actually-used) divisor.
    Fully vectorized: no per-head Python loop, no ``.item()`` syncs.
    """
    preds = _stack_head_preds(predictions)              # (B, 9)
    masked = weights * head_mask                        # (B, 9)
    weighted = ((preds - labels) ** 2 * masked).sum(dim=0)   # (9,)
    weight_total = masked.sum(dim=0)                    # (9,)
    per_head = torch.where(
        weight_total > 0,
        weighted / weight_total.clamp(min=1.0),
        torch.zeros_like(weighted),
    )                                                   # (9,)
    prefactor = preds.new_tensor(_HEAD_LOSS_WEIGHT)     # (9,)
    return (per_head * prefactor).sum()
