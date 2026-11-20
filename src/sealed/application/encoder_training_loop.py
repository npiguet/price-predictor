"""The ``train-encoder`` training loop: the optimizer and its warmup, one
batch's forward pass and losses, one training epoch with its trunk
gradient-norm probe, and the best-checkpoint tracker.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sealed.application.encoder_losses import _per_batch_weighted_mse, _stack_head_preds
from sealed.application.encoder_mlm import MlmMaskSpec, _draw_mlm_mask
from sealed.domain.encoder_model import SealedEncoderModel

# ── Best-checkpoint helper ──────────────────────────────────────────────


@dataclass
class _BestCheckpoint:
    """Track best-by-val-loss state through training. Mirrors the shape of
    ``price_predictor.application.train_transformer._BestCheckpoint``.

    Follow-up: when a third training entry point arrives, extract this to
    a shared utility (research.md "Third-instance check").
    """

    best_val_loss: float = float("inf")
    best_epoch: int = -1
    best_state_dict: dict | None = None

    def update(self, model: SealedEncoderModel, epoch: int, val_loss: float) -> bool:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return True
        return False


# ── Training loop ──────────────────────────────────────────────────────


def _make_optimizer(
    model: SealedEncoderModel, lr: float, total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + linear warmup over the first 5% of total scheduled steps.

    After the warmup window the learning rate stays at ``lr`` for the
    remainder of the run (FR-022 / Decision D-14).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup_steps = max(1, int(math.ceil(total_steps * 0.05)))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    return optimizer, scheduler


def _select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class _TrainContext:
    """Read-only context passed through the training and evaluation loops:
    the MLM loss weight, what the MLM mask draw masks, and the device the
    batches run on.
    """

    mlm_weight: float
    mlm_mask: MlmMaskSpec
    device: torch.device


@dataclass(frozen=True)
class _Batch:
    """One collated batch (see ``_pad_collate``), on the training device."""

    ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    head_mask: torch.Tensor

    @classmethod
    def on_device(
        cls, collated: tuple[torch.Tensor, ...], device: torch.device,
    ) -> _Batch:
        """Move the loader's ``(ids, mask, labels, weights, head_mask)`` to
        ``device``, in that order.
        """
        ids, mask, labels, weights, head_mask = collated
        return cls(
            ids.to(device), mask.to(device), labels.to(device),
            weights.to(device), head_mask.to(device),
        )


@dataclass(frozen=True)
class _Stepper:
    """The optimizer and its warmup schedule, stepped once per batch."""

    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LambdaLR


@dataclass
class _BatchResult:
    """The loss components plus the small tensors the eval loop needs.

    Deliberately does *not* hold the full prediction dict — only the
    stacked ``(B, 9)`` head predictions (for the val correlation) and,
    when any position was masked, the ``(N, vocab)`` MLM logits at those
    positions and their targets (for the val top-1 accuracy). The big
    ``(B, T, vocab)`` tensor of the old "logits everywhere" path never
    exists.
    """

    l_reg: torch.Tensor
    l_mlm: torch.Tensor
    l_total: torch.Tensor
    head_preds: torch.Tensor                   # (B, 9)
    mlm_logits_at_mask: torch.Tensor | None    # (N, vocab) — None if nothing masked
    mask_targets: torch.Tensor                 # (N,) — original ids at masked positions


def _forward_batch(
    model: SealedEncoderModel, batch: _Batch, ctx: _TrainContext,
) -> _BatchResult:
    """Run one batch end-to-end.

    The MLM head is applied only at the masked positions — it's
    position-wise, so this is exactly the ``Linear`` the old
    "project every position, then index ~15 %" path also wanted, minus
    the ~84 % wasted multiply and the ~hundreds-of-MB ``(B, T, vocab)``
    intermediate. The ``contextualized[mask_positions]`` gather runs
    on-device (``mask_positions`` is produced on-device by
    :func:`_draw_mlm_mask`), so there is no host sync.
    """
    ids = batch.ids
    masked_ids, mask_positions = _draw_mlm_mask(
        ids, batch.attention_mask, ctx.mlm_mask,
    )
    predictions = model(masked_ids, batch.attention_mask)
    head_preds = _stack_head_preds(predictions)                 # (B, 9)
    l_reg = _per_batch_weighted_mse(
        predictions, batch.labels, batch.weights, batch.head_mask,
    )
    contextualized = predictions["contextualized"]              # (B, T, d_model)
    if mask_positions.any():
        mask_targets = ids[mask_positions]                      # (N,)
        mlm_logits_at_mask = model.mlm_head(contextualized[mask_positions])  # (N, vocab)
        l_mlm = F.cross_entropy(mlm_logits_at_mask, mask_targets)
    else:
        mask_targets = ids.new_empty((0,))
        mlm_logits_at_mask = None
        l_mlm = contextualized.new_zeros(())
    l_total = l_reg + ctx.mlm_weight * l_mlm
    return _BatchResult(
        l_reg, l_mlm, l_total, head_preds, mlm_logits_at_mask, mask_targets,
    )


def _global_grad_norm(grads: Iterable[torch.Tensor | None]) -> float:
    """L2 norm of a list of gradient tensors as if flattened+concatenated.

    ``None`` entries (params unused by the loss being differentiated —
    e.g. ``l_mlm`` doesn't touch the attention-pool weights) are skipped.
    """
    squares = [g.detach().pow(2).sum() for g in grads if g is not None]
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()))


@dataclass
class _EpochTrainMetrics:
    """Per-epoch training averages plus a one-batch gradient-norm probe.

    ``trunk_grad_reg`` / ``trunk_grad_mlm_weighted`` are ‖∂L/∂θ‖ over the
    *shared encoder* params (token + card encoder), measured on the
    epoch's first batch — ``L_reg`` vs ``mlm_weight · L_mlm`` respectively.
    They answer "how hard does each objective pull the trunk?" (loss
    *magnitude* doesn't tell you that — a term can be large but flat).
    ``None`` when the first batch had no maskable position (so ``L_mlm``
    is a detached zero) — rare in practice.
    """

    reg: float
    mlm: float
    total: float
    trunk_grad_reg: float | None
    trunk_grad_mlm_weighted: float | None


def _probes_trunk(batch_idx: int, result: _BatchResult) -> bool:
    """Whether this batch runs the trunk gradient-norm probe: the epoch's
    first batch, when its regression loss has a graph and some position
    was masked.
    """
    return (
        batch_idx == 0
        and result.l_reg.requires_grad
        and result.mlm_logits_at_mask is not None
    )


def _trunk_grad_norms(
    result: _BatchResult, trunk_params: list[torch.Tensor], mlm_weight: float,
) -> tuple[float, float]:
    """‖∂L_reg/∂trunk‖ and ‖∂(w·L_mlm)/∂trunk‖, keeping the graph for the
    optimization ``backward()``.
    """
    grad_reg = _global_grad_norm(torch.autograd.grad(
        result.l_reg, trunk_params,
        retain_graph=True, allow_unused=True,
    ))
    grad_mlm_weighted = _global_grad_norm(torch.autograd.grad(
        mlm_weight * result.l_mlm, trunk_params,
        retain_graph=True, allow_unused=True,
    ))
    return grad_reg, grad_mlm_weighted


def _train_epoch(
    model: SealedEncoderModel,
    loader: DataLoader,
    stepper: _Stepper,
    ctx: _TrainContext,
) -> _EpochTrainMetrics:
    """Train for one epoch.

    On the epoch's first batch, before the optimization ``backward()``,
    runs two extra cheap backward passes (``torch.autograd.grad``, which
    does not touch ``.grad``) to measure ‖∂L_reg/∂trunk‖ and
    ‖∂(w·L_mlm)/∂trunk‖ — a per-epoch debug probe of the regression-vs-MLM
    pull on the shared encoder. ~2 extra backwards per epoch; negligible.
    """
    model.train()
    sum_reg = 0.0
    sum_mlm = 0.0
    sum_total = 0.0
    n_batches = 0
    trunk_params = (
        list(model.token_encoder.parameters())
        + list(model.card_encoder.parameters())
    )
    grad_reg: float | None = None
    grad_mlm_weighted: float | None = None
    for batch_idx, collated in enumerate(loader):
        batch = _Batch.on_device(collated, ctx.device)
        stepper.optimizer.zero_grad()
        result = _forward_batch(model, batch, ctx)
        if _probes_trunk(batch_idx, result):
            grad_reg, grad_mlm_weighted = _trunk_grad_norms(
                result, trunk_params, ctx.mlm_weight,
            )
        result.l_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        stepper.optimizer.step()
        stepper.scheduler.step()
        sum_reg += float(result.l_reg.item())
        sum_mlm += float(result.l_mlm.item())
        sum_total += float(result.l_total.item())
        n_batches += 1
    n = max(n_batches, 1)
    return _EpochTrainMetrics(
        reg=sum_reg / n,
        mlm=sum_mlm / n,
        total=sum_total / n,
        trunk_grad_reg=grad_reg,
        trunk_grad_mlm_weighted=grad_mlm_weighted,
    )


def _format_trunk_grads(metrics: _EpochTrainMetrics, mlm_weight: float) -> str:
    """Render the one-batch trunk gradient-norm probe as one log line."""
    g_reg = metrics.trunk_grad_reg
    g_mlm_w = metrics.trunk_grad_mlm_weighted
    if g_reg is None or g_mlm_w is None:
        return "trunk grad norm @1st batch: n/a (no maskable position)"
    raw_mlm = g_mlm_w / mlm_weight if mlm_weight > 0 else float("nan")
    ratio = g_mlm_w / g_reg if g_reg > 0 else float("inf")
    return (
        f"trunk grad norm @1st batch: reg={g_reg:.4f}  "
        f"mlm*w={g_mlm_w:.4f} (raw {raw_mlm:.4f})  mlm*w / reg = {ratio:.1f}x"
    )
