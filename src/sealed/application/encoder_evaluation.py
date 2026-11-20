"""The ``train-encoder`` validation pass: its losses, the MLM perplexity and
top-1 accuracy, and the per-head prediction-to-target correlation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from sealed.application.encoder_heads import _ALL_HEAD_NAMES, N_HEADS
from sealed.application.encoder_training_loop import _Batch, _forward_batch, _TrainContext
from sealed.domain.encoder_model import COLOR_ORDER, SealedEncoderModel

# ── Per-head prediction↔target correlation (val-set diagnostics) ───────


class _HeadCorrAccumulator:
    """Streaming Pearson correlation between predicted and target values,
    per regression head, over the cells a card actually contributes to.

    Pearson r is the most legible "is the encoder learning this head's
    signal?" metric: ~0 means it's predicting the mean (the aggregate
    MSE looks fine because the labels are shrunk toward neutral), ~0.3+
    means it's tracking the per-card signal.
    """

    def __init__(self) -> None:
        z = lambda: np.zeros(N_HEADS, dtype=np.float64)  # noqa: E731
        self._n = z()
        self._sx, self._sy = z(), z()
        self._sxx, self._syy, self._sxy = z(), z(), z()

    def add(
        self,
        preds: torch.Tensor,    # (B, 9)
        targets: torch.Tensor,  # (B, 9) — shrunk values, 0.0 at empty cells
        head_mask: torch.Tensor,  # (B, 9) — 1.0 where the cell is non-empty
    ) -> None:
        mask = head_mask.detach().cpu().numpy().astype(bool)
        p = preds.detach().cpu().numpy().astype(np.float64)
        t = targets.detach().cpu().numpy().astype(np.float64)
        for h in range(N_HEADS):
            sel = mask[:, h]
            if not sel.any():
                continue
            ph, th = p[sel, h], t[sel, h]
            self._n[h] += ph.size
            self._sx[h] += ph.sum()
            self._sy[h] += th.sum()
            self._sxx[h] += np.dot(ph, ph)
            self._syy[h] += np.dot(th, th)
            self._sxy[h] += np.dot(ph, th)

    def correlations(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for h, name in enumerate(_ALL_HEAD_NAMES):
            n = self._n[h]
            if n < 2:
                out[name] = None
                continue
            cov = n * self._sxy[h] - self._sx[h] * self._sy[h]
            vx = n * self._sxx[h] - self._sx[h] ** 2
            vy = n * self._syy[h] - self._sy[h] ** 2
            out[name] = (
                float(cov / math.sqrt(vx * vy)) if vx > 0.0 and vy > 0.0 else None
            )
        return out


@dataclass
class _EvalMetrics:
    """Per-epoch validation metrics, including human-readable diagnostics."""

    reg: float
    mlm: float          # cross-entropy in nats
    total: float        # reg + mlm_weight * mlm — the best-checkpoint metric
    mlm_perplexity: float          # exp(mlm)
    mlm_top1_accuracy: float       # fraction of masked tokens predicted correctly
    head_correlations: dict[str, float | None]  # Pearson r per head, None if undefined


def _format_head_corrs(corrs: dict[str, float | None]) -> str:
    """Render the per-head val correlations as one compact log line."""
    def fmt(value: float | None) -> str:
        return f"{value:+.2f}" if value is not None else "  --"

    signed = " ".join(
        f"{name}={fmt(corrs[name])}"
        for name in ("score_play", "score_draw", "played_rate", "cast_lift")
    )
    color = " ".join(f"{c}={fmt(corrs[f'color_lift_{c}'])}" for c in COLOR_ORDER)
    return f"val corr (pred vs target): {signed} | color_lift {color}"


@torch.no_grad()
def _eval_loss(
    model: SealedEncoderModel,
    loader: DataLoader,
    ctx: _TrainContext,
) -> _EvalMetrics:
    """Evaluate on the val set.

    Returns the loss components (``reg``/``mlm``/``total`` — the last is
    the best-checkpoint metric per FR-019) plus diagnostics: MLM
    perplexity and top-1 masked-token accuracy, and the per-head
    Pearson correlation between predictions and (shrunk) targets. The
    MLM mask is drawn at val time too (same ``--mlm-mask-prob``) so the
    numbers stay comparable across epochs (Decision D-12).
    """
    model.eval()
    sum_reg = sum_mlm = sum_total = 0.0
    n_batches = 0
    mlm_correct = 0
    mlm_total = 0
    corr = _HeadCorrAccumulator()
    for collated in loader:
        batch = _Batch.on_device(collated, ctx.device)
        result = _forward_batch(model, batch, ctx)
        sum_reg += float(result.l_reg.item())
        sum_mlm += float(result.l_mlm.item())
        sum_total += float(result.l_total.item())
        n_batches += 1
        # MLM top-1 accuracy at masked positions.
        if result.mlm_logits_at_mask is not None:
            mlm_correct += int(
                (result.mlm_logits_at_mask.argmax(dim=-1) == result.mask_targets).sum()
            )
            mlm_total += int(result.mask_targets.numel())
        # Per-head prediction↔target correlation over non-empty cells.
        corr.add(result.head_preds, batch.labels, batch.head_mask)
    n = max(n_batches, 1)
    mlm_avg = sum_mlm / n
    return _EvalMetrics(
        reg=sum_reg / n,
        mlm=mlm_avg,
        total=sum_total / n,
        mlm_perplexity=math.exp(mlm_avg),
        mlm_top1_accuracy=(mlm_correct / mlm_total) if mlm_total else 0.0,
        head_correlations=corr.correlations(),
    )
