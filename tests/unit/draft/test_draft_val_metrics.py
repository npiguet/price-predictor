"""Validation metric helpers: imitation top-k + per-pack critic MSE (FR-037, SC-005)."""

from __future__ import annotations

import math

import numpy as np

from draft.application.train_draft_agent import (
    imitation_topk_accuracy,
    per_pack_critic_mse,
)


def test_imitation_top1_and_top3() -> None:
    # (logits over pack actions, target index)
    predictions = [
        (np.array([3.0, 1.0, 0.0]), 0),  # top-1 hit
        (np.array([1.0, 3.0, 0.0]), 0),  # miss top-1, hit top-3 (target ranked 2nd)
        (np.array([0.0, 1.0, 2.0, 5.0, 4.0]), 0),  # target ranked last -> miss top-3
    ]
    assert imitation_topk_accuracy(predictions, 1) == 1 / 3
    assert imitation_topk_accuracy(predictions, 3) == 2 / 3


def test_imitation_accuracy_empty_is_nan() -> None:
    assert math.isnan(imitation_topk_accuracy([], 1))


def test_per_pack_critic_mse_slices_by_pack() -> None:
    preds = [1.0, 2.0, 10.0, 12.0]
    targets = [1.0, 0.0, 10.0, 10.0]
    packs = [1, 1, 2, 2]
    out = per_pack_critic_mse(preds, targets, packs)
    # pack 1: errors 0, 2 -> mean(0, 4) = 2.0
    assert out[1] == 2.0
    # pack 2: errors 0, 2 -> mean(0, 4) = 2.0
    assert out[2] == 2.0


def test_per_pack_critic_mse_empty() -> None:
    assert per_pack_critic_mse([], [], []) == {}
