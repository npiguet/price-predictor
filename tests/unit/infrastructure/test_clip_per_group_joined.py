"""Groups that share a name are clipped as one norm."""

from __future__ import annotations

import torch

from price_predictor.infrastructure.torch_training import clip_per_group


def _param(value: float) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.zeros(4))
    param.grad = torch.full((4,), value)
    return param


def test_two_groups_of_one_name_are_clipped_as_one_norm():
    first, second, other = _param(3.0), _param(4.0), _param(0.1)
    optimizer = torch.optim.AdamW([
        {"name": "encoder", "params": [first]},
        {"name": "encoder", "params": [second]},
        {"name": "head", "params": [other]},
    ])
    norms = clip_per_group(optimizer, max_norm=1.0)
    joined = (4 * 9.0 + 4 * 16.0) ** 0.5
    assert abs(norms["encoder"] - joined) < 1e-4
    assert abs(norms["head"] - 0.2) < 1e-4
    after = torch.sqrt(first.grad.pow(2).sum() + second.grad.pow(2).sum())
    assert abs(float(after) - 1.0) < 1e-4
    assert abs(float(first.grad[0]) / float(second.grad[0]) - 0.75) < 1e-4


def test_unnamed_groups_stay_clipped_apart():
    first, second = _param(3.0), _param(3.0)
    optimizer = torch.optim.AdamW([{"params": [first]}, {"params": [second]}])
    clip_per_group(optimizer, max_norm=1.0)
    assert abs(float(torch.linalg.vector_norm(first.grad)) - 1.0) < 1e-4
    assert abs(float(torch.linalg.vector_norm(second.grad)) - 1.0) < 1e-4
