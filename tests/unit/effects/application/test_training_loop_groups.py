"""The encoder and the head are clipped apart (FR-095)."""

from __future__ import annotations

import torch

from effects.application.training_loop import parameter_groups
from price_predictor.infrastructure.torch_training import clip_per_group


def test_groups_are_named_and_cover_every_parameter():
    encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 3)
    groups = parameter_groups(encoder, head)
    assert [g["name"] for g in groups] == ["encoder", "head"]
    total_params = len(list(encoder.parameters())) + len(list(head.parameters()))
    assert sum(len(g["params"]) for g in groups) == total_params
    identity = torch.nn.Embedding(4, 2)
    assert [g["name"] for g in parameter_groups(encoder, head, identity)][-1] == "identity"


def test_a_large_head_gradient_does_not_scale_the_encoder():
    encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 3)
    for p in encoder.parameters():
        p.grad = torch.full_like(p, 0.01)           # tiny encoder gradient
    for p in head.parameters():
        p.grad = torch.full_like(p, 100.0)          # huge head gradient
    optimizer = torch.optim.AdamW(parameter_groups(encoder, head), lr=1e-4)
    encoder_before = torch.cat([p.grad.flatten() for p in encoder.parameters()]).norm().item()
    norms = clip_per_group(optimizer, max_norm=1.0)
    encoder_after = torch.cat([p.grad.flatten() for p in encoder.parameters()]).norm().item()
    assert set(norms) == {"encoder", "head"}
    assert abs(encoder_after - encoder_before) < 1e-6      # under the norm: untouched
    assert torch.cat([p.grad.flatten() for p in head.parameters()]).norm().item() <= 1.0 + 1e-5
