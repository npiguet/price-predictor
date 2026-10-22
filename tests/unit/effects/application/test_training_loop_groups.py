"""The encoder and the head are clipped apart (FR-095)."""

from __future__ import annotations

import torch

from effects.application.training_loop import parameter_groups


def test_groups_are_named_and_cover_every_parameter():
    encoder, head = torch.nn.Linear(2, 2), torch.nn.Linear(2, 3)
    groups = parameter_groups(encoder, head)
    assert [g["name"] for g in groups] == ["encoder", "head"]
    total_params = len(list(encoder.parameters())) + len(list(head.parameters()))
    assert sum(len(g["params"]) for g in groups) == total_params
    identity = torch.nn.Embedding(4, 2)
    assert [g["name"] for g in parameter_groups(encoder, head, identity)][-1] == "identity"


