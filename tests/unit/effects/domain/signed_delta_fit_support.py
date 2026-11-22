"""Fitting one signed delta's four columns to targets at the loss's own optimum."""

from __future__ import annotations

import torch

from effects.domain.effect_head_layout import FIELDS_BY_NAME
from effects.domain.field_kinds import FIELD_KINDS, FieldScoring

#: Targets whose mean the decode must return, with that mean.
OPTIMUM_TARGETS = [([0, 0, 2, 2], 1.0), ([0, 0, 0, 3], 0.75), ([0, -1, -1, -1], -0.75)]


def fitted_columns(targets: list[float]) -> torch.Tensor:
    """The four signed-delta columns (three direction logits, then the
    magnitude's log rate) that minimise the field kind's loss on ``targets``,
    every row scored, zeros included."""
    kind = FIELD_KINDS[FIELDS_BY_NAME["life_delta"].type]
    target = torch.tensor(targets)
    scoring = FieldScoring(arity=1, cap=None, log_dispersion=torch.zeros(()))
    columns = torch.zeros(4, requires_grad=True)
    optimiser = torch.optim.LBFGS(
        [columns], lr=1.0, max_iter=500, tolerance_grad=1e-10,
        tolerance_change=1e-14, line_search_fn="strong_wolfe",
    )

    def closure():
        optimiser.zero_grad()
        loss = kind.loss(columns.expand(len(targets), 4), target, scoring)
        loss.backward()
        return loss

    optimiser.step(closure)
    return columns.detach()
