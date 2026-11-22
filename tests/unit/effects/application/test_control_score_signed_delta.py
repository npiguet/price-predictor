"""The control score's record-field decoder at a signed delta's loss optimum."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from effects.application.control_score import decoder_of
from effects.domain.effect_head_layout import VERDICT_LAYOUT, VERDICT_WIDTH
from tests.unit.effects.domain.signed_delta_fit_support import OPTIMUM_TARGETS, fitted_columns


@pytest.mark.parametrize(("targets", "expected"), OPTIMUM_TARGETS)
def test_the_verdict_cost_adjustment_decodes_to_the_mean_of_the_targets(targets, expected):
    """``cost_adjustment`` is the record heads' signed delta, one per mana
    symbol; the decode sums the members, so each member reads the mean."""
    spec = VERDICT_LAYOUT.by_name["cost_adjustment"]
    start, end = VERDICT_LAYOUT.slices["cost_adjustment"]
    row = torch.zeros(VERDICT_WIDTH)
    row[start:end] = fitted_columns([float(t) for t in targets]).repeat(spec.count)
    heads = SimpleNamespace(verdict=row.unsqueeze(0))
    decoded = decoder_of("cost_adjustment").candidates(heads, torch.tensor([0]), None)
    assert float(decoded[0, 0]) / spec.count == pytest.approx(expected, abs=1e-3)
