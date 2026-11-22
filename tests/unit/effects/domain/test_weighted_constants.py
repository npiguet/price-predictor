"""A weighted best constant fits a row of weight k as k repeated rows
(FR-059a): what a reader weighing validation rows by their class weight
fits, and the scale a class weight puts on a copy's rule-mask weights.

Tensors are built by hand: the property is arithmetic over any targets."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from effects.domain.field_kinds import FIELD_KINDS, fit_log_dispersion
from effects.domain.field_types import FieldType
from effects.domain.records import Curation
from effects.domain.rule_mask import FULL_WEIGHT, CopyWeights, weighed_by_class

#: Per row: its weight, an integer so the repeated rows exist.
_WEIGHTS = torch.tensor([1.0, 3.0, 2.0, 1.0, 2.0, 3.0])
_TARGETS = {
    FieldType.BINARY: (torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]), 1),
    FieldType.MULTI_BINARY: (torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],
                                           [1.0, 0.0], [0.0, 0.0]]), 2),
    FieldType.COUNT: (torch.tensor([0.0, 4.0, 1.0, 7.0, 2.0, 0.0]), 1),
    FieldType.SIGNED_DELTA: (torch.tensor([-2.0, 3.0, 0.0, 1.0, -1.0, 5.0]), 1),
    FieldType.CATEGORICAL: (torch.tensor([0.0, 2.0, 1.0, 2.0, 0.0, 1.0]), 3),
}


def _repeated(values: torch.Tensor) -> torch.Tensor:
    return values.repeat_interleave(_WEIGHTS.long(), dim=0)


@pytest.mark.parametrize("field_type", list(_TARGETS), ids=str)
def test_a_weighted_best_constant_is_the_repeated_rows(field_type) -> None:
    target, arity = _TARGETS[field_type]
    kind = FIELD_KINDS[field_type]
    weighted = kind.best_constant(target, arity=arity, cap=10.0, weights=_WEIGHTS)
    repeated = kind.best_constant(_repeated(target), arity=arity, cap=10.0)
    assert torch.allclose(weighted, repeated, atol=1e-6)
    # Unweighted it differs, or the test reads nothing.
    assert not torch.allclose(kind.best_constant(target, arity=arity, cap=10.0), repeated,
                              atol=1e-4)


def test_a_weighted_dispersion_fit_is_the_repeated_rows() -> None:
    counts = torch.tensor([0.0, 9.0, 1.0, 14.0, 2.0, 0.0])
    mean = (counts * _WEIGHTS).sum() / _WEIGHTS.sum()
    weighted = fit_log_dispersion(counts, mean, _WEIGHTS)
    repeated = fit_log_dispersion(_repeated(counts), mean)
    assert float(weighted) == pytest.approx(float(repeated), abs=1e-5)


def test_no_weights_fits_as_before() -> None:
    target, arity = _TARGETS[FieldType.COUNT]
    kind = FIELD_KINDS[FieldType.COUNT]
    ones = torch.ones_like(_WEIGHTS)
    assert torch.allclose(kind.best_constant(target, arity=arity, cap=10.0, weights=ones),
                          kind.best_constant(target, arity=arity, cap=10.0))


@dataclass(frozen=True)
class _Copy:
    curation: Curation | None
    aimed_at: tuple | None = None


def test_a_class_weight_scales_every_weight_of_a_copy() -> None:
    masked = CopyWeights(0.25, frozenset({"zone_outcome"}), frozenset({"E1"}))
    scaled = weighed_by_class(masked, _Copy(Curation(1, 2, class_weight=3.0)))
    assert scaled.weight("zone_outcome") == 3.0 and scaled.weight("pt_change") == 0.75
    assert scaled.gate_weight("E1") == 3.0 and scaled.gate_weight("E2") == 0.0
    assert not weighed_by_class(FULL_WEIGHT, _Copy(Curation(1, 1, class_weight=2.0))).unmasked


def test_a_copy_of_weight_one_or_no_curation_is_unchanged() -> None:
    assert weighed_by_class(FULL_WEIGHT, _Copy(Curation(1, 1))) is FULL_WEIGHT
    assert weighed_by_class(FULL_WEIGHT, _Copy(None)) is FULL_WEIGHT
