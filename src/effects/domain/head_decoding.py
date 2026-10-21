"""Decoding the per-entity head's raw outputs into the quantities rules name.

No field of the head is a bare value. A count predicts a log rate; a signed
delta predicts three direction logits (negative, zero, positive) and then the
same log-rate magnitude; a categorical predicts logits over its classes. Every
reader that wants "how much damage" or "how likely to die" has to undo that
layout, and doing it in one place is what keeps gate 1's deviance and gate 2's
direction test reading the same column.
"""

from __future__ import annotations

import torch

from effects.domain.effect_head_layout import FIELD_SLICES, FieldSpec, FieldType

#: Field types that predict a magnitude under the count loss — "how much
#: damage", "how many cards", "how far the life total moved".
MAGNITUDE_TYPES: frozenset[FieldType] = frozenset(
    {FieldType.COUNT, FieldType.SIGNED_DELTA},
)


def field_outputs(outputs: torch.Tensor, spec: FieldSpec) -> torch.Tensor:
    """``spec``'s columns of a per-entity output, over any leading dims."""
    start, end = FIELD_SLICES[spec.name]
    return outputs[..., start:end]


def magnitude_log_rate(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """The predicted magnitude's log rate, from ``spec``'s columns.

    A count predicts only that column and a signed delta predicts direction
    then magnitude, so it is the last column either way.
    """
    if spec.type not in MAGNITUDE_TYPES:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no magnitude")
    return slice_[..., -1]


def expected_magnitude(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """The predicted magnitude itself: the Poisson rate, ``exp`` of its log."""
    return torch.exp(magnitude_log_rate(spec, slice_))


def direction_balance(spec: FieldSpec, slice_: torch.Tensor) -> torch.Tensor:
    """``P(positive) − P(negative)`` of a signed delta, in ``[-1, 1]``."""
    if spec.type is not FieldType.SIGNED_DELTA:
        raise ValueError(f"{spec.name} is a {spec.type} field and has no direction")
    probabilities = torch.softmax(slice_[..., :3], dim=-1)
    return probabilities[..., 2] - probabilities[..., 0]
