"""What each field type does with a tensor: its loss, the best constant
prediction under that loss, and the magnitude its count term trains on.

One object per :class:`~effects.domain.field_types.FieldType`, shared by the
per-entity head's fields and the value head's targets, so the losses, the
constant-predictor floor and the evaluation's deviance cannot read a type two
ways. Every magnitude takes the cap explicitly: the per-entity fields score at
:data:`~effects.domain.effect_head_input.MAGNITUDE_CAP`, where an amount above
it is the game compounding, while the value head's amounts are the text's own
and are scored uncapped (``cap=None``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as functional

from effects.domain.field_types import (
    DIRECTIONS,
    SIGNED_DIRECTIONS,
    SIGNED_MAGNITUDE,
    FieldType,
)

#: Rates are clamped away from zero and one before a log, or an outcome the
#: sample never showed would make the floor nan.
_RATE_FLOOR = 1e-6


def capped(target: torch.Tensor, cap: float | None) -> torch.Tensor:
    """``target`` with every magnitude above ``cap`` at the cap, sign kept;
    unchanged where ``cap`` is None."""
    if cap is None:
        return target
    return target.clamp(min=-cap, max=cap)


def counted(target: torch.Tensor, cap: float | None) -> torch.Tensor:
    """A count target clamped to ``[0, cap]``: a negative target (a heal
    netted into ``damage_taken``) makes the Poisson NLL unbounded below.
    Unchanged where ``cap`` is None."""
    if cap is None:
        return target
    return target.clamp(min=0.0, max=cap)


def delta_direction(target: torch.Tensor) -> torch.Tensor:
    """A signed delta's direction class, in :data:`DIRECTIONS` order:
    -1, 0 and 1 become 0, 1 and 2."""
    return torch.sign(target).long() + 1


def poisson(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Poisson NLL on a log-rate prediction, summed. ``full=True`` changes no
    gradient but puts every term's minimum near zero, so the loss does not
    track batch composition."""
    return functional.poisson_nll_loss(
        prediction, target, log_input=True, full=True, reduction="sum",
    )


def _mean(values: torch.Tensor) -> torch.Tensor:
    """The mean, or zero where nothing was selected."""
    if values.numel() == 0:
        return torch.zeros(())
    return values.mean()


def _logit(rate: torch.Tensor) -> torch.Tensor:
    """The logit a BCE-with-logits loss reads back as ``rate``."""
    rate = rate.clamp(_RATE_FLOOR, 1.0 - _RATE_FLOOR)
    return torch.log(rate / (1.0 - rate))


def _log_frequencies(counts: torch.Tensor) -> torch.Tensor:
    """Log class frequencies, which cross-entropy reads back as frequencies."""
    return torch.log((counts / counts.sum()).clamp(min=_RATE_FLOOR))


def _log_rate(magnitudes: torch.Tensor) -> torch.Tensor:
    """The log rate the count loss reads back as the mean magnitude."""
    return torch.log(_mean(magnitudes).clamp(min=_RATE_FLOOR))


class FieldKind(ABC):
    """One field type's behaviour on tensors.

    ``prediction`` is the field's columns, ``(rows, width)``; ``target`` its
    targets over the same rows, ``(rows, arity)`` for a per-member type and
    ``(rows,)`` otherwise.
    """

    __slots__ = ()

    @abstractmethod
    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor,
        *, arity: int, cap: float | None,
    ) -> torch.Tensor:
        """The field's loss over every row, summed (FR-080)."""

    @abstractmethod
    def best_constant(
        self, target: torch.Tensor, *, arity: int, cap: float | None,
    ) -> torch.Tensor:
        """The maximum-likelihood constant prediction under :meth:`loss`: the
        field's columns, or a scalar for a one-column type."""

    def magnitude(self, target: torch.Tensor, *, cap: float | None) -> torch.Tensor:
        """The magnitude the count term trains on, so anything scoring that
        term reads the target the same way."""
        raise ValueError(f"a {type(self).__name__} field has no magnitude")


class _Binary(FieldKind):
    __slots__ = ()

    def loss(self, prediction, target, *, arity, cap):
        return functional.binary_cross_entropy_with_logits(
            prediction.squeeze(-1), target.float(), reduction="sum",
        )

    def best_constant(self, target, *, arity, cap):
        return _logit(_mean(target))


class _MultiBinary(FieldKind):
    __slots__ = ()

    def loss(self, prediction, target, *, arity, cap):
        return functional.binary_cross_entropy_with_logits(
            prediction, target.float(), reduction="sum",
        )

    def best_constant(self, target, *, arity, cap):
        return _logit(target.mean(dim=0))


class _Count(FieldKind):
    __slots__ = ()

    def loss(self, prediction, target, *, arity, cap):
        return poisson(prediction.squeeze(-1), self.magnitude(target, cap=cap))

    def best_constant(self, target, *, arity, cap):
        return _log_rate(self.magnitude(target, cap=cap))

    def magnitude(self, target, *, cap):
        return counted(target.float(), cap)


class _SignedDelta(FieldKind):
    """Direction (categorical) plus magnitude (count), summed, so a gain and
    a loss are different predictions rather than a hedge."""

    __slots__ = ()

    def loss(self, prediction, target, *, arity, cap):
        target = capped(target.float(), cap)
        direction_logits = prediction[..., SIGNED_DIRECTIONS]
        log_magnitude = prediction[..., SIGNED_MAGNITUDE]
        direction = delta_direction(target)
        loss = functional.cross_entropy(
            direction_logits.reshape(-1, len(DIRECTIONS)), direction.reshape(-1),
            reduction="sum",
        )
        return loss + poisson(log_magnitude, target.abs())

    def best_constant(self, target, *, arity, cap):
        direction = delta_direction(capped(target, cap)).reshape(-1)
        frequencies = _log_frequencies(
            torch.bincount(direction, minlength=len(DIRECTIONS)).float()
        )
        return torch.cat([frequencies, _log_rate(self.magnitude(target, cap=cap)).reshape(1)])

    def magnitude(self, target, *, cap):
        return capped(target.float(), cap).abs()


class _Categorical(FieldKind):
    __slots__ = ()

    def loss(self, prediction, target, *, arity, cap):
        return functional.cross_entropy(
            prediction.reshape(-1, arity), target.reshape(-1).long(), reduction="sum",
        )

    def best_constant(self, target, *, arity, cap):
        return _log_frequencies(
            torch.bincount(target.long().reshape(-1), minlength=arity).float()
        )


FIELD_KINDS: dict[FieldType, FieldKind] = {
    FieldType.BINARY: _Binary(),
    FieldType.MULTI_BINARY: _MultiBinary(),
    FieldType.COUNT: _Count(),
    FieldType.SIGNED_DELTA: _SignedDelta(),
    FieldType.CATEGORICAL: _Categorical(),
}
