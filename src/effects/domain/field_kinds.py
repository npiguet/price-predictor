"""What each field type does with a tensor: its loss, the best constant
prediction under that loss, and the magnitude its count term trains on.

One object per :class:`~effects.domain.field_types.FieldType`, shared by the
per-entity head's fields, the per-record heads' fields and the value head's
targets, so the losses, the constant-predictor floor and the evaluation's
deviance cannot read a type two ways. Every magnitude takes the cap
explicitly: the per-entity fields score at
:data:`~effects.domain.effect_head_input.MAGNITUDE_CAP`, where an amount above
it is the game compounding, while the value head's amounts are the text's own
and are scored uncapped (``cap=None``).

Every count and amount is scored by a negative binomial (FR-063): a mean, the
``exp`` of the field's log-rate column, and a dispersion ``θ`` per field, the
``exp`` of a learned log-dispersion. The variance is ``μ + μ²/θ``, so a field
whose amounts spread wider than a Poisson's stops paying for it in its mean,
and as ``θ → ∞`` the loss is the Poisson's.
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

#: The range a log-dispersion is read in: ``θ`` from about 0.0025, wider than
#: any count the corpus holds, to about 22,000, a Poisson for every mean the
#: cap allows. Bounded because a Poisson-like field drifts upward for good.
LOG_DISPERSION_MIN = -6.0
LOG_DISPERSION_MAX = 10.0
#: Where a learned log-dispersion starts: ``θ = 10``, within a few percent of
#: a Poisson's variance for the small means most fields have, so training
#: starts from the loss gen-2 trained on.
INITIAL_LOG_DISPERSION = 2.302585092994046

#: Newton steps of the constant predictor's dispersion fit, after the moment
#: estimate: the profile likelihood is smooth and concave in ``log θ`` near
#: its maximum, so a handful converge.
_NEWTON_STEPS = 25


def capped(target: torch.Tensor, cap: float | None) -> torch.Tensor:
    """``target`` with every magnitude above ``cap`` at the cap, sign kept;
    unchanged where ``cap`` is None."""
    if cap is None:
        return target
    return target.clamp(min=-cap, max=cap)


def counted(target: torch.Tensor, cap: float | None) -> torch.Tensor:
    """A count target clamped to ``[0, cap]``: a negative target (a heal
    netted into ``damage_taken``) has no likelihood under a count model.
    Clamped below at zero even where ``cap`` is None."""
    if cap is None:
        return target.clamp(min=0.0)
    return target.clamp(min=0.0, max=cap)


def delta_direction(target: torch.Tensor) -> torch.Tensor:
    """A signed delta's direction class, in :data:`DIRECTIONS` order:
    -1, 0 and 1 become 0, 1 and 2."""
    return torch.sign(target).long() + 1


def bounded_log_dispersion(log_dispersion: torch.Tensor) -> torch.Tensor:
    """A log-dispersion read in :data:`LOG_DISPERSION_MIN` …
    :data:`LOG_DISPERSION_MAX`."""
    return log_dispersion.clamp(LOG_DISPERSION_MIN, LOG_DISPERSION_MAX)


def negative_binomial(
    log_mean: torch.Tensor, target: torch.Tensor, log_dispersion: torch.Tensor,
) -> torch.Tensor:
    """The negative binomial NLL of each ``target``, elementwise, in the
    shape ``log_mean`` and ``target`` broadcast to with ``log_dispersion``.

    Exact, ``log y!`` included, so as ``θ → ∞`` it is the Poisson NLL with
    its full constant. Computed in float64: ``lgamma(y + θ) − lgamma(θ)``
    cancels to a few nats out of ``θ log θ``, which float32 cannot hold for
    a large ``θ``. No ``exp`` of the mean is taken, so a wild log-rate on an
    entity the loss masks away stays finite.
    """
    log_mean = log_mean.double()
    target = target.double()
    log_theta = bounded_log_dispersion(log_dispersion.double())
    theta = torch.exp(log_theta)
    terms = (
        torch.lgamma(target + 1.0) + torch.lgamma(theta) - torch.lgamma(target + theta)
        + (theta + target) * torch.logaddexp(log_theta, log_mean)
        - theta * log_theta - target * log_mean
    )
    # Float32 like every other loss term, whatever autocast handed in.
    return terms.float()


def negative_binomial_deviance(
    mean: torch.Tensor, target: torch.Tensor, log_dispersion: torch.Tensor,
) -> torch.Tensor:
    """Each ``target``'s unit deviance under a negative binomial of ``mean``
    and the given dispersion: twice the log-likelihood ratio of the saturated
    prediction, zero where ``mean`` is the target. Tends to the Poisson's as
    ``θ → ∞``."""
    mean = mean.double().clamp(min=1e-9)
    target = target.double()
    theta = torch.exp(bounded_log_dispersion(log_dispersion.double()))
    ratio = torch.where(
        target > 0, target * torch.log(target.clamp(min=1e-300) / mean),
        torch.zeros_like(target),
    )
    return 2.0 * (ratio - (target + theta) * torch.log((target + theta) / (mean + theta)))


def _mean(values: torch.Tensor) -> torch.Tensor:
    """The mean, or zero where nothing was selected."""
    if values.numel() == 0:
        return torch.zeros(())
    return values.float().mean()


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


def fit_log_dispersion(target: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """The maximum-likelihood log-dispersion of counts ``target``, each under
    its own constant ``mean`` (broadcast to it): a moment estimate, then
    Newton steps in ``log θ`` on the likelihood, bounded as the loss reads it.

    The means stay fixed: for any ``θ`` the likelihood's best constant mean
    is the sample mean, so the joint fit is the mean, then ``θ``. Counts no
    wider than a Poisson's fit at the upper bound.
    """
    y = target.double().reshape(-1)
    mu = torch.broadcast_to(mean.double(), target.shape).reshape(-1).clamp(min=_RATE_FLOOR)
    if y.numel() == 0:
        return torch.tensor(LOG_DISPERSION_MAX)
    excess = float(((y - mu) ** 2 - mu).sum())
    if excess <= 0.0:
        return torch.tensor(LOG_DISPERSION_MAX)
    log_theta = torch.tensor(float((mu ** 2).sum()) / excess, dtype=torch.float64).log()
    log_theta = log_theta.clamp(LOG_DISPERSION_MIN, LOG_DISPERSION_MAX)
    for _ in range(_NEWTON_STEPS):
        theta = log_theta.exp()
        score = (
            torch.digamma(y + theta) - torch.digamma(theta) + log_theta + 1.0
            - torch.log(theta + mu) - (theta + y) / (theta + mu)
        ).sum()
        curvature = (
            torch.polygamma(1, y + theta) - torch.polygamma(1, theta) + 1.0 / theta
            - 2.0 / (theta + mu) + (theta + y) / (theta + mu) ** 2
        ).sum()
        # In log θ: d/du = θ·score, d²/du² = θ²·curvature + θ·score.
        gradient = theta * score
        hessian = theta * theta * curvature + gradient
        if not torch.isfinite(hessian) or hessian >= 0:
            # Not concave here: step uphill by the gradient's sign instead.
            step = torch.sign(gradient) * 0.5
        else:
            step = (-gradient / hessian).clamp(-2.0, 2.0)
        log_theta = (log_theta + step).clamp(LOG_DISPERSION_MIN, LOG_DISPERSION_MAX)
        if abs(float(step)) < 1e-10:
            break
    return log_theta.float()


class FieldKind(ABC):
    """One field type's behaviour on tensors.

    ``prediction`` is the field's columns, ``(..., width)``; ``target`` its
    targets over the same leading dims, ``(..., arity)`` for a per-member type
    and ``(...)`` otherwise. ``log_dispersion`` broadcasts to the leading dims
    and is read only by a type with a magnitude, which requires it.
    """

    __slots__ = ()

    @abstractmethod
    def terms(
        self, prediction: torch.Tensor, target: torch.Tensor,
        *, arity: int, cap: float | None, log_dispersion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The field's loss at each position of the leading dims (FR-080)."""

    def loss(
        self, prediction: torch.Tensor, target: torch.Tensor,
        *, arity: int, cap: float | None, log_dispersion: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The field's loss over every row, summed (FR-080)."""
        return self.terms(
            prediction, target, arity=arity, cap=cap, log_dispersion=log_dispersion,
        ).sum()

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


def _required(log_dispersion: torch.Tensor | None, kind: FieldKind) -> torch.Tensor:
    if log_dispersion is None:
        raise ValueError(
            f"a {type(kind).__name__} field is scored by a negative binomial "
            "and needs its log-dispersion"
        )
    return log_dispersion


class _Binary(FieldKind):
    __slots__ = ()

    def terms(self, prediction, target, *, arity, cap, log_dispersion=None):
        return functional.binary_cross_entropy_with_logits(
            prediction[..., 0], target.float(), reduction="none",
        )

    def best_constant(self, target, *, arity, cap):
        return _logit(_mean(target))


class _MultiBinary(FieldKind):
    __slots__ = ()

    def terms(self, prediction, target, *, arity, cap, log_dispersion=None):
        return functional.binary_cross_entropy_with_logits(
            prediction, target.float(), reduction="none",
        ).sum(-1)

    def best_constant(self, target, *, arity, cap):
        return _logit(target.float().mean(dim=0))


class _Count(FieldKind):
    __slots__ = ()

    def terms(self, prediction, target, *, arity, cap, log_dispersion=None):
        return negative_binomial(
            prediction[..., 0], self.magnitude(target, cap=cap),
            _required(log_dispersion, self),
        )

    def best_constant(self, target, *, arity, cap):
        return _log_rate(self.magnitude(target, cap=cap))

    def magnitude(self, target, *, cap):
        return counted(target.float(), cap)


class _SignedDelta(FieldKind):
    """Direction (categorical) plus magnitude (count), summed, so a gain and
    a loss are different predictions rather than a hedge."""

    __slots__ = ()

    def terms(self, prediction, target, *, arity, cap, log_dispersion=None):
        target = capped(target.float(), cap)
        direction_logits = prediction[..., SIGNED_DIRECTIONS]
        direction = functional.cross_entropy(
            direction_logits.reshape(-1, len(DIRECTIONS)),
            delta_direction(target).reshape(-1), reduction="none",
        ).reshape(target.shape)
        return direction + negative_binomial(
            prediction[..., SIGNED_MAGNITUDE], target.abs(),
            _required(log_dispersion, self),
        )

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

    def terms(self, prediction, target, *, arity, cap, log_dispersion=None):
        return functional.cross_entropy(
            prediction.reshape(-1, arity), target.reshape(-1).long(), reduction="none",
        ).reshape(target.shape)

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
