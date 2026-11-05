"""The negative binomial every count and amount is scored by (FR-063, T2)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as functional

from effects.domain.field_kinds import (
    FIELD_KINDS,
    INITIAL_LOG_DISPERSION,
    LOG_DISPERSION_MAX,
    LOG_DISPERSION_MIN,
    bounded_log_dispersion,
    fit_log_dispersion,
    negative_binomial,
    negative_binomial_deviance,
)
from effects.domain.field_types import FieldType


def poisson_nll(log_rate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """The exact Poisson NLL, ``log y!`` included."""
    return functional.poisson_nll_loss(
        log_rate, target, log_input=True, full=False, reduction="none",
    ) + torch.lgamma(target + 1.0)


def sampled(mean: float, log_dispersion: float, size: int, seed: int) -> torch.Tensor:
    """Negative binomial draws as a Poisson of a gamma-distributed rate."""
    generator = torch.Generator().manual_seed(seed)
    theta = math.exp(log_dispersion)
    rate = torch._standard_gamma(torch.full((size,), theta, dtype=torch.float64), generator)
    return torch.poisson(rate * mean / theta, generator)


class TestNegativeBinomial:
    TARGETS = torch.tensor([0.0, 1.0, 2.0, 5.0, 13.0, 40.0])
    LOG_RATES = torch.log(torch.tensor([0.3, 1.0, 4.0, 5.0, 9.0, 30.0]))

    def test_it_reduces_to_the_poisson_as_the_dispersion_grows(self):
        """The variance is mu + mu^2 / theta: as theta grows, the Poisson's."""
        poisson = poisson_nll(self.LOG_RATES, self.TARGETS)
        gaps = [
            float((negative_binomial(self.LOG_RATES, self.TARGETS, torch.tensor(value))
                   - poisson).abs().max())
            for value in (0.0, 3.0, 6.0, LOG_DISPERSION_MAX)
        ]
        assert gaps == sorted(gaps, reverse=True)
        assert gaps[-1] < 0.05

    def test_it_is_the_negative_log_probability(self):
        """Exact, so its probabilities over every count sum to one."""
        counts = torch.arange(0, 400, dtype=torch.float32)
        for log_dispersion in (-1.0, 0.0, 2.0):
            nll = negative_binomial(
                torch.full_like(counts, math.log(3.0)), counts, torch.tensor(log_dispersion),
            )
            assert float(torch.exp(-nll.double()).sum()) == pytest.approx(1.0, abs=1e-4)

    def test_its_mean_is_best_where_it_meets_the_target(self):
        target = torch.full((5,), 4.0)
        for log_dispersion in (-2.0, 0.0, 5.0):
            def loss(rate, log_dispersion=log_dispersion):
                return float(negative_binomial(
                    torch.full((5,), math.log(rate)), target, torch.tensor(log_dispersion),
                ).sum())

            assert loss(4.0) < loss(3.0)
            assert loss(4.0) < loss(5.5)

    def test_a_wide_dispersion_pays_less_for_a_far_count(self):
        """What the negative binomial is for: one rare large count no longer
        drags the mean, because its dispersion learns the spread."""
        far = torch.tensor([30.0])
        near = torch.log(torch.tensor([2.0]))
        assert negative_binomial(near, far, torch.tensor(0.0)) < negative_binomial(
            near, far, torch.tensor(LOG_DISPERSION_MAX),
        )

    def test_it_stays_finite_for_a_wild_log_rate(self):
        """A masked-away entity's prediction is arbitrary; its term must not
        poison the batch with an inf times a zero weight."""
        values = negative_binomial(
            torch.tensor([-80.0, 80.0]), torch.tensor([0.0, 3.0]), torch.tensor(1.0),
        )
        assert torch.isfinite(values).all()

    def test_its_gradient_reaches_the_dispersion(self):
        log_dispersion = torch.tensor(0.0, requires_grad=True)
        negative_binomial(
            torch.log(torch.full((50,), 3.0)), sampled(3.0, 0.0, 50, seed=1).float(),
            log_dispersion,
        ).sum().backward()
        assert log_dispersion.grad is not None and float(log_dispersion.grad) != 0.0

    def test_it_is_float32_under_any_input_dtype(self):
        loss = negative_binomial(
            torch.zeros(3, dtype=torch.bfloat16), torch.ones(3), torch.tensor(0.0),
        )
        assert loss.dtype == torch.float32

    def test_the_dispersion_is_read_in_its_bounds(self):
        assert float(bounded_log_dispersion(torch.tensor(99.0))) == LOG_DISPERSION_MAX
        assert float(bounded_log_dispersion(torch.tensor(-99.0))) == LOG_DISPERSION_MIN
        assert LOG_DISPERSION_MIN < INITIAL_LOG_DISPERSION < LOG_DISPERSION_MAX


class TestDeviance:
    def test_it_is_zero_where_the_mean_is_the_target(self):
        target = torch.tensor([0.0, 2.0, 7.0])
        assert torch.allclose(
            negative_binomial_deviance(target, target, torch.tensor(0.5)),
            torch.zeros(3, dtype=torch.float64), atol=1e-8,
        )

    def test_it_is_twice_the_nll_above_the_saturated_one(self):
        mean, target = torch.tensor([2.0, 5.0]), torch.tensor([4.0, 1.0])
        log_dispersion = torch.tensor(0.7)
        deviance = negative_binomial_deviance(mean, target, log_dispersion)
        excess = (
            negative_binomial(torch.log(mean), target, log_dispersion)
            - negative_binomial(torch.log(target), target, log_dispersion)
        )
        assert torch.allclose(deviance.float(), 2 * excess, atol=1e-4)


class TestDispersionFit:
    """The constant predictor fits its dispersion by moments, then Newton
    steps on the likelihood (research T2)."""

    @pytest.mark.parametrize("true", [-1.0, 0.0, 1.5])
    def test_it_recovers_the_dispersion_counts_were_drawn_with(self, true):
        counts = sampled(4.0, true, 20_000, seed=7)
        fitted = float(fit_log_dispersion(counts, counts.mean()))
        assert fitted == pytest.approx(true, abs=0.15)

    def test_it_is_the_likelihood_s_maximum(self):
        counts = sampled(2.5, 0.3, 4000, seed=3)
        mean = counts.mean()
        fitted = fit_log_dispersion(counts, mean)

        def nll(log_dispersion):
            return float(negative_binomial(
                torch.log(mean).expand_as(counts).float(), counts.float(),
                torch.tensor(log_dispersion),
            ).sum())

        best = nll(float(fitted))
        assert best <= nll(float(fitted) + 0.05)
        assert best <= nll(float(fitted) - 0.05)

    def test_counts_no_wider_than_a_poisson_s_fit_at_the_upper_bound(self):
        counts = torch.tensor([2.0, 2.0, 3.0, 2.0, 1.0, 2.0])
        assert float(fit_log_dispersion(counts, counts.mean())) == LOG_DISPERSION_MAX

    def test_each_count_may_have_its_own_mean(self):
        """Members sharing a field's one dispersion each keep their own mean."""
        low, high = sampled(1.0, 0.5, 8000, seed=4), sampled(9.0, 0.5, 8000, seed=5)
        counts = torch.stack([low, high], dim=1)
        fitted = float(fit_log_dispersion(counts, counts.mean(0)))
        assert fitted == pytest.approx(0.5, abs=0.15)

    def test_nothing_to_fit_reads_as_a_poisson(self):
        assert float(fit_log_dispersion(torch.zeros(0), torch.tensor(1.0))) == LOG_DISPERSION_MAX


class TestKinds:
    def test_a_count_term_is_the_negative_binomial_of_its_magnitude(self):
        kind = FIELD_KINDS[FieldType.COUNT]
        prediction = torch.tensor([[0.5], [1.0]])
        target = torch.tensor([-2.0, 3.0])  # a heal netted negative scores as zero
        log_dispersion = torch.tensor(0.4)
        expected = negative_binomial(
            prediction[:, 0], torch.tensor([0.0, 3.0]), log_dispersion,
        )
        assert torch.allclose(
            kind.terms(prediction, target, arity=1, cap=40.0, log_dispersion=log_dispersion),
            expected,
        )

    def test_a_signed_delta_term_adds_direction_to_magnitude(self):
        kind = FIELD_KINDS[FieldType.SIGNED_DELTA]
        prediction = torch.tensor([[0.0, 0.0, 2.0, 1.0]])
        log_dispersion = torch.tensor(1.0)
        direction = functional.cross_entropy(prediction[:, :3], torch.tensor([2]))
        magnitude = negative_binomial(prediction[:, 3], torch.tensor([3.0]), log_dispersion)
        assert torch.allclose(
            kind.terms(prediction, torch.tensor([3.0]), arity=1, cap=40.0,
                       log_dispersion=log_dispersion),
            direction + magnitude,
        )

    def test_terms_keep_every_leading_dim(self):
        """A group of same-kind fields is one call over (rows, fields)."""
        log_dispersion = torch.zeros(5)
        for field_type, width, target in (
            (FieldType.BINARY, 1, torch.ones(3, 5)),
            (FieldType.COUNT, 1, torch.ones(3, 5)),
            (FieldType.SIGNED_DELTA, 4, torch.ones(3, 5)),
            (FieldType.CATEGORICAL, 6, torch.ones(3, 5)),
            (FieldType.MULTI_BINARY, 6, torch.ones(3, 5, 6)),
        ):
            terms = FIELD_KINDS[field_type].terms(
                torch.zeros(3, 5, width), target, arity=6 if width == 6 else 1,
                cap=40.0, log_dispersion=log_dispersion,
            )
            assert terms.shape == (3, 5), field_type
