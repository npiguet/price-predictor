"""The quantities the gates measure: gate 1's three margins and gate 3's geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from effects.domain.gate_metrics import (
    balanced_accuracy,
    cosine_distance,
    f1_score,
    mean_pairwise_cosine,
    poisson_deviance,
    top_component_share,
)


class TestGateOneMetrics:
    def test_f1_is_one_for_a_perfect_gate(self):
        observed = np.array([1, 0, 1, 1])
        assert f1_score(observed, observed) == pytest.approx(1.0)

    def test_f1_is_zero_when_nothing_true_is_caught(self):
        assert f1_score(np.array([0, 0]), np.array([1, 1])) == 0.0

    def test_f1_of_nothing_is_nan_rather_than_zero(self):
        """An empty stratum is unmeasured, which gate 1 fails as such."""
        assert math.isnan(f1_score(np.array([]), np.array([])))

    def test_deviance_is_zero_for_an_exact_prediction(self):
        observed = np.array([1.0, 3.0, 5.0])
        assert poisson_deviance(observed, observed) == pytest.approx(0.0, abs=1e-9)

    def test_deviance_grows_with_error(self):
        observed = np.array([3.0, 3.0])
        near = poisson_deviance(np.array([3.1, 3.1]), observed)
        far = poisson_deviance(np.array([9.0, 9.0]), observed)
        assert far > near

    def test_deviance_handles_zero_observations(self):
        value = poisson_deviance(np.array([0.5, 0.5]), np.array([0.0, 0.0]))
        assert math.isfinite(value)

    def test_an_empty_slice_is_nan_rather_than_zero(self):
        assert math.isnan(poisson_deviance(np.array([]), np.array([])))

    def test_zone_accuracy_weighs_every_observed_outcome_the_same(self):
        """Answering the dominant class every time scores one over the number
        of classes observed, not that class's share (FR-109)."""
        observed = np.array([0] * 9 + [1])
        always_stayed = np.zeros(10, dtype=int)
        assert balanced_accuracy(always_stayed, observed) == pytest.approx(0.5)

    def test_zone_accuracy_of_nothing_is_nan(self):
        assert math.isnan(balanced_accuracy(np.array([]), np.array([])))


class TestGeometryMetrics:
    def test_mean_cosine_is_one_for_identical_vectors(self):
        vectors = np.tile(np.arange(1.0, 9.0), (50, 1))
        assert mean_pairwise_cosine(vectors, pairs=100) == pytest.approx(1.0)

    def test_mean_cosine_is_near_zero_for_random_high_dimensional_vectors(self):
        rng = np.random.default_rng(1)
        vectors = rng.normal(size=(500, 128))
        assert abs(mean_pairwise_cosine(vectors, pairs=2000)) < 0.1

    def test_a_pair_never_compares_a_vector_with_itself(self):
        """Self-pairs would drag the mean toward 1 and mask a real collapse."""
        vectors = np.eye(3)
        assert mean_pairwise_cosine(vectors, pairs=500) == pytest.approx(0.0)

    def test_the_sample_is_reproducible_under_a_seed(self):
        rng = np.random.default_rng(2)
        vectors = rng.normal(size=(200, 32))
        assert mean_pairwise_cosine(vectors, pairs=500, seed=7) == (
            mean_pairwise_cosine(vectors, pairs=500, seed=7)
        )

    def test_a_single_vector_is_nan_rather_than_a_score(self):
        assert math.isnan(mean_pairwise_cosine(np.ones((1, 8))))

    def test_the_top_component_of_a_line_explains_everything(self):
        vectors = np.outer(np.linspace(-1, 1, 100), np.array([1.0, 0.0, 0.0]))
        assert top_component_share(vectors) == pytest.approx(1.0)

    def test_the_top_component_of_an_isotropic_cloud_is_small(self):
        rng = np.random.default_rng(3)
        assert top_component_share(rng.normal(size=(500, 32))) < 0.15


class TestCosineDistance:
    def test_cosine_distance_is_zero_for_a_vector_and_itself(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0)

    def test_cosine_distance_of_a_zero_vector_is_nan(self):
        assert math.isnan(cosine_distance(np.zeros(3), np.ones(3)))
