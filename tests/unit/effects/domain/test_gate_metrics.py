"""The quantities the gates measure: gate 1's three margins and gate 3's geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from effects.domain.effect_head_layout import (
    DISPERSED_FIELDS,
    DISPERSION_INDEX,
    FIELD_SLICES,
    FIELDS_BY_NAME,
    GATE_INDEX,
    PER_ENTITY_WIDTH,
    ZONE_OUTCOMES,
)
from effects.domain.field_kinds import LOG_DISPERSION_MAX
from effects.domain.gate_metrics import (
    MarginAccumulator,
    balanced_accuracy,
    cosine_distance,
    f1_score,
    mean_deviance,
    mean_pairwise_cosine,
    top_component_share,
)
from tests.unit.effects.builders import fields_everywhere, make_entity_targets

#: A dispersion as good as a Poisson's.
POISSON = np.array(LOG_DISPERSION_MAX)


def poisson_deviance(predicted, observed):
    """The Poisson deviance, by its own formula."""
    predicted = np.asarray(predicted, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(observed > 0, observed * np.log(observed / predicted), 0.0)
    return float(2.0 * np.mean(ratio - (observed - predicted)))


def deviance(predicted, observed, log_dispersion=POISSON):
    return mean_deviance(np.asarray(predicted), np.asarray(observed), log_dispersion)


def model_dispersion(value: float = 0.0) -> torch.Tensor:
    return torch.full((len(DISPERSED_FIELDS),), value)


class TestGateOneMetrics:
    def test_f1_is_one_for_a_perfect_gate(self):
        observed = np.array([1, 0, 1, 1])
        assert f1_score(observed, observed) == pytest.approx(1.0)

    def test_f1_is_zero_when_nothing_true_is_caught(self):
        assert f1_score(np.array([0, 0]), np.array([1, 1])) == 0.0

    def test_f1_of_nothing_is_nan_rather_than_zero(self):
        """An empty stratum is unmeasured, which gate 1 fails as such."""
        assert math.isnan(f1_score(np.array([]), np.array([])))

    @pytest.mark.parametrize("log_dispersion", [-2.0, 0.0, 3.0])
    def test_deviance_is_zero_for_an_exact_prediction(self, log_dispersion):
        observed = np.array([1.0, 3.0, 5.0])
        assert deviance(observed, observed, np.array(log_dispersion)) == pytest.approx(
            0.0, abs=1e-9,
        )

    def test_deviance_grows_with_error(self):
        observed = np.array([3.0, 3.0])
        near = deviance(np.array([3.1, 3.1]), observed, np.array(0.5))
        far = deviance(np.array([9.0, 9.0]), observed, np.array(0.5))
        assert far > near

    def test_deviance_handles_zero_observations(self):
        value = deviance(np.array([0.5, 0.5]), np.array([0.0, 0.0]), np.array(0.0))
        assert math.isfinite(value)

    def test_an_empty_slice_is_nan_rather_than_zero(self):
        assert math.isnan(deviance(np.array([]), np.array([])))

    def test_deviance_reduces_to_the_poisson_as_the_dispersion_grows(self):
        """FR-063: the negative binomial deviance tends to the Poisson's as
        the dispersion goes to infinity; at the upper bound within 0.1%."""
        predicted = np.array([0.5, 2.0, 3.0, 7.5])
        observed = np.array([0.0, 1.0, 5.0, 4.0])
        poisson = poisson_deviance(predicted, observed)
        gaps = [
            abs(deviance(predicted, observed, np.array(value)) - poisson)
            for value in (2.0, 5.0, LOG_DISPERSION_MAX)
        ]
        assert gaps == sorted(gaps, reverse=True)
        assert gaps[-1] < 1e-3 * poisson

    def test_a_wider_dispersion_forgives_a_large_miss(self):
        """A run that learned its counts spread wide pays less for one far
        from its mean: each run is read under its own dispersion."""
        predicted, observed = np.array([2.0]), np.array([12.0])
        assert deviance(predicted, observed, np.array(0.0)) < deviance(
            predicted, observed, POISSON,
        )

    def test_each_observation_reads_its_own_dispersion(self):
        predicted, observed = np.array([2.0, 2.0]), np.array([6.0, 6.0])
        mixed = deviance(predicted, observed, np.array([0.0, LOG_DISPERSION_MAX]))
        assert mixed == pytest.approx(
            (deviance(predicted[:1], observed[:1], np.array(0.0))
             + deviance(predicted[:1], observed[:1], POISSON)) / 2,
        )

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


class TestMarginAccumulator:
    """The three margins pool every chunk's entities, then reduce once."""

    def _chunk(self, affected_damage: float, observed_damage: float = 2.0):
        gathered = torch.zeros(1, 2, PER_ENTITY_WIDTH)
        gathered[0, 0, GATE_INDEX] = 3.0   # predicted affected
        gathered[0, 1, GATE_INDEX] = -3.0  # predicted unaffected
        start, _ = FIELD_SLICES["zone_outcome"]
        gathered[0, 0, start + ZONE_OUTCOMES.index("died")] = 5.0
        damage, _ = FIELD_SLICES["damage_taken"]
        gathered[0, 0, damage] = torch.log(torch.tensor(affected_damage))
        targets = fields_everywhere({
            "zone_outcome": torch.tensor(
                [[float(ZONE_OUTCOMES.index("died")), 0.0]],
            ),
            "damage_taken": torch.tensor([[observed_damage, 0.0]]),
        })
        return gathered, make_entity_targets(torch.tensor([[1.0, 0.0]]), targets)

    def test_a_perfect_chunk_scores_perfectly(self):
        margins = MarginAccumulator(
            (FIELDS_BY_NAME["zone_outcome"], FIELDS_BY_NAME["damage_taken"]),
            model_dispersion(),
        )
        margins.add(*self._chunk(2.0))
        metrics = margins.metrics()
        assert metrics.affected_gate_f1 == pytest.approx(1.0)
        assert metrics.zone_outcome_accuracy == pytest.approx(1.0)
        assert metrics.mean_deviance == pytest.approx(0.0, abs=1e-6)

    def test_the_deviance_reads_the_model_s_own_dispersion(self):
        def scored(log_dispersion: float) -> float:
            dispersion = model_dispersion(LOG_DISPERSION_MAX)
            dispersion[DISPERSION_INDEX["damage_taken"]] = log_dispersion
            margins = MarginAccumulator((FIELDS_BY_NAME["damage_taken"],), dispersion)
            margins.add(*self._chunk(1.0, observed_damage=8.0))
            return margins.metrics().mean_deviance

        assert scored(0.0) < scored(LOG_DISPERSION_MAX)
        assert scored(LOG_DISPERSION_MAX) == pytest.approx(
            poisson_deviance(np.array([1.0]), np.array([8.0])), rel=1e-3,
        )

    def test_a_count_field_without_the_model_s_dispersion_is_refused(self):
        with pytest.raises(ValueError, match="log-dispersions"):
            MarginAccumulator((FIELDS_BY_NAME["damage_taken"],))

    def test_a_count_netted_negative_scores_as_none_as_it_trains(self):
        """A heal netted into ``damage_taken`` trains as zero damage, not as
        the heal's size."""
        margins = MarginAccumulator((FIELDS_BY_NAME["damage_taken"],), model_dispersion())
        margins.add(*self._chunk(1e-6, observed_damage=-2.0))
        assert margins.metrics().mean_deviance == pytest.approx(0.0, abs=1e-4)

    def test_nothing_added_is_unmeasured_rather_than_perfect(self):
        metrics = MarginAccumulator(()).metrics()
        assert math.isnan(metrics.affected_gate_f1)
        assert math.isnan(metrics.mean_deviance)
