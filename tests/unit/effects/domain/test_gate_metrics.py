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
)
from effects.domain.field_kinds import LOG_DISPERSION_MAX
from effects.domain.gate_metrics import (
    GATE1_MIN_ZONE_CLASS_CELLS,
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
        from its mean, which is why gate 1 reads both models under one."""
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
        observed = np.array([0] * 90 + [1] * 10)
        always_stayed = np.zeros(100, dtype=int)
        assert balanced_accuracy(always_stayed, observed) == pytest.approx(0.5)

    def test_zone_accuracy_of_nothing_is_nan(self):
        assert math.isnan(balanced_accuracy(np.array([]), np.array([])))

    def test_a_class_of_three_cells_does_not_move_the_zone_accuracy(self):
        """Balanced over a 3-cell class, one of its cells would move the
        accuracy by a twelfth of a class's weight, near the gate's 0.05 margin."""
        observed = np.array([0] * 20 + [1] * 12 + [2] * 3)
        wrong = np.where(observed == 2, 0, observed)
        assert balanced_accuracy(wrong, observed) == balanced_accuracy(observed, observed)
        assert balanced_accuracy(wrong, observed) == pytest.approx(1.0)

    def test_a_class_of_ten_cells_counts(self):
        observed = np.array([0] * 20 + [1] * GATE1_MIN_ZONE_CLASS_CELLS)
        always_stayed = np.zeros(observed.size, dtype=int)
        assert balanced_accuracy(always_stayed, observed) == pytest.approx(0.5)

    def test_fewer_than_two_classes_of_ten_cells_is_unmeasured(self):
        """One class's accuracy is one a constant answer scores perfectly."""
        observed = np.array([0] * 20 + [1] * (GATE1_MIN_ZONE_CLASS_CELLS - 1))
        assert math.isnan(balanced_accuracy(observed, observed))


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



#: The checkpoints' default --deviance-floor: no field of these chunks is at
#: or below it unless it never moves.
FLOOR = 0.04


def _cells(field: str, means, observed):
    """``gathered`` and targets for one magnitude field, every entity affected:
    ``means`` the predicted magnitudes, ``observed`` the targets."""
    count = len(means)
    gathered = torch.zeros(1, count, PER_ENTITY_WIDTH)
    gathered[0, :, GATE_INDEX] = 3.0
    start, _ = FIELD_SLICES[field]
    column = start + FIELDS_BY_NAME[field].type.magnitude_column
    gathered[0, :, column] = torch.log(torch.tensor(means, dtype=torch.float32))
    targets = fields_everywhere({field: torch.tensor([observed], dtype=torch.float32)})
    return gathered, make_entity_targets(torch.ones(1, count), targets)


def _accumulator(fields, dispersion=None, *, deviance_floor=FLOOR, own=None):
    return MarginAccumulator(
        tuple(FIELDS_BY_NAME[name] for name in fields),
        model_dispersion() if dispersion is None else dispersion,
        deviance_floor=deviance_floor, own_log_dispersion=own,
    )


def _deviance(fields, *chunks, **kwargs):
    margins = _accumulator(fields, **kwargs)
    for chunk in chunks:
        margins.add(*chunk)
    return margins.metrics()


class TestMarginAccumulator:
    """The three margins pool every chunk's entities, then reduce once."""

    def test_a_perfect_chunk_scores_perfectly(self):
        metrics = _deviance(("damage_taken",), _cells("damage_taken", [2.0, 5.0], [2.0, 5.0]))
        assert metrics.affected_gate_f1 == pytest.approx(1.0)
        assert metrics.mean_deviance == pytest.approx(0.0, abs=1e-6)

    def test_the_deviance_reads_the_dispersion_it_is_given(self):
        def scored(log_dispersion: float) -> float:
            dispersion = model_dispersion(LOG_DISPERSION_MAX)
            dispersion[DISPERSION_INDEX["damage_taken"]] = log_dispersion
            chunk = _cells("damage_taken", [1.0, 1.0], [8.0, 0.0])
            return _deviance(("damage_taken",), chunk, dispersion=dispersion).mean_deviance

        assert scored(0.0) < scored(LOG_DISPERSION_MAX)
        assert scored(LOG_DISPERSION_MAX) == pytest.approx(
            poisson_deviance(np.array([1.0, 1.0]), np.array([8.0, 0.0])), rel=1e-3,
        )

    def test_two_models_differing_only_in_dispersion_score_the_same_deviance(self):
        """Each read under its own, a wider dispersion alone would win the
        margin; under the main model's the same predictions
        score the same, and each own-dispersion figure is kept beside it."""
        chunk = _cells("damage_taken", [0.5, 0.5, 0.5], [0.0, 3.0, 1.0])
        main = model_dispersion(4.6)
        narrow, wide = (_deviance(("damage_taken",), chunk, dispersion=main, own=own)
                        for own in (main, model_dispersion(-2.3)))
        assert narrow.mean_deviance == wide.mean_deviance
        assert narrow.own_dispersion_deviance == narrow.mean_deviance
        assert wide.own_dispersion_deviance < narrow.own_dispersion_deviance

    def test_each_field_weighs_the_same_whatever_its_cells(self):
        """Pooled per cell, the field with the most cells would set the margin."""
        many = _cells("damage_taken", [1.0] * 30, [3.0] * 15 + [0.0] * 15)
        few = _cells("power_delta", [1.0, 1.0], [4.0, 0.0])
        both = _deviance(("damage_taken", "power_delta"), many, few)
        each = [_deviance((field,), chunk).mean_deviance
                for field, chunk in (("damage_taken", many), ("power_delta", few))]
        assert both.mean_deviance == pytest.approx(np.mean(each))

    def test_a_field_at_or_under_the_deviance_floor_is_left_out(self):
        """A field that never moves in the stratum has a constant-prediction
        deviance of zero: its residual rate would weigh as much as a field
        that carries the signal."""
        moving = _cells("damage_taken", [1.0, 1.0], [3.0, 0.0])
        still = _cells("power_delta", [2.0] * 5, [0.0] * 5)
        assert _deviance(("damage_taken", "power_delta"), moving, still).mean_deviance == (
            _deviance(("damage_taken",), moving).mean_deviance)

    def test_no_field_over_the_floor_is_unmeasured_rather_than_zero(self):
        chunk = _cells("damage_taken", [1.0, 1.0], [3.0, 0.0])
        assert math.isnan(_deviance(("damage_taken",), chunk, deviance_floor=10.0).mean_deviance)

    def test_a_count_field_without_the_model_s_dispersion_is_refused(self):
        with pytest.raises(ValueError, match="log-dispersions"):
            MarginAccumulator((FIELDS_BY_NAME["damage_taken"],), deviance_floor=FLOOR)

    def test_a_count_netted_negative_scores_as_none_as_it_trains(self):
        """A heal netted into ``damage_taken`` trains as zero damage, not as
        the heal's size."""
        chunk = _cells("damage_taken", [1e-6, 2.0], [-2.0, 2.0])
        assert _deviance(("damage_taken",), chunk).mean_deviance == pytest.approx(0.0, abs=1e-4)

    def test_nothing_added_is_unmeasured_rather_than_perfect(self):
        metrics = MarginAccumulator((), deviance_floor=FLOOR).metrics()
        assert math.isnan(metrics.affected_gate_f1)
        assert math.isnan(metrics.mean_deviance)
