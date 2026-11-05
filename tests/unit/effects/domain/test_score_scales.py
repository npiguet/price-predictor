"""The scores, their common scale and their worst-case standard errors
(FR-068b; research E4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from effects.domain.score_scales import (
    auc,
    auc_standard_error,
    clipped,
    component_standard_error,
    grid_score,
    grid_standard_error,
    loss_ratio_standard_error,
    r_squared,
    rescaled_auc,
)


class TestAuc:
    def test_it_is_scikit_learns_with_ties_and_weights(self):
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(3)
        labels = rng.random(400) < 0.3
        scores = rng.integers(0, 12, 400).astype(float)  # many ties
        weight = rng.random(400) + 0.1
        assert auc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
        assert auc(labels, scores, weight) == pytest.approx(
            roc_auc_score(labels, scores, sample_weight=weight), abs=1e-12)

    def test_a_tie_counts_half(self):
        assert auc([True, False], [0.5, 0.5]) == 0.5
        assert auc([True, False, False], [1.0, 0.0, 1.0]) == 0.75

    def test_one_label_alone_is_nan(self):
        assert math.isnan(auc([True, True], [0.1, 0.2]))
        assert math.isnan(auc([False, False], [0.1, 0.2]))

    def test_non_finite_scores_are_dropped_first(self):
        assert auc([True, False, False], [1.0, 0.0, float("nan")]) == 1.0
        assert math.isnan(auc([True, False], [1.0, float("nan")]))


class TestRSquared:
    def test_a_perfect_prediction_is_one_and_the_mean_zero(self):
        y = np.array([1.0, 2.0, 4.0])
        assert r_squared(y, y) == 1.0
        assert r_squared(y, np.full(3, y.mean())) == pytest.approx(0.0)

    def test_weights_weigh_the_rows(self):
        y, prediction = np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0])
        unweighted = r_squared(y, prediction)
        assert r_squared(y, prediction, np.array([1.0, 1.0, 0.01])) > unweighted

    def test_a_constant_target_is_nan(self):
        assert math.isnan(r_squared([2.0, 2.0], [1.0, 3.0]))


class TestRescaling:
    def test_auc_maps_chance_to_zero_and_perfect_to_one(self):
        assert rescaled_auc(0.5) == 0.0
        assert rescaled_auc(1.0) == 1.0
        assert rescaled_auc(0.0) == -1.0

    def test_a_grid_is_one_minus_its_error_over_the_grid_means(self):
        outcomes = [0.0, 0.0, 1.0, 1.0]  # mean 0.5, B = 0.5
        assert grid_score(outcomes, outcomes) == 1.0
        assert grid_score([0.5] * 4, outcomes) == 0.0
        assert grid_score([0.25, 0.25, 0.75, 0.75], outcomes) == pytest.approx(0.5)

    def test_a_grid_with_a_constant_outcome_has_no_score(self):
        assert math.isnan(grid_score([0.2, 0.9], [1.0, 1.0]))

    def test_clipping_floors_at_minus_one_and_keeps_nan(self):
        assert clipped(-3.0) == -1.0
        assert clipped(0.4) == 0.4
        assert math.isnan(clipped(float("nan")))


class TestStandardErrors:
    def test_the_rule_test_bound_is_the_specs(self):
        """``2·√((2n + 1) / (12n²))``: 67 a side is above 0.1 and left out,
        68 a side is the first kept."""
        assert auc_standard_error(67, 67) == pytest.approx(0.1001, abs=1e-4)
        assert auc_standard_error(67, 67) > 0.1
        assert auc_standard_error(68, 68) == pytest.approx(0.0994, abs=1e-4)
        assert auc_standard_error(68, 68) <= 0.1
        assert auc_standard_error(68, 68) == pytest.approx(
            2 * math.sqrt((2 * 68 + 1) / (12 * 68 ** 2)))

    def test_a_missing_polarity_is_an_infinite_error(self):
        assert auc_standard_error(5, 0) == math.inf
        assert auc_standard_error(0, 5) == math.inf

    def test_three_items_at_68_make_a_component_of_0_0574(self):
        item = auc_standard_error(68, 68)
        assert component_standard_error([item] * 3) == pytest.approx(0.0574, abs=1e-4)

    def test_an_empty_component_is_an_infinite_error(self):
        assert component_standard_error([]) == math.inf

    def test_the_grid_bound(self):
        """``(1/C) Σ_c 1/(2√N_c) / B``: four cells of 25 records give a spread
        of 0.1, over B = 0.5."""
        assert grid_standard_error([25] * 4, [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.2)

    def test_a_grid_with_a_constant_outcome_or_an_empty_cell_is_left_out(self):
        assert grid_standard_error([25, 25], [1.0, 1.0]) == math.inf
        assert grid_standard_error([25, 0], [0.0, 1.0]) == math.inf
        with pytest.raises(ValueError):
            grid_standard_error([25], [0.0, 1.0])

    def test_the_loss_ratio_clusters_by_game(self):
        """One game holding every loss is no better than one record: the
        bound is 1. Four equal games give √4·s / 4s = 0.5."""
        assert loss_ratio_standard_error(["g"] * 4, [1.0] * 4) == 1.0
        assert loss_ratio_standard_error(["a", "b", "c", "d"], [1.0] * 4) == 0.5

    def test_the_loss_ratio_weighs_each_record(self):
        assert loss_ratio_standard_error(["a", "b"], [1.0, 1.0], [3.0, 1.0]) == pytest.approx(
            math.sqrt(10) / 4)

    def test_no_weighted_loss_is_an_infinite_error(self):
        assert loss_ratio_standard_error([], []) == math.inf
        assert loss_ratio_standard_error(["a"], [1.0], [0.0]) == math.inf
