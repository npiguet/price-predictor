"""Builder-validation statistics (FR-042, SC-007)."""

from __future__ import annotations

import math

import numpy as np

from draft.application.validate_builder import compute_diagnostic


def test_perfect_rank_agreement() -> None:
    picker = [1.0, 2.0, 3.0, 4.0, 5.0]
    sa = [1.5, 2.5, 3.5, 4.5, 5.5]  # SA uniformly 0.5 above, same ranking
    sa2 = [1.4, 2.6, 3.4, 4.6, 5.4]
    diag = compute_diagnostic(picker, sa, sa2)
    assert math.isclose(diag.picker_vs_sa_spearman, 1.0)
    assert math.isclose(diag.gap_median, 0.5)
    assert math.isclose(diag.gap_iqr, 0.0)
    assert diag.n_pools == 5


def test_gap_median_and_iqr() -> None:
    picker = [0.0, 0.0, 0.0, 0.0]
    sa = [1.0, 2.0, 3.0, 4.0]      # gaps = 1,2,3,4
    sa2 = [1.0, 2.0, 3.0, 4.0]
    diag = compute_diagnostic(picker, sa, sa2)
    # median of [1,2,3,4] = 2.5; IQR = Q3 - Q1 = 3.25 - 1.75 = 1.5
    assert math.isclose(diag.gap_median, 2.5)
    assert math.isclose(diag.gap_iqr, 1.5)


def test_sa_vs_sa_reference_is_independent_of_picker() -> None:
    picker = [5.0, 4.0, 3.0, 2.0, 1.0]          # anti-correlated with SA
    sa = [1.0, 2.0, 3.0, 4.0, 5.0]
    sa2 = [1.0, 2.0, 3.0, 4.0, 5.0]              # SA tracks itself perfectly
    diag = compute_diagnostic(picker, sa, sa2)
    assert math.isclose(diag.picker_vs_sa_spearman, -1.0)
    assert math.isclose(diag.sa_vs_sa_spearman, 1.0)


def test_single_pool_correlation_is_nan() -> None:
    diag = compute_diagnostic([1.0], [1.0], [1.0])
    assert math.isnan(diag.picker_vs_sa_spearman)
    assert math.isnan(diag.sa_vs_sa_spearman)


def test_matches_scipy_on_noisy_data() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    picker = list(rng.standard_normal(50))
    sa = list(np.asarray(picker) + rng.standard_normal(50) * 0.3)
    sa2 = list(np.asarray(picker) + rng.standard_normal(50) * 0.3)
    diag = compute_diagnostic(picker, sa, sa2)
    expected, _ = spearmanr(picker, sa)
    assert math.isclose(diag.picker_vs_sa_spearman, float(expected))
