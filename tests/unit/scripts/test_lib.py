"""The helpers every probe script shares."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from _lib import corpora, jsonio, markdown, stats


class TestOlsCluster:
    def test_matches_the_cr1_sandwich_summed_cluster_by_cluster(self):
        X, y, _ = _heteroscedastic()
        clusters = np.arange(len(y)) % 17
        fit = stats.ols_cluster(X, y, clusters)
        bread = np.linalg.inv(X.T @ X)
        resid = y - X @ fit.beta
        meat = sum(np.outer(X[clusters == g].T @ resid[clusters == g],
                            X[clusters == g].T @ resid[clusters == g]) for g in range(17))
        n, k = X.shape
        cov = bread @ meat @ bread * (17 / 16) * ((n - 1) / (n - k))
        assert np.allclose(fit.se, np.sqrt(np.diag(cov)))

    def test_cluster_mean_se_of_nothing_is_nan(self):
        assert all(np.isnan(stats.cluster_mean_se([], [])))


def test_oof_predict_never_fits_on_the_group_it_predicts():
    # Each group's target is its own constant: a fit that saw the group would
    # predict it exactly, one that did not predicts the other groups' mean.
    groups = np.repeat(np.arange(5), 4)
    X, y = np.zeros((20, 1)), groups.astype(float)
    from sklearn.dummy import DummyRegressor

    pred = stats.oof_predict(DummyRegressor(), X, y, groups)
    assert np.allclose(pred, (y.sum() - 4 * y) / 16)


def test_pairwise_correct_counts_a_tie_as_half():
    assert stats.pairwise_correct([0.2, -0.1, 0.0]).tolist() == [1.0, 0.0, 0.5]


class TestJson:
    def test_non_finite_floats_become_null_and_numpy_unwraps(self, tmp_path):
        path = tmp_path / "out" / "r.json"
        jsonio.write_json(path, {"a": np.float32(np.nan), 2: np.arange(2), "s": {"y", "x"},
                                 "p": Path("q"), "b": np.bool_(True), "i": float("inf")})
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "a": None, "2": [0, 1], "s": ["x", "y"], "p": "q", "b": True, "i": None}

    def test_digits_round_every_float(self):
        assert jsonio.plain([1.23456789, np.float64(2.0004)], digits=3) == [1.235, 2.0]


class TestMarkdown:
    def test_a_pipe_in_a_cell_is_escaped(self):
        assert markdown.md_table(["a"], [["x | y"]]).splitlines()[-1] == "| x \\| y |"

    @pytest.mark.parametrize("value, cell", [(None, "-"), (float("nan"), "nan"),
                                             (0.123456, "0.1235"), ("n/a", "n/a")])
    def test_num(self, value, cell):
        assert markdown.num(value) == cell


def _heteroscedastic(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ [0.2, 0.7] + rng.normal(size=n) * (1 + np.abs(X[:, 1]))
    return X, y, rng.uniform(0.2, 1.0, size=n)


class TestWls:
    def test_scaling_every_weight_leaves_the_fit_and_its_errors_unchanged(self):
        # A robust SE whose middle term weights residuals by w instead of w² moves
        # with a common rescaling of the weights; the estimator itself does not.
        X, y, w = _heteroscedastic()
        base, scaled = stats.wls(X, y, w), stats.wls(X, y, 10 * w)
        assert np.allclose(base.beta, scaled.beta)
        assert np.allclose(base.se, scaled.se)

    def test_unit_weights_give_the_hc0_ordinary_fit(self):
        X, y, _ = _heteroscedastic()
        fit = stats.wls(X, y)
        resid = y - X @ fit.beta
        bread = np.linalg.inv(X.T @ X)
        hc0 = bread @ (X.T @ (X * resid[:, None] ** 2)) @ bread
        assert np.allclose(fit.se, np.sqrt(np.diag(hc0)))

    def test_non_finite_rows_and_zero_weights_are_dropped(self):
        X, y, w = _heteroscedastic()
        y[0], w[1] = np.nan, 0.0
        assert stats.wls(X, y, w).n == len(y) - 2


def test_spearman_averages_tied_ranks():
    x = np.array([1, 1, 2, 3, 3, 3, 4], dtype=float)
    y = np.array([3, 1, 2, 5, 4, 6, 7], dtype=float)
    x_ranks = [1.5, 1.5, 3, 5, 5, 5, 7]  # tied values share their mean rank
    y_ranks = [3, 1, 2, 5, 4, 6, 7]
    assert stats.spearman(x, y) == pytest.approx(np.corrcoef(x_ranks, y_ranks)[0, 1])


class TestIterMatches:
    LINE = "2026-01-01T00:00:00Z;run;M21;a;b;X|Y;Z;ABA;ABA;61\n"

    def test_reads_gzip_and_skips_malformed_and_truncated_lines(self, tmp_path):
        path = tmp_path / "m.txt.gz"
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(self.LINE + "too;few;fields\n" + self.LINE + self.LINE[:-12])
        matches = list(corpora.iter_matches(path))
        assert len(matches) == 2
        assert matches[0].deck_a.cards == ("X", "Y")
        assert corpora.decided_winner(matches[0]) is corpora.Side.A

    def test_limit_stops_early(self, tmp_path):
        path = tmp_path / "m.txt"
        path.write_text(self.LINE * 5, encoding="utf-8")
        assert len(list(corpora.iter_matches(path, limit=3))) == 3


def test_card_column_finds_the_name_column_and_skips_blank_cells(tmp_path):
    path = tmp_path / "hints.csv"
    path.write_text("card_name,score\nShock,0.5\nOpt,\n", encoding="utf-8")
    assert corpora.card_column(path, "score", sign=-1.0) == {"Shock": -0.5}
    assert corpora.card_column(path, "missing") == {}
