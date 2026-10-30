"""The helpers every probe script shares."""

from __future__ import annotations

import gzip

import numpy as np
import pytest
from _lib import corpora, stats


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
