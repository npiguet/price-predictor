"""Tests for the extracted ridge-probe harness.

The solver is checked against the closed-form weighted ridge solution on a small
synthetic matrix, so a refactor that changes the maths is caught rather than
merely a refactor that changes the API.
"""

from __future__ import annotations

import numpy as np
import pytest

from price_predictor.application.ridge_probes import (
    ALPHA_GRID,
    HEADS,
    _choose_alpha,
    _pearson,
    _r2,
    _ridge_solve,
    build_label_table,
    fit_probes,
    head_effective_n,
    head_weight,
    load_labels,
    to_logit,
)

_WIN_RATE_HEADER = (
    "card_name;wins_when_played;wins_when_in_deck;losses_when_played;"
    "losses_when_in_deck;raw_score_play;shrunk_score_play;raw_score_draw;"
    "shrunk_score_draw;raw_played_rate;shrunk_played_rate;raw_cast_lift;"
    "shrunk_cast_lift;raw_color_lift_W;shrunk_color_lift_W;raw_color_lift_U;"
    "shrunk_color_lift_U;raw_color_lift_B;shrunk_color_lift_B;raw_color_lift_R;"
    "shrunk_color_lift_R;raw_color_lift_G;shrunk_color_lift_G"
)


def _closed_form_ridge(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float,
) -> tuple[np.ndarray, float]:
    """Weighted ridge with an exact, unpenalised intercept, solved directly."""
    sw = w / w.sum()
    x_mean = sw @ X
    y_mean = float(sw @ y)
    Xc = X - x_mean
    yc = y - y_mean
    gram = Xc.T @ (Xc * w[:, None]) + alpha * np.eye(X.shape[1])
    coef = np.linalg.solve(gram, (Xc * w[:, None]).T @ yc)
    return coef, y_mean - float(x_mean @ coef)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    true_coef = np.array([1.5, -0.5, 0.0, 2.0])
    y = X @ true_coef + 3.0 + 0.01 * rng.normal(size=60)
    w = rng.uniform(0.2, 1.0, size=60)
    return X, y, w


class TestRidgeSolve:
    def test_matches_the_closed_form_solution(self, synthetic):
        X, y, w = synthetic
        fits = _ridge_solve(X, y, w, ALPHA_GRID)
        for alpha in ALPHA_GRID:
            coef, intercept = fits[alpha]
            want_coef, want_intercept = _closed_form_ridge(X, y, w, alpha)
            np.testing.assert_allclose(coef, want_coef, atol=1e-9)
            assert intercept == pytest.approx(want_intercept, abs=1e-9)

    def test_solves_every_alpha_in_one_call(self, synthetic):
        X, y, w = synthetic
        fits = _ridge_solve(X, y, w, ALPHA_GRID)
        assert sorted(fits) == sorted(float(a) for a in ALPHA_GRID)

    def test_a_larger_penalty_shrinks_the_coefficients(self, synthetic):
        X, y, w = synthetic
        fits = _ridge_solve(X, y, w, ALPHA_GRID)
        norms = [np.linalg.norm(fits[a][0]) for a in ALPHA_GRID]
        assert norms == sorted(norms, reverse=True)

    def test_recovers_a_noiseless_linear_target_at_a_small_penalty(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 3))
        true_coef = np.array([0.7, -1.2, 0.4])
        y = X @ true_coef + 2.5
        w = np.ones(200)
        coef, intercept = _ridge_solve(X, y, w, [1e-8])[1e-8]
        np.testing.assert_allclose(coef, true_coef, atol=1e-5)
        assert intercept == pytest.approx(2.5, abs=1e-5)


class TestMetrics:
    def test_r2_is_one_for_an_exact_fit(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _r2(y, y) == pytest.approx(1.0)

    def test_r2_is_zero_for_the_weighted_mean_predictor(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([1.0, 1.0, 1.0, 1.0])
        pred = np.full_like(y, y.mean())
        assert _r2(y, pred, w) == pytest.approx(0.0)

    def test_r2_is_nan_when_the_target_has_no_variance(self):
        y = np.full(5, 2.0)
        assert np.isnan(_r2(y, np.zeros(5)))

    def test_pearson_of_a_line_is_one(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _pearson(a, 2 * a + 1) == pytest.approx(1.0)

    def test_pearson_needs_two_points(self):
        assert np.isnan(_pearson(np.array([1.0]), np.array([2.0])))


class TestToLogit:
    def test_inverts_the_logistic(self):
        p = np.array([0.25, 0.5, 0.75])
        np.testing.assert_allclose(1.0 / (1.0 + np.exp(-to_logit(p))), p, atol=1e-12)

    def test_clips_the_endpoints_instead_of_returning_infinity(self):
        out = to_logit(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(out))


class TestChooseAlpha:
    def test_picks_a_grid_member(self, synthetic):
        X, y, w = synthetic
        alpha, cv_r2 = _choose_alpha(X, y, w, folds=3)
        assert alpha in [float(a) for a in ALPHA_GRID]
        assert cv_r2 > 0.9  # the target is nearly noiseless

    def test_honours_an_explicit_grid(self, synthetic):
        X, y, w = synthetic
        alpha, _ = _choose_alpha(X, y, w, folds=3, alphas=(1.0, 10.0))
        assert alpha in (1.0, 10.0)

    def test_is_deterministic_under_a_fixed_seed(self, synthetic):
        X, y, w = synthetic
        assert _choose_alpha(X, y, w, folds=3) == _choose_alpha(X, y, w, folds=3)


class TestHeadWeighting:
    def test_weight_is_n_over_n_plus_k(self):
        assert head_weight(20.0, k=20.0) == pytest.approx(0.5)
        assert head_weight(60.0, k=20.0) == pytest.approx(0.75)

    def test_a_dead_cell_weighs_nothing(self):
        assert head_weight(None) == 0.0
        assert head_weight(0.0) == 0.0

    def test_played_rate_n_is_the_in_deck_count(self):
        record = {
            "wins_when_in_deck": 30, "losses_when_in_deck": 20,
            "wins_when_played": 10, "losses_when_played": 5,
        }
        assert head_effective_n(record, "played_rate") == 50.0

    def test_cast_lift_n_is_the_smaller_side_of_the_split(self):
        record = {
            "wins_when_in_deck": 30, "losses_when_in_deck": 20,
            "wins_when_played": 10, "losses_when_played": 5,
        }
        # played = 15, not played = 35; the lift is limited by the smaller arm.
        assert head_effective_n(record, "cast_lift") == 15.0

    def test_score_play_n_is_recovered_from_the_raw_shrunk_ratio(self):
        # shrunk / raw == n / (n + k); with k = 20 and n = 80 the ratio is 0.8.
        record = {
            "wins_when_in_deck": 200, "losses_when_in_deck": 200,
            "wins_when_played": 60, "losses_when_played": 40,
            "raw_score_play": 0.5, "shrunk_score_play": 0.4,
            "raw_score_draw": None, "shrunk_score_draw": None,
        }
        assert head_effective_n(record, "score_play") == pytest.approx(80.0)

    def test_score_play_falls_back_to_half_the_in_deck_count(self):
        """A numerator at the rounding floor cannot carry the ratio identity."""
        record = {
            "wins_when_in_deck": 200, "losses_when_in_deck": 200,
            "wins_when_played": 60, "losses_when_played": 40,
            "raw_score_play": 0.0, "shrunk_score_play": 0.0,
            "raw_score_draw": None, "shrunk_score_draw": None,
        }
        assert head_effective_n(record, "score_play") == 200.0


class TestLoadLabelsAndTable:
    def _write(self, tmp_path, rows: list[str]):
        path = tmp_path / "cards-win-rates.txt"
        path.write_text("\n".join([_WIN_RATE_HEADER, *rows]) + "\n", encoding="utf-8")
        return path

    def _row(self, name: str) -> str:
        cells = [name, "60", "200", "40", "200"]
        cells += ["0.50000", "0.40000", "0.30000", "0.24000"]   # score_play/draw
        cells += ["0.60000", "0.55000", "0.10000", "0.08000"]   # played_rate/cast_lift
        for _ in "WUBRG":
            cells += ["0.02000", "0.01500"]
        return ";".join(cells)

    def test_counters_parse_as_int_and_labels_as_float(self, tmp_path):
        path = self._write(tmp_path, [self._row("Serra Angel")])
        labels = load_labels(path)
        record = labels["Serra Angel"]
        assert record["wins_when_played"] == 60
        assert isinstance(record["wins_when_played"], int)
        assert record["shrunk_score_play"] == pytest.approx(0.4)

    def test_an_empty_cell_is_none_not_zero(self, tmp_path):
        row = self._row("Vanilla").split(";")
        row[5] = row[6] = ""  # raw/shrunk score_play carry no signal
        path = self._write(tmp_path, [";".join(row)])
        record = load_labels(path)["Vanilla"]
        assert record["raw_score_play"] is None
        assert record["shrunk_score_play"] is None

    def test_a_short_row_is_skipped(self, tmp_path):
        path = self._write(tmp_path, [self._row("Good"), "Truncated;1;2"])
        assert set(load_labels(path)) == {"Good"}

    def test_accented_names_survive_the_utf8_read(self, tmp_path):
        path = self._write(tmp_path, [self._row("Márton Stromgald")])
        assert "Márton Stromgald" in load_labels(path)

    def test_the_table_is_row_aligned_with_the_requested_names(self, tmp_path):
        path = self._write(
            tmp_path, [self._row("A"), self._row("B"), self._row("C")]
        )
        table = build_label_table(["C", "A", "B"], win_rates_path=path)
        assert list(table["name"]) == ["C", "A", "B"]

    def test_an_unlabelled_name_gets_zero_weight_not_a_missing_row(self, tmp_path):
        path = self._write(tmp_path, [self._row("Known")])
        table = build_label_table(["Known", "Unknown"], win_rates_path=path)
        assert len(table) == 2
        assert bool(table.loc[1, "has_label"]) is False
        for head in HEADS:
            assert table.loc[1, f"w_{head}"] == 0.0
            assert np.isnan(table.loc[1, f"shrunk_{head}"])

    def test_val_names_choose_the_split(self, tmp_path):
        path = self._write(tmp_path, [self._row("A"), self._row("B")])
        table = build_label_table(["A", "B"], win_rates_path=path, val_names={"B"})
        assert list(table["split"]) == ["train", "val"]

    def test_a_repeated_name_is_primary_only_once(self, tmp_path):
        path = self._write(tmp_path, [self._row("A")])
        table = build_label_table(["A", "A"], win_rates_path=path)
        assert table["is_primary"].sum() == 1


class TestFitProbes:
    def _table_and_embeddings(self, n: int = 120, dim: int = 5):
        """A label table whose ``score_play`` is an exact linear function of X."""
        import pandas as pd

        rng = np.random.default_rng(3)
        X = rng.normal(size=(n, dim))
        coef = np.array([1.0, -0.5, 0.25, 0.0, 0.75])[:dim]
        rows = []
        for i in range(n):
            row = {
                "name": f"card{i}",
                "split": "val" if i % 5 == 0 else "train",
                "is_primary": True,
            }
            for head in HEADS:
                row[f"shrunk_{head}"] = float(X[i] @ coef) + 0.1
                row[f"w_{head}"] = 1.0
            rows.append(row)
        return pd.DataFrame(rows), X

    def test_recovers_a_linear_label_from_the_embedding(self):
        table, X = self._table_and_embeddings()
        probes = fit_probes(table, X, heads=("score_play",), folds=3)
        assert probes.probes["score_play"].metrics["cv_r2"] > 0.99

    def test_reports_both_splits(self):
        table, X = self._table_and_embeddings()
        probes = fit_probes(table, X, heads=("score_play",), folds=3)
        metrics = probes.probes["score_play"].metrics
        assert metrics["train_n"] + metrics["val_n"] == len(table)

    def test_honest_mode_fits_on_the_train_split_only(self):
        table, X = self._table_and_embeddings()
        honest = fit_probes(table, X, heads=("score_play",), mode="honest", folds=3)
        n_train = int((table["split"] == "train").sum())
        assert honest.probes["score_play"].n_fit == n_train

    def test_played_rate_is_fitted_in_both_spaces(self):
        table, X = self._table_and_embeddings()
        probes = fit_probes(table, X, heads=("played_rate",), folds=3)
        assert set(probes.probes) == {"played_rate", "played_rate@logit"}

    def test_zero_weight_rows_are_dropped_from_the_fit(self):
        table, X = self._table_and_embeddings()
        table.loc[:9, "w_score_play"] = 0.0
        probes = fit_probes(table, X, heads=("score_play",), folds=3)
        assert probes.probes["score_play"].n_fit == len(table) - 10

    def test_an_unknown_mode_is_rejected(self):
        table, X = self._table_and_embeddings()
        with pytest.raises(ValueError, match="mode must be"):
            fit_probes(table, X, mode="sideways")

    def test_probe_predict_applies_coefficients_and_intercept(self):
        table, X = self._table_and_embeddings()
        probe = fit_probes(table, X, heads=("score_play",), folds=3).probes["score_play"]
        np.testing.assert_allclose(
            probe.predict(X), X @ probe.coef + probe.intercept, atol=1e-12
        )

    def test_probe_set_key_names_mode_and_weighting(self):
        table, X = self._table_and_embeddings()
        weighted = fit_probes(table, X, heads=("score_play",), folds=3)
        unweighted = fit_probes(
            table, X, heads=("score_play",), weighted=False, folds=3
        )
        assert weighted.key == "fidelity_w"
        assert unweighted.key == "fidelity_u"
