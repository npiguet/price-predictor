"""Small statistics the probe scripts share.

Every function drops non-finite inputs before computing and returns NaN, never
raises, when there is too little left to compute on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats


def finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    """``x`` and ``y`` as float arrays, restricted to the rows finite in both."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _degenerate(x: np.ndarray, y: np.ndarray, min_n: int) -> bool:
    return len(x) < min_n or x.std() == 0 or y.std() == 0


def pearson(x, y, min_n: int = 3) -> float:
    x, y = finite_pair(x, y)
    if _degenerate(x, y, min_n):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def pearson_fisher_se(r: float, n: int) -> float:
    """Large-sample SE of a Pearson ``r``: the Fisher-z SE, delta-mapped back to r."""
    if n < 4 or not np.isfinite(r):
        return float("nan")
    return float((1 - r * r) / np.sqrt(n - 3))


def spearman(x, y, min_n: int = 3) -> float:
    """Spearman rank correlation, ties given their average rank."""
    x, y = finite_pair(x, y)
    if _degenerate(x, y, min_n):
        return float("nan")
    return float(_scipy_stats.spearmanr(x, y).statistic)


def auc(labels, scores) -> float:
    """ROC AUC of ``scores`` against yes/no ``labels``, ties counted half; NaN
    without both labels."""
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.all() or not labels.any():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def bootstrap_ci(values, stat=np.mean, n_boot: int = 4000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap interval of ``stat`` over the finite ``values``."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = stat(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def mean_se(values) -> tuple[float, float, int]:
    """Mean, its standard error and the count, over the finite non-None ``values``."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return float("nan"), float("nan"), 0
    se = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")
    return float(a.mean()), se, int(a.size)


def wcorr(x, y, w=None) -> float:
    """Weighted Pearson correlation over the rows finite in ``x``, ``y`` and ``w``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[ok], y[ok], w[ok]
    if len(x) < 3 or w.sum() <= 0:
        return float("nan")
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cxy = np.average((x - mx) * (y - my), weights=w)
    cxx = np.average((x - mx) ** 2, weights=w)
    cyy = np.average((y - my) ** 2, weights=w)
    return float(cxy / np.sqrt(cxx * cyy)) if cxx > 0 and cyy > 0 else float("nan")


def weighted_r2(y, pred, w=None) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    mean = float((w * y).sum() / w.sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass(frozen=True)
class LinearFit:
    """Coefficients of a least-squares fit with their standard errors."""

    beta: np.ndarray
    se: np.ndarray
    r2: float
    n: int
    rank: int

    @property
    def t(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.beta / self.se


def wls(X, y, w=None) -> LinearFit:
    """Weighted least squares with HC0 (heteroscedasticity-robust) standard errors.

    Rows with a non-finite value or a non-positive weight are dropped. The
    sandwich's middle term weights each squared residual by ``w²``: the
    estimator is ``(X'WX)⁻¹ X'W y``, so its score contributions are ``w·r·x``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & np.isfinite(w) & (w > 0)
    X, y, w = X[ok], y[ok], w[ok]
    bread = np.linalg.pinv(X.T @ (X * w[:, None]))
    beta = bread @ (X.T @ (w * y))
    resid = y - X @ beta
    score = X * (w * resid)[:, None]
    cov = bread @ (score.T @ score) @ bread
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return LinearFit(beta, se, weighted_r2(y, X @ beta, w), int(ok.sum()),
                     int(np.linalg.matrix_rank(X)))


def ols(X, y) -> LinearFit:
    """Ordinary least squares with classical standard errors.

    Tolerates a rank-deficient design: ``lstsq`` returns the minimum-norm
    solution and the residual degrees of freedom count the rank, not the columns.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    cov = float(resid @ resid) / (len(y) - rank) * np.linalg.pinv(X.T @ X)
    return LinearFit(beta, np.sqrt(np.abs(np.diag(cov))),
                     1.0 - float(resid.var() / y.var()), len(y), int(rank))


def participation_ratio(eigenvalues) -> float:
    """``(Σλ)² / Σλ²``: how many dimensions a space effectively uses.

    ``d`` when every direction carries equal variance, 1 when one carries it
    all. Scale-free, so variance shares read the same as eigenvalues.
    """
    lam = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    squares = float((lam ** 2).sum())
    return float(lam.sum()) ** 2 / squares if squares > 0 else float("nan")


def covariance_eigenvalues(points) -> np.ndarray:
    """Eigenvalues of the sample covariance of a point cloud (rows are points)."""
    points = np.asarray(points, dtype=np.float64)
    centred = points - points.mean(axis=0, keepdims=True)
    return np.linalg.eigvalsh(centred.T @ centred / max(1, len(centred) - 1))


def softmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    e = np.exp(x - x.max())
    return e / e.sum()


def kl_divergence(p, q, floor: float = 0.0) -> float:
    """``KL(p‖q)`` over the support of ``p``; ``floor`` guards a zero in ``q``."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    s = p > 0
    return float(np.sum(p[s] * np.log(p[s] / np.maximum(q[s], floor))))


def js_distance(p, q) -> float:
    """Jensen-Shannon distance: the square root of the JS divergence."""
    m = 0.5 * (np.asarray(p, dtype=float) + np.asarray(q, dtype=float))
    return float(np.sqrt(max(0.0, 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))))


def ridge_cv_r2(X, Y, groups, alpha: float, n_splits: int = 5, **ridge_options) -> np.ndarray:
    """Out-of-fold R² of a grouped ridge fit: one per column of ``Y``, then pooled."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold

    pred = np.zeros_like(Y)
    for train, test in GroupKFold(n_splits=n_splits).split(X, Y, groups):
        pred[test] = Ridge(alpha=alpha, **ridge_options).fit(X[train], Y[train]).predict(X[test])
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return np.append(1 - ss_res / ss_tot, 1 - ss_res.sum() / ss_tot.sum())
