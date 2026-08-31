"""Shared CPU-only scaffolding for the axis-structure probes (R3/R4/R5/R8/R17/R18).

Never loads the encoder and never touches the GPU: everything here reads the
cached ``.npz`` embeddings and the fitted ridge probes that ``p0_build`` left
in ``output/encoder-probes/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "encoder_probes"))

import probe_lib as pl  # noqa: E402

OUT = pl.SCRATCH
GROUNDING = OUT / "card_table.pkl"  # written by p0b_card_table.py

SD = {
    "score_play": 0.0618,
    "score_draw": 0.0618,
    "played_rate": 0.1247,
    "cast_lift": 0.0845,
    "color_lift": 0.024,
}

COLORS = tuple("WUBRG")
ALLIED = {frozenset(p) for p in (("W", "U"), ("U", "B"), ("B", "R"), ("R", "G"), ("G", "W"))}
ENEMY = {frozenset(p) for p in (("W", "B"), ("U", "R"), ("B", "G"), ("R", "W"), ("G", "U"))}

_CACHE: dict = {}


def load_all() -> dict:
    """``join`` (primary rows only), row-aligned ``emb``, probes, card table."""
    if _CACHE:
        return _CACHE
    join = pl.build_join()
    join = join[join["is_primary"]].reset_index(drop=True)
    emb = pl.load_embedding_matrix(list(join["name"]), join)

    table = pd.read_pickle(GROUNDING)
    extra = pd.read_pickle(OUT / "l_extra_flags.pkl")
    table = table.merge(extra, on="card_name", how="left")
    feat_cols = [c for c in table.columns if c not in set(join.columns) | {"card_name"}]
    table = table[["card_name"] + feat_cols].rename(columns={"card_name": "name"})
    join = join.merge(table, on="name", how="left")

    _CACHE.update(
        join=join,
        emb=emb,
        pf=pl.load_probes("fidelity", True),
        ph=pl.load_probes("honest", True),
        val=(join["split"] == "val").to_numpy(),
        train=(join["split"] == "train").to_numpy(),
    )
    return _CACHE


# ── small statistical helpers ───────────────────────────────────────────


def wcorr(a, b, w=None) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if w is None:
        w = np.ones(len(a))
    w = np.asarray(w, float) * m
    if w.sum() <= 0:
        return float("nan")
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    am = (w * a).sum() / w.sum()
    bm = (w * b).sum() / w.sum()
    cov = (w * (a - am) * (b - bm)).sum()
    va = (w * (a - am) ** 2).sum()
    vb = (w * (b - bm) ** 2).sum()
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else float("nan")


def corr_se(r: float, n: int) -> float:
    """Fisher-z standard error mapped back to r units (first order)."""
    if n < 4 or not np.isfinite(r):
        return float("nan")
    return float((1 - r * r) / np.sqrt(n - 3))


def cosine(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Weighted least squares with HC0 SEs. Returns (beta, se, t)."""
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    xtx = Xw.T @ Xw
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (Xw.T @ yw)
    resid = y - X @ beta
    meat = (X * (w * resid**2)[:, None]).T @ X
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    return beta, se, t


def ridge_fit_eval(X_tr, y_tr, X_te, y_te, w_tr=None, alphas=pl.ALPHA_GRID):
    """Ridge with CV-chosen alpha on the train part; returns (r2_te, alpha, coef, b)."""
    w_tr = np.ones(len(y_tr)) if w_tr is None else w_tr
    alpha, _ = pl._choose_alpha(X_tr, y_tr, w_tr, folds=5)
    coef, b = pl._ridge_solve(X_tr, y_tr, w_tr, [alpha])[alpha]
    pred = X_te @ coef + b
    return pl._r2(y_te, pred), alpha, coef, b


def kfold_r2(X, y, model_fn, folds=5, seed=42):
    """Out-of-fold R² for an sklearn-style ``model_fn()``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    pred = np.empty(len(y))
    for part in parts:
        mask = np.ones(len(y), bool)
        mask[part] = False
        m = model_fn()
        m.fit(X[mask], y[mask])
        pred[part] = m.predict(X[part])
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot, pred


def card_colors(join: pd.DataFrame) -> np.ndarray:
    """(N, 5) bool: does the card's mana cost contain a pip of this colour."""
    return np.stack([join[f"pip_{c.lower()}"].fillna(0).to_numpy() > 0 for c in COLORS], 1)


def num(join: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(join[col], errors="coerce").to_numpy(float)


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}f}"


def md_table(rows, headers) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)
