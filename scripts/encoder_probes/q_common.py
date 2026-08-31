"""Shared helpers for the Q series (R6 pool-query specialization, R7 decodability).

Read-only with respect to the other agents' probe scripts: everything here
either imports ``probe_lib`` or re-implements the small pieces the Q battery
needs (an honest-split ridge over an arbitrary design matrix and target).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

SCRATCH = pl.SCRATCH
CARD_TABLE = SCRATCH / "card_table.pkl"  # written by p0b_card_table.py

N_BLOCKS = 8
BLOCK = pl.TEXT_DIM // N_BLOCKS  # 64

SD_SCORE_PLAY = 0.06181
SD_PLAYED_RATE = 0.1247


def load_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """Primary joined cards + their 512-dim text vectors, card-table merged."""
    join = pl.build_join()
    join = join[join["is_primary"]].reset_index(drop=True)
    emb = pl.load_embedding_matrix(list(join["name"]), join)
    table = pd.read_pickle(CARD_TABLE)
    skip = set(join.columns) | {"card_name"}
    feature_cols = [c for c in table.columns if c not in skip]
    merged = join.merge(
        table[["card_name", *feature_cols]],
        left_on="name", right_on="card_name", how="left",
    )
    return merged, emb


def honest_ridge(
    X: np.ndarray,
    y: np.ndarray,
    is_train: np.ndarray,
    is_val: np.ndarray,
    keep: np.ndarray | None = None,
    *,
    alphas=(0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
) -> dict:
    """Ridge fitted on the encoder's train split; val R2 is the report number.

    Alpha is chosen by 5-fold CV inside the train split, exactly as
    ``probe_lib.fit_probes`` does, so the numbers here are comparable to the
    p0 honest/unweighted table.
    """
    keep = np.ones(len(y), dtype=bool) if keep is None else keep
    good = keep & np.isfinite(y)
    tr = good & is_train
    va = good & is_val
    w = np.ones(int(tr.sum()))
    alpha, cv_r2 = pl._choose_alpha(X[tr], y[tr], w, folds=5)
    coef, b = pl._ridge_solve(X[tr], y[tr], w, [alpha])[alpha]
    pred_va = X[va] @ coef + b
    pred_tr = X[tr] @ coef + b
    return {
        "alpha": alpha,
        "cv_r2": cv_r2,
        "train_r2": pl._r2(y[tr], pred_tr),
        "val_r2": pl._r2(y[va], pred_va),
        "val_pearson": pl._pearson(y[va], pred_va),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "coef": coef,
        "intercept": b,
        "val_mask": va,
        "val_pred": pred_va,
    }


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank AUC of a binary target against a continuous score."""
    y_true = np.asarray(y_true).astype(bool)
    n_pos, n_neg = int(y_true.sum()), int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y_true].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def participation_ratio(matrix: np.ndarray) -> float:
    """Effective rank of a point cloud: (sum lambda)^2 / sum lambda^2."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    cov = (centred.T @ centred) / max(1, len(centred) - 1)
    evals = np.linalg.eigvalsh(cov.astype(np.float64))
    evals = np.clip(evals, 0.0, None)
    return float(evals.sum() ** 2 / (evals ** 2).sum())


def first_pc_scores(matrix: np.ndarray) -> np.ndarray:
    """Per-card projection onto the block's leading principal component."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred.astype(np.float64), full_matrices=False)
    return centred @ vt[0]


def fmt(x: float, n: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{n}f}"
