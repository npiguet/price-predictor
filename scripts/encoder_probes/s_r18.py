"""R18 — the nameability curve: how much of the encoder's taste is a feature list."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import probe_lib as pl
import s_common as S
from _lib.stats import wls
from sklearn.ensemble import HistGradientBoostingRegressor

d = S.load_all()
join, emb = d["join"], d["emb"]
pf = d["pf"]
pred_f = pl.predict_labels(emb, pf)

TARGETS = {}
for head in ("score_play", "played_rate"):
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    TARGETS[f"{head} — label"] = (y, w, have, head)
    TARGETS[f"{head} — encoder (fidelity pred)"] = (
        np.asarray(pred_f[head], float), w, have, head)


def cv_r2(X, y, w, cat_idx, folds=5, seed=42, model="gbm"):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    pred = np.empty(len(y))
    for part in np.array_split(order, folds):
        m = np.ones(len(y), bool)
        m[part] = False
        if model == "gbm":
            g = HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                early_stopping=True, validation_fraction=0.12, random_state=42,
                categorical_features=cat_idx or None)
            g.fit(X[m], y[m], sample_weight=w[m])
            pred[part] = g.predict(X[part])
        else:
            Xa = np.column_stack([np.ones(m.sum()), X[m]])
            b = wls(Xa, y[m], w[m]).beta
            pred[part] = np.column_stack([np.ones(len(part)), X[part]]) @ b
    return pl._r2(y, pred, w)


rows = []
blocks = S.nameable_blocks(join)
for tname, (y, w, have, head) in TARGETS.items():
    cols: list[tuple[str, np.ndarray]] = []
    for bname, block in blocks:
        cols += block
        cat = S.categorical_indices(cols)
        X = np.column_stack([c for _, c in cols])[have]
        r_gbm = cv_r2(X, y[have], w[have], cat, model="gbm")
        r_lin = cv_r2(X, y[have], w[have], cat, model="linear")
        rows.append({"target": tname, "block": bname, "k": len(cols),
                     "gbm_cv_r2": r_gbm, "lin_cv_r2": r_lin, "n": int(have.sum())})
        print(f"{tname:42s} {bname:28s} k={len(cols):3d} gbm={r_gbm:.4f} lin={r_lin:.4f}",
              flush=True)

curve = pd.DataFrame(rows)
curve.to_pickle(S.OUT / "s_r18_curve.pkl")

out = {"curve": curve.to_dict("records")}

# reconciliation numbers
ph = d["ph"]
out["reference"] = {
    "equivalence_class_text_explainable_vs_members": 0.515,
    "honest_val_r2_score_play": ph.probes["score_play"].metrics["val_r2"],
    "honest_val_r2_played_rate": ph.probes["played_rate"].metrics["val_r2"],
    "fidelity_insample_r2_score_play": pf.probes["score_play"].metrics["in_sample_r2"],
    "fidelity_insample_r2_played_rate": pf.probes["played_rate"].metrics["in_sample_r2"],
}


def weighted_var(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.average((v - np.average(v, weights=w)) ** 2, weights=w))


# variance budget: how much of the *label* variance the encoder's taste itself carries
for y, w, have, head in list(TARGETS.values())[::2]:
    p = np.asarray(pred_f[head], float)
    out[f"budget_{head}"] = {"var_label": weighted_var(y[have], w[have]),
                             "var_encoder_taste": weighted_var(p[have], w[have])}

with open(S.OUT / "s_r18.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out["reference"], indent=1))
