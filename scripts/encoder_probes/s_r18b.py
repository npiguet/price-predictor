"""R18 follow-up — the nameable feature list vs the encoder on the SAME held-out cards.

``s_r18``'s curve is 5-fold CV over every card; the honest probe's 0.370 is the
encoder's own seed-42 val split. This refits the top of the curve on the
encoder's train split and scores it on the encoder's val split, so the two
numbers answer the same question, and adds the union model.
"""

from __future__ import annotations

import json

import numpy as np
import probe_lib as pl
import s_common as S
from _lib.stats import wls
from sklearn.ensemble import HistGradientBoostingRegressor

d = S.load_all()
join, emb = d["join"], d["emb"]
val, train = d["val"], d["train"]


cols = [c for _, block in S.nameable_blocks(join) for c in block]
cat = S.categorical_indices(cols)
F = np.column_stack([c for _, c in cols])
names = [n for n, _ in cols]

out = {"n_features": len(names)}
rows = []
for head in ("score_play", "played_rate"):
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    tr, va = have & train, have & val
    variants = {
        "nameable features (GBM)": (F, "gbm"),
        "nameable features (linear)": (F, "lin"),
        "512-d embedding (ridge probe)": (emb, "ridge"),
        "embedding + nameable features (GBM)": (np.column_stack([emb, F]), "gbm"),
    }
    for vname, (X, kind) in variants.items():
        if kind == "gbm":
            cf = cat if X.shape[1] == F.shape[1] else [c + emb.shape[1] for c in cat]
            g = HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.08, early_stopping=True,
                validation_fraction=0.12, random_state=42,
                categorical_features=cf or None)
            g.fit(X[tr], y[tr], sample_weight=w[tr])
            p_tr, p_va = g.predict(X[tr]), g.predict(X[va])
        elif kind == "lin":
            Xa = np.column_stack([np.ones(len(X)), X])
            b = wls(Xa[tr], y[tr], w[tr]).beta
            p_tr, p_va = Xa[tr] @ b, Xa[va] @ b
        else:
            _, _, coef, b0 = pl.ridge_fit_eval(X[tr], y[tr], X[va], y[va], w[tr])
            p_tr, p_va = X[tr] @ coef + b0, X[va] @ coef + b0
        rows.append([head, vname, int(tr.sum()), int(va.sum()),
                     f"{pl._r2(y[tr], p_tr, w[tr]):.4f}",
                     f"{pl._r2(y[va], p_va, w[va]):.4f}"])
        print(rows[-1], flush=True)
out["same_split"] = rows

with open(S.OUT / "s_r18b.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
