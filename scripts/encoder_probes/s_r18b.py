"""R18 follow-up — the nameable feature list vs the encoder on the SAME held-out cards.

``s_r18``'s curve is 5-fold CV over every card; the honest probe's 0.370 is the
encoder's own seed-42 val split. This refits the top of the curve on the
encoder's train split and scores it on the encoder's val split, so the two
numbers answer the same question, and adds the union model.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val, train = d["val"], d["train"]


def col(c):
    return np.nan_to_num(S.num(join, c))


mv = col("mv")
fly_cols = ["kw_flying", "kw_flying_anywhere", "ph_flying_grant"]
kw_rest = [c for c in sorted(join.columns) if c.startswith("kw_") and c not in fly_cols]
ph_cols = sorted(c for c in join.columns if c.startswith("ph_"))
year = pd.to_numeric(join["first_year"], errors="coerce").to_numpy(float)
year = np.where(np.isfinite(year), year, np.nanmedian(year))
ERA = [("first_year", year),
       ("first_rarity", pd.Categorical(join["first_rarity"].fillna("unknown")).codes.astype(float)),
       ("first_set_code", pd.Categorical(join["first_set_code"].fillna("unknown")).codes.astype(float))]
BLOCKS = [
    ("MV", [("mv", mv), ("mv2", mv**2)]),
    ("+ type class", [(c, col(c)) for c in
                      ("is_creature", "is_instant", "is_sorcery", "is_artifact",
                       "is_enchantment", "is_land", "is_planeswalker", "is_aura",
                       "is_equipment", "is_vehicle", "is_legendary", "is_basic")]),
    ("+ P/T", [("power", col("power")), ("toughness", col("toughness")),
               ("has_pt", np.isfinite(S.num(join, "power")).astype(float))]),
    ("+ flying", [(c, col(c)) for c in fly_cols if c in join.columns]),
    ("+ keyword flags", [(c, col(c)) for c in kw_rest if c != "kw_count"]
                        + [("kw_count", col("kw_count"))]),
    ("+ phrase / archetype flags", [(c, col(c)) for c in ph_cols]),
    ("+ pips", [(c, col(c)) for c in
                ("pip_w", "pip_u", "pip_b", "pip_r", "pip_g", "pip_c", "pips_total",
                 "generic", "n_colors", "is_gold", "hybrid", "has_x",
                 "is_colorless_cost", "phyrexian")]),
    ("+ ability-line count", [(c, col(c)) for c in
                              ("n_abilities", "n_lines", "n_body_lines", "n_words",
                               "n_body_words", "n_sym", "n_types", "big_tribe")]),
    ("+ era / rarity", None),
]

cols, cat = [], []
for bname, block in BLOCKS:
    if block is None:
        cols += ERA
        cat += [len(cols) - 1, len(cols) - 2]
    else:
        cols += block
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
            b, _, _ = S.wls(Xa[tr], y[tr], w[tr])
            p_tr, p_va = Xa[tr] @ b, Xa[va] @ b
        else:
            r2, alpha, coef, b0 = S.ridge_fit_eval(X[tr], y[tr], X[va], y[va], w[tr])
            p_tr, p_va = X[tr] @ coef + b0, X[va] @ coef + b0
        rows.append([head, vname, int(tr.sum()), int(va.sum()),
                     f"{pl._r2(y[tr], p_tr, w[tr]):.4f}",
                     f"{pl._r2(y[va], p_va, w[va]):.4f}"])
        print(rows[-1], flush=True)
out["same_split"] = rows

with open(S.OUT / "s_r18b.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
