"""R18 — the nameability curve: how much of the encoder's taste is a feature list."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
pf = d["pf"]
pred_f = pl.predict_labels(emb, pf)


def col(c):
    return np.nan_to_num(S.num(join, c))


mv = col("mv")
kw_cols = sorted(c for c in join.columns if c.startswith("kw_"))
ph_cols = sorted(c for c in join.columns if c.startswith("ph_"))
fly_cols = ["kw_flying", "kw_flying_anywhere", "ph_flying_grant"]
kw_rest = [c for c in kw_cols if c not in fly_cols]

BLOCKS = [
    ("MV", [("mv", mv), ("mv2", mv**2)]),
    ("+ type class", [(c, col(c)) for c in
                      ("is_creature", "is_instant", "is_sorcery", "is_artifact",
                       "is_enchantment", "is_land", "is_planeswalker", "is_aura",
                       "is_equipment", "is_vehicle", "is_legendary", "is_basic")]),
    ("+ P/T", [("power", col("power")), ("toughness", col("toughness")),
               ("has_pt", np.isfinite(S.num(join, "power")).astype(float))]),
    ("+ flying", [(c, col(c)) for c in fly_cols if c in join.columns]),
    ("+ keyword flags", [(c, col(c)) for c in kw_rest] + [("kw_count", col("kw_count"))]),
    ("+ phrase / archetype flags", [(c, col(c)) for c in ph_cols]),
    ("+ pips", [(c, col(c)) for c in
                ("pip_w", "pip_u", "pip_b", "pip_r", "pip_g", "pip_c", "pips_total",
                 "generic", "n_colors", "is_gold", "hybrid", "has_x",
                 "is_colorless_cost", "phyrexian")]),
    ("+ ability-line count", [(c, col(c)) for c in
                              ("n_abilities", "n_lines", "n_body_lines", "n_words",
                               "n_body_words", "n_sym", "n_types", "big_tribe")]),
    ("+ era / rarity", None),   # handled specially (adds a categorical set code)
]

year = pd.to_numeric(join["first_year"], errors="coerce").to_numpy(float)
year = np.where(np.isfinite(year), year, np.nanmedian(year))
rarity = pd.Categorical(join["first_rarity"].fillna("unknown")).codes.astype(float)
setcode = pd.Categorical(join["first_set_code"].fillna("unknown")).codes.astype(float)
ERA = [("first_year", year), ("first_rarity", rarity), ("first_set_code", setcode)]

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
            b, _, _ = S.wls(Xa, y[m], w[m])
            pred[part] = np.column_stack([np.ones(len(part)), X[part]]) @ b
    return pl._r2(y, pred, w)


rows = []
for tname, (y, w, have, head) in TARGETS.items():
    cols: list[tuple[str, np.ndarray]] = []
    cat: list[int] = []
    for bname, block in BLOCKS:
        if block is None:
            cols += ERA
            cat += [len(cols) - 1, len(cols) - 2]
        else:
            cols += block
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
out["reference"] = {
    "equivalence_class_text_explainable_vs_members": 0.515,
    "honest_val_r2_score_play": pl.load_probes("honest", True).probes["score_play"].metrics["val_r2"],
    "honest_val_r2_played_rate": pl.load_probes("honest", True).probes["played_rate"].metrics["val_r2"],
    "fidelity_insample_r2_score_play": pf.probes["score_play"].metrics["in_sample_r2"],
    "fidelity_insample_r2_played_rate": pf.probes["played_rate"].metrics["in_sample_r2"],
}

# variance budget: how much of the *label* variance the encoder's taste itself carries
for head in ("score_play", "played_rate"):
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    p = np.asarray(pred_f[head], float)
    out[f"budget_{head}"] = {
        "var_label": float(np.average((y[have] - np.average(y[have], weights=w[have]))**2,
                                      weights=w[have])),
        "var_encoder_taste": float(np.average((p[have] - np.average(p[have], weights=w[have]))**2,
                                              weights=w[have])),
    }

with open(S.OUT / "s_r18.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out["reference"], indent=1))
