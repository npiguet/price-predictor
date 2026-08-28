"""R8 — played_rate, the encoder's loudest axis: cost, hostility, content."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val, train = d["val"], d["train"]
pf, ph = d["pf"], d["ph"]
pred_f = pl.predict_labels(emb, pf)
pred_h = pl.predict_labels(emb, ph)

mv = np.nan_to_num(S.num(join, "mv"))
y = S.num(join, "shrunk_played_rate")
w = S.num(join, "w_played_rate")
have = np.isfinite(y) & (w > 0)
tr, va = have & train, have & val

out: dict = {}

# ── (a) how much of played_rate is cost? ────────────────────────────────
def col(c):
    return np.nan_to_num(S.num(join, c))


cost = [mv, mv**2, col("pips_total"), col("generic"), col("n_colors"),
        col("is_colorless_cost"), col("has_x"), col("hybrid")]
types = [col(c) for c in ("is_creature", "is_instant", "is_sorcery", "is_artifact",
                          "is_enchantment", "is_land", "is_planeswalker", "is_aura",
                          "is_equipment", "is_vehicle", "is_legendary")]
pt = [col("power"), col("toughness"), np.isfinite(S.num(join, "power")).astype(float)]
kws = [col(c) for c in join.columns if c.startswith("kw_")]
phs = [col(c) for c in join.columns if c.startswith("ph_")]

rows_a = []
for name, cols in (("MV, MV² only", [mv, mv**2]),
                   ("cost line (MV, MV², pips, generic, colours, X, hybrid)", cost),
                   ("cost + type flags", cost + types),
                   ("cost + type + P/T", cost + types + pt),
                   ("cost + type + P/T + keywords", cost + types + pt + kws),
                   ("cost + type + P/T + keywords + phrases",
                    cost + types + pt + kws + phs)):
    X = np.column_stack([np.ones(len(join))] + cols)
    b, _, _ = S.wls(X[tr], y[tr], w[tr])
    rows_a.append([name, len(cols), f"{pl._r2(y[tr], X[tr] @ b, w[tr]):.4f}",
                   f"{pl._r2(y[va], X[va] @ b, w[va]):.4f}"])
rows_a.append(["512-dim embedding, honest ridge probe", 512, "0.9009",
               f"{ph.probes['played_rate'].metrics['val_r2']:.4f}"])
out["a_nested_r2"] = rows_a

# ── (b) the AI-hostility classes ────────────────────────────────────────
noncre_artifact = (col("is_artifact") > 0) & (col("is_creature") == 0)
classes = {
    "fog / prevent damage": (col("ph_fog") + col("ph_prevent_damage")) > 0,
    "sweeper": col("ph_sweeper") > 0,
    "counterspell": col("ph_counterspell") > 0,
    "morph": col("ph_morph") > 0,
    "mana rock": noncre_artifact & (col("ph_tap_for_mana") > 0),
}
type_cell = np.select(
    [col("is_creature") > 0, col("is_land") > 0, col("is_instant") > 0,
     col("is_sorcery") > 0, col("is_artifact") > 0, col("is_enchantment") > 0],
    [0, 1, 2, 3, 4, 5], default=6)
mv_cell = np.clip(mv, 0, 8).astype(int)
cell = mv_cell * 10 + type_cell


def matched_deficit(flag, values, wts):
    """Cell-weighted (MV × broad type) mean difference, class − control."""
    tot_w = 0.0
    tot = 0.0
    per_cell = 0
    for cval in np.unique(cell):
        m = (cell == cval) & np.isfinite(values) & (wts > 0)
        a, b = m & flag, m & ~flag
        na, nb = int(a.sum()), int(b.sum())
        if na == 0 or nb < 5:
            continue
        wa = min(na, nb)
        tot += wa * (np.average(values[a], weights=wts[a])
                     - np.average(values[b], weights=wts[b]))
        tot_w += wa
        per_cell += 1
    return (tot / tot_w if tot_w else np.nan), per_cell


targets_b = {
    "label played_rate": (y, w),
    "fidelity pred played_rate": (np.asarray(pred_f["played_rate"], float), w),
    "honest pred played_rate": (np.asarray(pred_h["played_rate"], float), w),
}
rows_b = []
for name, flag in classes.items():
    row = [name, int(flag.sum())]
    for tname, (vals, wts) in targets_b.items():
        dfc, ncell = matched_deficit(flag, vals, wts)
        row += [f"{dfc:+.4f}", f"{dfc / S.SD['played_rate']:+.2f}"]
    rows_b.append(row)
out["b_matched_deficits"] = rows_b

# one shared direction?
dirs = {}
for name, flag in classes.items():
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    clf.fit(emb, flag.astype(int))
    v = clf.coef_[0]
    dirs[name] = v / np.linalg.norm(v)
names = list(dirs)
D = np.stack([dirs[n] for n in names])
out["b_direction_cosines"] = [
    [names[i]] + [f"{float(D[i] @ D[j]):+.3f}" for j in range(len(names))]
    for i in range(len(names))
]
out["b_direction_names"] = names
u_svd, s_svd, vt_svd = np.linalg.svd(D, full_matrices=False)
out["b_shared_component_var_share"] = [float(x) for x in (s_svd**2 / (s_svd**2).sum())]
u = vt_svd[0]
out["b_cos_shared_vs_played_probe"] = S.cosine(u, ph.probes["played_rate"].coef)
out["b_cos_shared_vs_score_probe"] = S.cosine(u, ph.probes["score_play"].coef)
out["b_loadings_on_shared"] = {n: float(dirs[n] @ u) for n in names}

# deflate the shared direction out of the embeddings and re-read the probe
emb_def = emb - np.outer(emb @ u, u)
probe = ph.probes["played_rate"]
pred_def = emb_def @ probe.coef + probe.intercept
pred_raw = emb @ probe.coef + probe.intercept
# also deflate the single best-aligned direction: the played_rate probe itself
pc_pr = probe.coef / np.linalg.norm(probe.coef)
emb_def2 = emb - np.outer(emb @ pc_pr, pc_pr)
pred_def2 = emb_def2 @ probe.coef + probe.intercept
rows_b2 = []
for name, flag in classes.items():
    base, _ = matched_deficit(flag, pred_raw, w)
    defl, _ = matched_deficit(flag, pred_def, w)
    rows_b2.append([name, f"{base:+.4f}", f"{defl:+.4f}",
                    f"{(1 - defl / base) * 100:+.0f}%" if base else "—"])
out["b_deflation"] = rows_b2
out["b_pred_sd_before_after"] = [float(pred_raw.std()), float(pred_def.std()),
                                 float(pred_def2.std())]

# ── (c) what loads on the played_rate direction ─────────────────────────
flag_cols = [c for c in join.columns
             if c.startswith(("kw_", "ph_", "is_")) and c != "is_primary"]
extra_cols = ["big_tribe", "has_x", "hybrid", "phyrexian", "is_colorless_cost"]
flag_cols = sorted(set(flag_cols) | set(extra_cols))
base = np.column_stack([np.ones(len(join)), mv, mv**2, col("is_creature")])
rows_c = []
for c in flag_cols:
    fv = col(c)
    if fv.min() == fv.max() or ((fv > 0).sum() < 40):
        continue
    fv = (fv > 0).astype(float)
    for tname, vals in (("pred", np.asarray(pred_f["played_rate"], float)),
                        ("label", y)):
        X = np.column_stack([base, fv])
        b, se, t = S.wls(X[have], vals[have], w[have])
        if tname == "pred":
            bp, tp = b[-1], t[-1]
        else:
            bl, tl = b[-1], t[-1]
    rows_c.append([c, int(fv.sum()), bp / S.SD["played_rate"], tp,
                   bl / S.SD["played_rate"], tl])
rc = pd.DataFrame(rows_c, columns=["feature", "n", "pred_sd", "pred_t", "label_sd", "label_t"])
rc = rc.sort_values("pred_sd")
out["c_bottom"] = [[r.feature, int(r.n), f"{r.pred_sd:+.3f}", f"{r.pred_t:+.1f}",
                    f"{r.label_sd:+.3f}", f"{r.label_t:+.1f}"]
                   for r in rc.head(18).itertuples()]
out["c_top"] = [[r.feature, int(r.n), f"{r.pred_sd:+.3f}", f"{r.pred_t:+.1f}",
                 f"{r.label_sd:+.3f}", f"{r.label_t:+.1f}"]
                for r in rc.tail(18).iloc[::-1].itertuples()]
out["c_corr_pred_label_effects"] = float(np.corrcoef(rc.pred_sd, rc.label_sd)[0, 1])
out["c_n_features"] = int(len(rc))

with open(S.OUT / "s_r8.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
