"""R17 follow-up — the MLM-neighbourhood test, purged of its mechanical term.

Regressing (pred − y) on (nb_mean − y) is biased: both sides carry −y, so the
slope is positive even for a prediction that ignores the neighbourhood. The
honest question is whether ``nb_mean`` adds anything to ``pred`` **once the
card's own label is controlled for**, so that is what this fits.
"""

from __future__ import annotations

import json

import numpy as np

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val = d["val"]
pf, ph = d["pf"], d["ph"]
pred_f = pl.predict_labels(emb, pf)
pred_h = pl.predict_labels(emb, ph)
nb_idx = np.load(S.OUT / "s_r17_neighbours.npy")
n_in = S.num(join, "n_in_deck")

Xn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
nb_cos = np.array([float(Xn[i] @ Xn[nb_idx[i]].mean(0)) for i in range(len(Xn))])

flag_cols = [c for c in join.columns if c.startswith(("kw_", "ph_"))]
rare_masks = {c: (np.nan_to_num(S.num(join, c)) > 0) for c in flag_cols}
rare_masks = {c: v for c, v in rare_masks.items() if 20 <= v.sum() < 200}
rare_any = np.any(np.stack(list(rare_masks.values())), 0)

out: dict = {"n_rare_cards": int(rare_any.sum()),
             "mean_cosine_to_neighbour_centroid": float(nb_cos.mean()),
             "p10_cosine": float(np.quantile(nb_cos, 0.1))}

rows = []
for head in ("score_play", "played_rate"):
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    nb_mean = np.array([np.nanmean(y[nb_idx[i]]) for i in range(len(y))])
    for tag, p in (("fidelity/all", np.asarray(pred_f[head], float)),
                   ("honest/val", np.asarray(pred_h[head], float))):
        for sub_name, sub in (("all cards", np.ones(len(join), bool)),
                              ("rare-mechanic cards", rare_any),
                              ("n_in_deck < 200", n_in < 200),
                              ("n_in_deck ≥ 2000", n_in >= 2000)):
            m = have & np.isfinite(nb_mean) & sub
            if tag == "honest/val":
                m = m & val
            if m.sum() < 60:
                continue
            one = np.ones(int(m.sum()))
            XA = np.column_stack([one, y[m]])
            XB = np.column_stack([one, y[m], nb_mean[m]])
            bA, _, _ = S.wls(XA, p[m], w[m])
            bB, seB, tB = S.wls(XB, p[m], w[m])
            r2A = pl._r2(p[m], XA @ bA, w[m])
            r2B = pl._r2(p[m], XB @ bB, w[m])
            rows.append([head, tag, sub_name, int(m.sum()),
                         f"{bA[1]:+.3f}", f"{bB[1]:+.3f}", f"{bB[2]:+.3f}",
                         f"{seB[2]:.3f}", f"{tB[2]:+.1f}",
                         f"{r2A:.4f}", f"{r2B:.4f}", f"{r2B - r2A:+.4f}"])
out["neighbourhood_controlled"] = rows

# the naive (biased) slope and its mechanical null, for the record
rows_null = []
for head in ("score_play", "played_rate"):
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    nb_mean = np.array([np.nanmean(y[nb_idx[i]]) for i in range(len(y))])
    for tag, p in (("fidelity/all", np.asarray(pred_f[head], float)),
                   ("honest/val", np.asarray(pred_h[head], float))):
        m = have & np.isfinite(nb_mean) & (val if tag == "honest/val"
                                           else np.ones(len(join), bool))
        pull = nb_mean[m] - y[m]
        # observed
        X = np.column_stack([np.ones(int(m.sum())), pull])
        b, se, t = S.wls(X, p[m] - y[m], w[m])
        # mechanical null: replace the prediction with the corpus mean
        mu = np.average(y[m], weights=w[m])
        b0, _, _ = S.wls(X, mu - y[m], w[m])
        # and with a label-preserving permutation of the prediction
        rng = np.random.default_rng(0)
        ps = p[m].copy()
        rng.shuffle(ps)
        b1, _, _ = S.wls(X, ps - y[m], w[m])
        rows_null.append([head, tag, int(m.sum()), f"{b[1]:+.3f}",
                          f"{b0[1]:+.3f}", f"{b1[1]:+.3f}"])
out["naive_slope_vs_null"] = rows_null

with open(S.OUT / "s_r17b.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
