"""R5 — what the five color_lift heads actually taught the encoder."""

from __future__ import annotations

import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val, train = d["val"], d["train"]
ph = d["ph"]
pred_h = pl.predict_labels(emb, ph)

own = S.card_colors(join)                       # (N,5) bool: pip of that colour
pips = np.nan_to_num(S.num(join, "pips_total"))
n_colors = np.nan_to_num(S.num(join, "n_colors"))
lab = np.column_stack([S.num(join, f"shrunk_color_lift_{c}") for c in S.COLORS])
prd = np.column_stack([np.asarray(pred_h[f"color_lift_{c}"], float) for c in S.COLORS])
wcl = np.column_stack([S.num(join, f"w_color_lift_{c}") for c in S.COLORS])
lab_play = S.num(join, "shrunk_score_play")

out: dict = {}

# ── (a) ridge ceiling vs a nonlinear upper bound ────────────────────────
rows_a = []
for i, c in enumerate(S.COLORS):
    y, w = lab[:, i], wcl[:, i]
    have = np.isfinite(y) & (w > 0)
    tr, va = have & train, have & val
    r2_ridge, alpha, _, _ = S.ridge_fit_eval(emb[tr], y[tr], emb[va], y[va], w[tr])
    g = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        early_stopping=True, validation_fraction=0.15, random_state=42)
    g.fit(emb[tr], y[tr], sample_weight=w[tr])
    r2_gb = pl._r2(y[va], g.predict(emb[va]), w[va])
    rows_a.append([c, int(tr.sum()), int(va.sum()), alpha,
                   f"{r2_ridge:.4f}", f"{r2_gb:.4f}", f"{r2_gb - r2_ridge:+.4f}",
                   g.n_iter_])
out["a_ridge_vs_gbm"] = rows_a

# reference: the same GBM on score_play and played_rate
rows_a2 = []
for head in ("score_play", "played_rate"):
    y, w = S.num(join, f"shrunk_{head}"), S.num(join, f"w_{head}")
    have = np.isfinite(y) & (w > 0)
    tr, va = have & train, have & val
    r2_ridge, alpha, _, _ = S.ridge_fit_eval(emb[tr], y[tr], emb[va], y[va], w[tr])
    g = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, early_stopping=True,
        validation_fraction=0.15, random_state=42)
    g.fit(emb[tr], y[tr], sample_weight=w[tr])
    rows_a2.append([head, f"{r2_ridge:.4f}",
                    f"{pl._r2(y[va], g.predict(emb[va]), w[va]):.4f}"])
out["a_reference_heads"] = rows_a2

# ── (b) off-diagonal magnitude: quality or pips? ────────────────────────
off = ~own
n_off = off.sum(1)
with np.errstate(invalid="ignore"):
    mean_off_pred = np.where(n_off > 0, np.nansum(np.where(off, prd, 0), 1) / np.maximum(n_off, 1), np.nan)
    lab_off = np.where(off, lab, np.nan)
    mean_off_lab = np.nanmean(lab_off, 1)
sel = np.isfinite(mean_off_lab) & np.isfinite(mean_off_pred)
rows_b = []
for tag, y in (("predicted mean off-colour lift", mean_off_pred),
               ("label mean off-colour lift", mean_off_lab)):
    for xname, x in (("pred score_play", np.asarray(pred_h["score_play"], float)),
                     ("label score_play", lab_play),
                     ("pred played_rate", np.asarray(pred_h["played_rate"], float)),
                     ("label played_rate", S.num(join, "shrunk_played_rate")),
                     ("total pips", pips),
                     ("n_colors", n_colors),
                     ("MV", np.nan_to_num(S.num(join, "mv")))):
        m = sel & np.isfinite(x)
        r = S.wcorr(y[m], x[m])
        rows_b.append([tag, xname, int(m.sum()), f"{r:+.4f}", f"±{S.corr_se(r, int(m.sum())):.4f}"])
out["b_offdiag_correlates"] = rows_b
out["b_frac_negative_pred"] = float(np.nanmean(np.where(off, prd, np.nan) < 0))
out["b_frac_negative_lab"] = float(np.nanmean(lab_off < 0))
out["b_mean_off_pred"] = float(np.nanmean(mean_off_pred))
out["b_mean_off_lab"] = float(np.nanmean(mean_off_lab))

# ── (c) allied vs enemy on the doubly-centred off-diagonal residual ─────
mono = own.sum(1) == 1
def allied_enemy(mat):
    """Doubly-centre the off-diagonal cells of mono-colour cards, then split."""
    M = np.where(off, mat, np.nan)[mono]
    M = M - np.nanmean(M, 1, keepdims=True)          # per-card centring
    M = M - np.nanmean(M, 0, keepdims=True)          # per-column centring
    ocol = np.argmax(own[mono], 1)
    a_vals, e_vals = [], []
    for r in range(M.shape[0]):
        src = S.COLORS[ocol[r]]
        for k, c in enumerate(S.COLORS):
            v = M[r, k]
            if not np.isfinite(v):
                continue
            pair = frozenset((src, c))
            if pair in S.ALLIED:
                a_vals.append(v)
            elif pair in S.ENEMY:
                e_vals.append(v)
    a, e = np.array(a_vals), np.array(e_vals)
    diff = a.mean() - e.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + e.var(ddof=1) / len(e))
    return len(a), len(e), a.mean(), e.mean(), diff, se


rows_c = []
for tag, mat in (("predicted", prd), ("label", lab)):
    na, ne, ma, me, diff, se = allied_enemy(mat)
    rows_c.append([tag, na, ne, f"{ma:+.5f}", f"{me:+.5f}", f"{diff:+.5f}",
                   f"{se:.5f}", f"{diff / se:+.1f}", f"{diff / S.SD['color_lift']:+.3f}"])
out["c_allied_enemy"] = rows_c
out["c_n_mono"] = int(mono.sum())

# ── (d) how much of the colour heads is colour identity? ────────────────
# d1: can the embedding read colour identity off the text?
rows_d1 = []
for i, c in enumerate(S.COLORS):
    y = own[:, i].astype(int)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(emb[train], y[train])
    p = clf.predict_proba(emb[val])[:, 1]
    acc = float(((p > 0.5).astype(int) == y[val]).mean())
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(len(p))
    pos, neg = y[val] == 1, y[val] == 0
    auc = float((ranks[pos].mean() - (pos.sum() - 1) / 2) / neg.sum())
    rows_d1.append([c, int(y[val].sum()), f"{acc:.4f}", f"{auc:.4f}"])
out["d_identity_readoff"] = rows_d1

# d2: R² of colour_lift from colour identity alone vs from the embedding
rows_d2 = []
ident = np.column_stack([own.astype(float), n_colors, pips, np.nan_to_num(S.num(join, "mv"))])
for i, c in enumerate(S.COLORS):
    y, w = lab[:, i], wcl[:, i]
    have = np.isfinite(y) & (w > 0)
    tr, va = have & train, have & val
    # identity-only (own-colour dummies)
    X_id = np.column_stack([np.ones(len(join)), own.astype(float)])
    b, _, _ = S.wls(X_id[tr], y[tr], w[tr])
    r2_id = pl._r2(y[va], X_id[va] @ b, w[va])
    # identity + colour count/pips/MV
    X_id2 = np.column_stack([np.ones(len(join)), ident])
    b2, _, _ = S.wls(X_id2[tr], y[tr], w[tr])
    r2_id2 = pl._r2(y[va], X_id2[va] @ b2, w[va])
    # embedding ridge, and embedding ridge on the identity-residual
    r2_emb, alpha, coef, b0 = S.ridge_fit_eval(emb[tr], y[tr], emb[va], y[va], w[tr])
    resid = y - X_id2 @ b2
    r2_res, _, _, _ = S.ridge_fit_eval(emb[tr], resid[tr], emb[va], resid[va], w[tr])
    rows_d2.append([c, f"{r2_id:.4f}", f"{r2_id2:.4f}", f"{r2_emb:.4f}", f"{r2_res:.4f}"])
out["d_identity_share"] = rows_d2

with open(S.OUT / "s_r5.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
