"""R3 — is there one winnability axis? (play vs draw)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val, train = d["val"], d["train"]
pf, ph = d["pf"], d["ph"]

y_play = S.num(join, "shrunk_score_play")
y_draw = S.num(join, "shrunk_score_draw")
w_play = S.num(join, "w_score_play")
w_draw = S.num(join, "w_score_draw")
w_both = np.minimum(w_play, w_draw)
n_in = S.num(join, "n_in_deck")

pred_f = pl.predict_labels(emb, pf)
pred_h = pl.predict_labels(emb, ph)

out: dict = {}

# ── (a) output correlations vs label correlation ────────────────────────
rows = []
for tag, mask in (("all cards", np.ones(len(join), bool)), ("val cards", val),
                  ("train cards", train)):
    have = mask & np.isfinite(y_play) & np.isfinite(y_draw) & (w_both > 0)
    n = int(have.sum())
    for label, series in (
        ("labels", (y_play[have], y_draw[have])),
        ("fidelity preds", (pred_f["score_play"].to_numpy()[have],
                            pred_f["score_draw"].to_numpy()[have])),
        ("honest preds", (pred_h["score_play"].to_numpy()[have],
                          pred_h["score_draw"].to_numpy()[have])),
    ):
        r_u = S.wcorr(*series)
        r_w = S.wcorr(*series, w=w_both[have])
        rows.append([tag, label, n, f"{r_u:+.4f}", f"±{S.corr_se(r_u, n):.4f}", f"{r_w:+.4f}"])
out["a_corr"] = rows

# label correlation of the *difference* with each level
diff_lab = y_play - y_draw
for tag, mask in (("all", np.ones(len(join), bool)), ("val", val)):
    have = mask & np.isfinite(diff_lab)
    out[f"a_diff_sd_{tag}"] = float(np.std(diff_lab[have]))
have200 = np.isfinite(diff_lab) & (n_in >= 200)
out["a_diff_sd_n200"] = float(np.std(diff_lab[have200]))
out["a_diff_sd_pred_f"] = float(np.std(
    (pred_f["score_play"] - pred_f["score_draw"]).to_numpy()))
out["a_diff_sd_pred_h_val"] = float(np.std(
    (pred_h["score_play"] - pred_h["score_draw"]).to_numpy()[val]))

# ── (b) probe-weight geometry ───────────────────────────────────────────
cos = {
    "fidelity play·draw": S.cosine(pf.probes["score_play"].coef, pf.probes["score_draw"].coef),
    "honest play·draw": S.cosine(ph.probes["score_play"].coef, ph.probes["score_draw"].coef),
    "fidelity play·played_rate": S.cosine(pf.probes["score_play"].coef,
                                          pf.probes["played_rate"].coef),
    "fidelity play·cast_lift": S.cosine(pf.probes["score_play"].coef,
                                        pf.probes["cast_lift"].coef),
}
out["b_cosines"] = cos

# PCA of the card cloud (primary cards)
Xc = emb - emb.mean(0)
U, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
evals = (sv**2) / (len(Xc) - 1)
out["b_spectrum"] = {
    "total_var": float(evals.sum()),
    "frac_top1": float(evals[0] / evals.sum()),
    "frac_top5": float(evals[:5].sum() / evals.sum()),
    "frac_top10": float(evals[:10].sum() / evals.sum()),
    "frac_top20": float(evals[:20].sum() / evals.sum()),
    "frac_top50": float(evals[:50].sum() / evals.sum()),
    "participation_ratio": float(evals.sum() ** 2 / (evals**2).sum()),
}

# Decompose the output correlation across PCs:
#   corr(w_p·x, w_d·x) = Σ_k λ_k a_k b_k / sqrt(Σ λ_k a_k² · Σ λ_k b_k²)
for tag, ps in (("fidelity", pf), ("honest", ph)):
    a = Vt @ ps.probes["score_play"].coef
    b = Vt @ ps.probes["score_draw"].coef
    ca, cb = evals * a * a, evals * b * b
    cab = evals * a * b
    denom = np.sqrt(ca.sum() * cb.sum())
    order = np.argsort(-np.abs(cab))
    out[f"b_pc_{tag}"] = {
        "r_from_spectrum": float(cab.sum() / denom),
        "cos_raw": float(a @ b / np.sqrt((a @ a) * (b @ b))),
        "top_pc_rows": [
            [int(k), float(evals[k] / evals.sum()), float(a[k]), float(b[k]),
             float(cab[k] / denom), float(ca[k] / ca.sum()), float(cb[k] / cb.sum())]
            for k in order[:8]
        ],
        "cum_r_top1": float(cab[order[:1]].sum() / denom),
        "cum_r_top3": float(cab[order[:3]].sum() / denom),
        "cum_r_top10": float(cab[order[:10]].sum() / denom),
        "var_share_play_top1": float(ca.max() / ca.sum()),
        "var_share_draw_top1": float(cb.max() / cb.sum()),
        "output_sd_play": float(np.sqrt(ca.sum())),
        "output_sd_draw": float(np.sqrt(cb.sum())),
    }

# ── (c) can ANY direction predict the label difference? ─────────────────
have = np.isfinite(diff_lab) & (w_both > 0)
tr = have & train
va = have & val
res_c = []
for floor, tag in ((0, "all"), (200, "n≥200"), (800, "n≥800")):
    trm, vam = tr & (n_in >= floor), va & (n_in >= floor)
    r2, alpha, coef, b = S.ridge_fit_eval(
        emb[trm], diff_lab[trm], emb[vam], diff_lab[vam], w_both[trm])
    pred = emb[vam] @ coef + b
    res_c.append([tag, int(trm.sum()), int(vam.sum()), alpha, r2,
                  S.wcorr(pred, diff_lab[vam])])
    # placebo: same fit on a shuffled target
    rng = np.random.default_rng(0)
    y_sh = diff_lab[trm].copy()
    rng.shuffle(y_sh)
    coef2, b2 = pl._ridge_solve(emb[trm], y_sh, w_both[trm], [alpha])[alpha]
    res_c.append([tag + " (shuffled target)", int(trm.sum()), int(vam.sum()), alpha,
                  pl._r2(diff_lab[vam], emb[vam] @ coef2 + b2),
                  S.wcorr(emb[vam] @ coef2 + b2, diff_lab[vam])])
out["c_ridge_on_difference"] = res_c

# reference: the same honest pipeline on score_play itself
r2p, ap, _, _ = S.ridge_fit_eval(emb[tr], y_play[tr], emb[va], y_play[va], w_play[tr])
out["c_reference_score_play_val_r2"] = r2p

# ── (d) are the label difference's correlates visible in predictions? ───
flags = {
    "haste": join["kw_haste"], "sweeper": join["ph_sweeper"],
    "counterspell": join["ph_counterspell"], "flash": join["kw_flash"],
    "defender": join["kw_defender"], "card-draw spell": join["ph_draw_a_card"],
    "unconditional removal": join["ph_uncond_removal"], "mana rock": join["ph_tap_for_mana"],
    "kicker": join["ph_kicker"], "fight": join["ph_fight"],
}
mv = S.num(join, "mv")
mv = np.where(np.isfinite(mv), mv, 0.0)
creature = np.nan_to_num(S.num(join, "is_creature"))
base = np.column_stack([np.ones(len(join)), mv, mv**2, creature])

targets = {
    "label diff": diff_lab,
    "fidelity pred diff": (pred_f["score_play"] - pred_f["score_draw"]).to_numpy(),
    "honest pred diff": (pred_h["score_play"] - pred_h["score_draw"]).to_numpy(),
}
sel = np.isfinite(diff_lab) & (n_in >= 200)
rows_d = []
noise_sd = 0.0408  # l_report f
for name, f in flags.items():
    fv = S.num(join, f.name) if hasattr(f, "name") else np.asarray(f, float)
    fv = np.where(np.isfinite(fv), fv, 0.0)
    row = [name, int(fv[sel].sum())]
    for tname, y in targets.items():
        m = sel & np.isfinite(y)
        X = np.column_stack([base[m], fv[m]])
        beta, se, t = S.wls(X, y[m], w_both[m])
        row += [f"{beta[-1]:+.4f}", f"{se[-1]:.4f}", f"{t[-1]:+.1f}",
                f"{beta[-1] / noise_sd:+.3f}"]
    rows_d.append(row)
out["d_correlates"] = rows_d

# how well does the predicted difference track the label difference at all?
for tag, y in (("fidelity", targets["fidelity pred diff"]),
               ("honest", targets["honest pred diff"])):
    m = np.isfinite(diff_lab) & (n_in >= 200)
    out[f"d_corr_preddiff_labeldiff_{tag}_n200"] = S.wcorr(y[m], diff_lab[m], w_both[m])
    mv_ = m & val
    out[f"d_corr_preddiff_labeldiff_{tag}_val_n200"] = S.wcorr(y[mv_], diff_lab[mv_], w_both[mv_])

with open(S.OUT / "s_r3.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
