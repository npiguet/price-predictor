"""R5 — what the five color_lift heads actually taught the encoder."""

from __future__ import annotations

import json

import numpy as np
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import auc, pearson_fisher_se, wcorr, wls
from sklearn.linear_model import LogisticRegression


def colour_labels(fr: S.ProbeFrame) -> list[pl.Label]:
    return [fr.label(f"color_lift_{c}") for c in S.COLORS]


def ridge_vs_gbm(fr: S.ProbeFrame) -> dict:
    """(a) the ridge ceiling of each colour head against a nonlinear upper
    bound, and the same GBM on score_play and played_rate for reference."""
    emb = fr.emb
    rows = []
    for c, lab in zip(S.COLORS, colour_labels(fr)):
        tr, va = lab.have & fr.train, lab.have & fr.val
        r2_ridge, alpha, _, _ = pl.ridge_fit_eval(emb[tr], lab.y[tr], emb[va], lab.y[va],
                                                  lab.w[tr])
        g = S.gbm(400, 0.06, 0.15)
        g.fit(emb[tr], lab.y[tr], sample_weight=lab.w[tr])
        r2_gb = pl._r2(lab.y[va], g.predict(emb[va]), lab.w[va])
        rows.append([c, int(tr.sum()), int(va.sum()), alpha,
                     f"{r2_ridge:.4f}", f"{r2_gb:.4f}", f"{r2_gb - r2_ridge:+.4f}",
                     g.n_iter_])
    ref = []
    for head in ("score_play", "played_rate"):
        lab = fr.label(head)
        tr, va = lab.have & fr.train, lab.have & fr.val
        r2_ridge, _, _, _ = pl.ridge_fit_eval(emb[tr], lab.y[tr], emb[va], lab.y[va],
                                              lab.w[tr])
        g = S.gbm(400, 0.06, 0.15)
        g.fit(emb[tr], lab.y[tr], sample_weight=lab.w[tr])
        ref.append([head, f"{r2_ridge:.4f}",
                    f"{pl._r2(lab.y[va], g.predict(emb[va]), lab.w[va]):.4f}"])
    return {"a_ridge_vs_gbm": rows, "a_reference_heads": ref}


def off_diagonal(fr: S.ProbeFrame, own, lab, prd, pips, n_colors) -> dict:
    """(b) off-diagonal magnitude: is it quality, or pips?"""
    off = ~own
    n_off = off.sum(1)
    with np.errstate(invalid="ignore"):
        mean_off_pred = np.where(
            n_off > 0, np.nansum(np.where(off, prd, 0), 1) / np.maximum(n_off, 1), np.nan)
        lab_off = np.where(off, lab, np.nan)
        mean_off_lab = np.nanmean(lab_off, 1)
    sel = np.isfinite(mean_off_lab) & np.isfinite(mean_off_pred)
    rows = []
    for tag, y in (("predicted mean off-colour lift", mean_off_pred),
                   ("label mean off-colour lift", mean_off_lab)):
        for xname, x in (("pred score_play", np.asarray(fr.pred_h["score_play"], float)),
                         ("label score_play", fr.label("score_play").y),
                         ("pred played_rate", np.asarray(fr.pred_h["played_rate"], float)),
                         ("label played_rate", fr.label("played_rate").y),
                         ("total pips", pips),
                         ("n_colors", n_colors),
                         ("MV", fr.num("mv", 0))):
            m = sel & np.isfinite(x)
            r = wcorr(y[m], x[m])
            rows.append([tag, xname, int(m.sum()), f"{r:+.4f}",
                         f"±{pearson_fisher_se(r, int(m.sum())):.4f}"])
    return {"b_offdiag_correlates": rows,
            "b_frac_negative_pred": float(np.nanmean(np.where(off, prd, np.nan) < 0)),
            "b_frac_negative_lab": float(np.nanmean(lab_off < 0)),
            "b_mean_off_pred": float(np.nanmean(mean_off_pred)),
            "b_mean_off_lab": float(np.nanmean(mean_off_lab))}


def allied_enemy(mat: np.ndarray, own: np.ndarray, mono: np.ndarray):
    """Doubly-centre the off-diagonal cells of mono-colour cards, then split."""
    M = np.where(~own, mat, np.nan)[mono]
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


def identity_share(fr: S.ProbeFrame, own, labs, pips, n_colors) -> dict:
    """(d) how much of the colour heads is colour identity?"""
    emb, train, val = fr.emb, fr.train, fr.val
    # d1: can the embedding read colour identity off the text?
    rows_d1 = []
    for i, c in enumerate(S.COLORS):
        y = own[:, i].astype(int)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(emb[train], y[train])
        p = clf.predict_proba(emb[val])[:, 1]
        acc = float(((p > 0.5).astype(int) == y[val]).mean())
        rows_d1.append([c, int(y[val].sum()), f"{acc:.4f}", f"{auc(y[val] == 1, p):.4f}"])

    # d2: R² of colour_lift from colour identity alone vs from the embedding
    rows_d2 = []
    ident = np.column_stack([own.astype(float), n_colors, pips, fr.num("mv", 0)])
    X_id = np.column_stack([np.ones(len(fr)), own.astype(float)])
    X_id2 = np.column_stack([np.ones(len(fr)), ident])
    for c, lab in zip(S.COLORS, labs):
        y, w = lab.y, lab.w
        tr, va = lab.have & train, lab.have & val
        r2_id = pl._r2(y[va], X_id[va] @ wls(X_id[tr], y[tr], w[tr]).beta, w[va])
        b2 = wls(X_id2[tr], y[tr], w[tr]).beta
        r2_id2 = pl._r2(y[va], X_id2[va] @ b2, w[va])
        # embedding ridge, and embedding ridge on the identity-residual
        r2_emb, _, _, _ = pl.ridge_fit_eval(emb[tr], y[tr], emb[va], y[va], w[tr])
        resid = y - X_id2 @ b2
        r2_res, _, _, _ = pl.ridge_fit_eval(emb[tr], resid[tr], emb[va], resid[va], w[tr])
        rows_d2.append([c, f"{r2_id:.4f}", f"{r2_id2:.4f}", f"{r2_emb:.4f}", f"{r2_res:.4f}"])
    return {"d_identity_readoff": rows_d1, "d_identity_share": rows_d2}


def main() -> None:
    fr = S.ProbeFrame()
    own = S.card_colors(fr.join)                    # (N,5) bool: pip of that colour
    pips = fr.num("pips_total", 0)
    n_colors = fr.num("n_colors", 0)
    labs = colour_labels(fr)
    lab = np.column_stack([lb.y for lb in labs])
    prd = np.column_stack([np.asarray(fr.pred_h[f"color_lift_{c}"], float) for c in S.COLORS])

    out = ridge_vs_gbm(fr)
    out |= off_diagonal(fr, own, lab, prd, pips, n_colors)

    # (c) allied vs enemy on the doubly-centred off-diagonal residual
    mono = own.sum(1) == 1
    rows_c = []
    for tag, mat in (("predicted", prd), ("label", lab)):
        na, ne, ma, me, diff, se = allied_enemy(mat, own, mono)
        rows_c.append([tag, na, ne, f"{ma:+.5f}", f"{me:+.5f}", f"{diff:+.5f}",
                       f"{se:.5f}", f"{diff / se:+.1f}", f"{diff / pl.SD['color_lift']:+.3f}"])
    out["c_allied_enemy"] = rows_c
    out["c_n_mono"] = int(mono.sum())

    out |= identity_share(fr, own, labs, pips, n_colors)

    write_json(S.OUT / "s_r5.json", out, indent=1)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
