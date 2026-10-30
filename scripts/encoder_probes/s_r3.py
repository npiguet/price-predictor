"""R3 — is there one winnability axis? (play vs draw)."""

from __future__ import annotations

import json

import numpy as np
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import participation_ratio, pearson_fisher_se, wcorr, wls

NOISE_SD = 0.0408  # l_report f: the play-draw difference's noise SD
CORRELATES = {
    "haste": "kw_haste", "sweeper": "ph_sweeper", "counterspell": "ph_counterspell",
    "flash": "kw_flash", "defender": "kw_defender", "card-draw spell": "ph_draw_a_card",
    "unconditional removal": "ph_uncond_removal", "mana rock": "ph_tap_for_mana",
    "kicker": "ph_kicker", "fight": "ph_fight",
}


def output_correlations(fr: S.ProbeFrame, play: pl.Label, draw: pl.Label,
                        w_both: np.ndarray) -> list[list]:
    """(a) play·draw correlation of the labels and of both probe sets' predictions."""
    rows = []
    for tag, mask in (("all cards", np.ones(len(fr), bool)), ("val cards", fr.val),
                      ("train cards", fr.train)):
        have = mask & np.isfinite(play.y) & np.isfinite(draw.y) & (w_both > 0)
        n = int(have.sum())
        for label, series in (
            ("labels", (play.y[have], draw.y[have])),
            ("fidelity preds", (fr.pred_f["score_play"].to_numpy()[have],
                                fr.pred_f["score_draw"].to_numpy()[have])),
            ("honest preds", (fr.pred_h["score_play"].to_numpy()[have],
                              fr.pred_h["score_draw"].to_numpy()[have])),
        ):
            r_u = wcorr(*series)
            r_w = wcorr(*series, w=w_both[have])
            rows.append([tag, label, n, f"{r_u:+.4f}", f"±{pearson_fisher_se(r_u, n):.4f}",
                         f"{r_w:+.4f}"])
    return rows


def difference_spread(fr: S.ProbeFrame, diff_lab: np.ndarray, n_in: np.ndarray) -> dict:
    """SD of the label difference, and of each probe set's predicted difference."""
    out = {}
    for tag, mask in (("all", np.ones(len(fr), bool)), ("val", fr.val)):
        out[f"a_diff_sd_{tag}"] = float(np.std(diff_lab[mask & np.isfinite(diff_lab)]))
    out["a_diff_sd_n200"] = float(np.std(diff_lab[np.isfinite(diff_lab) & (n_in >= 200)]))
    out["a_diff_sd_pred_f"] = float(np.std(
        (fr.pred_f["score_play"] - fr.pred_f["score_draw"]).to_numpy()))
    out["a_diff_sd_pred_h_val"] = float(np.std(
        (fr.pred_h["score_play"] - fr.pred_h["score_draw"]).to_numpy()[fr.val]))
    return out


def probe_geometry(fr: S.ProbeFrame) -> dict:
    """(b) cosines between head probes, the card cloud's spectrum, and the
    play·draw output correlation decomposed across its principal components."""
    pf, ph = fr.pf, fr.ph
    out = {"b_cosines": {
        "fidelity play·draw": S.cosine(pf.probes["score_play"].coef,
                                       pf.probes["score_draw"].coef),
        "honest play·draw": S.cosine(ph.probes["score_play"].coef,
                                     ph.probes["score_draw"].coef),
        "fidelity play·played_rate": S.cosine(pf.probes["score_play"].coef,
                                              pf.probes["played_rate"].coef),
        "fidelity play·cast_lift": S.cosine(pf.probes["score_play"].coef,
                                            pf.probes["cast_lift"].coef),
    }}

    Xc = fr.emb - fr.emb.mean(0)
    _, sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    evals = (sv**2) / (len(Xc) - 1)
    out["b_spectrum"] = {
        "total_var": float(evals.sum()),
        "frac_top1": float(evals[0] / evals.sum()),
        "frac_top5": float(evals[:5].sum() / evals.sum()),
        "frac_top10": float(evals[:10].sum() / evals.sum()),
        "frac_top20": float(evals[:20].sum() / evals.sum()),
        "frac_top50": float(evals[:50].sum() / evals.sum()),
        "participation_ratio": participation_ratio(evals),
    }

    # corr(w_p·x, w_d·x) = Σ_k λ_k a_k b_k / sqrt(Σ λ_k a_k² · Σ λ_k b_k²)
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
    return out


def ridge_on_difference(fr: S.ProbeFrame, diff_lab: np.ndarray, play: pl.Label,
                        w_both: np.ndarray, n_in: np.ndarray) -> dict:
    """(c) can ANY direction predict the label difference? Against a shuffled
    target, and against the same pipeline on score_play itself."""
    emb = fr.emb
    have = np.isfinite(diff_lab) & (w_both > 0)
    tr, va = have & fr.train, have & fr.val
    rows = []
    for floor, tag in ((0, "all"), (200, "n≥200"), (800, "n≥800")):
        trm, vam = tr & (n_in >= floor), va & (n_in >= floor)
        r2, alpha, coef, b = pl.ridge_fit_eval(
            emb[trm], diff_lab[trm], emb[vam], diff_lab[vam], w_both[trm])
        pred = emb[vam] @ coef + b
        rows.append([tag, int(trm.sum()), int(vam.sum()), alpha, r2,
                     wcorr(pred, diff_lab[vam])])
        y_sh = diff_lab[trm].copy()
        np.random.default_rng(0).shuffle(y_sh)
        coef2, b2 = pl._ridge_solve(emb[trm], y_sh, w_both[trm], [alpha])[alpha]
        rows.append([tag + " (shuffled target)", int(trm.sum()), int(vam.sum()), alpha,
                     pl._r2(diff_lab[vam], emb[vam] @ coef2 + b2),
                     wcorr(emb[vam] @ coef2 + b2, diff_lab[vam])])
    r2p, _, _, _ = pl.ridge_fit_eval(emb[tr], play.y[tr], emb[va], play.y[va], play.w[tr])
    return {"c_ridge_on_difference": rows, "c_reference_score_play_val_r2": r2p}


def difference_correlates(fr: S.ProbeFrame, diff_lab: np.ndarray, w_both: np.ndarray,
                          n_in: np.ndarray) -> dict:
    """(d) are the label difference's correlates visible in the predictions?"""
    base = S.base_controls(fr.join)
    targets = {
        "label diff": diff_lab,
        "fidelity pred diff": (fr.pred_f["score_play"] - fr.pred_f["score_draw"]).to_numpy(),
        "honest pred diff": (fr.pred_h["score_play"] - fr.pred_h["score_draw"]).to_numpy(),
    }
    sel = np.isfinite(diff_lab) & (n_in >= 200)
    rows = []
    for name, col in CORRELATES.items():
        fv = fr.num(col, 0)
        row = [name, int(fv[sel].sum())]
        for y in targets.values():
            m = sel & np.isfinite(y)
            fit = wls(np.column_stack([base[m], fv[m]]), y[m], w_both[m])
            row += [f"{fit.beta[-1]:+.4f}", f"{fit.se[-1]:.4f}", f"{fit.t[-1]:+.1f}",
                    f"{fit.beta[-1] / NOISE_SD:+.3f}"]
        rows.append(row)
    out = {"d_correlates": rows}

    # how well does the predicted difference track the label difference at all?
    for tag in ("fidelity", "honest"):
        y = targets[f"{tag} pred diff"]
        out[f"d_corr_preddiff_labeldiff_{tag}_n200"] = wcorr(y[sel], diff_lab[sel], w_both[sel])
        sv = sel & fr.val
        out[f"d_corr_preddiff_labeldiff_{tag}_val_n200"] = wcorr(y[sv], diff_lab[sv], w_both[sv])
    return out


def main() -> None:
    fr = S.ProbeFrame()
    play, draw = fr.label("score_play"), fr.label("score_draw")
    w_both = np.minimum(play.w, draw.w)
    n_in = fr.num("n_in_deck")
    diff_lab = play.y - draw.y

    out: dict = {"a_corr": output_correlations(fr, play, draw, w_both)}
    out |= difference_spread(fr, diff_lab, n_in)
    out |= probe_geometry(fr)
    out |= ridge_on_difference(fr, diff_lab, play, w_both, n_in)
    out |= difference_correlates(fr, diff_lab, w_both, n_in)

    write_json(S.OUT / "s_r3.json", out, indent=1)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
