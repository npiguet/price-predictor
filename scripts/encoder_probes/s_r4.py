"""R4 — cast_lift head anatomy: is it a (score_play, MV) re-encoding?"""

from __future__ import annotations

import json

import numpy as np
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import oof_predict, pearson_fisher_se, wcorr, wls
from sklearn.linear_model import LinearRegression

OTHER_HEADS = ("score_play", "score_draw", "played_rate", "played_rate@logit")


def output_correlations(fr: S.ProbeFrame, cast: pl.Label, sel: np.ndarray,
                        mv: np.ndarray, pips: np.ndarray) -> list[list]:
    """(a) the honest cast_lift prediction and label against the other heads,
    MV and pips, on val cards."""
    pred_h, n = fr.pred_h, int(sel.sum())
    lab_play, lab_rate = fr.label("score_play").y, fr.label("played_rate").y
    pairs = {
        "pred cast_lift ↔ pred score_play": (pred_h["cast_lift"], pred_h["score_play"]),
        "pred cast_lift ↔ pred score_draw": (pred_h["cast_lift"], pred_h["score_draw"]),
        "pred cast_lift ↔ pred played_rate": (pred_h["cast_lift"], pred_h["played_rate"]),
        "pred cast_lift ↔ MV": (pred_h["cast_lift"], mv),
        "pred cast_lift ↔ total pips": (pred_h["cast_lift"], pips),
        "pred cast_lift ↔ label cast_lift": (pred_h["cast_lift"], cast.y),
        "label cast_lift ↔ label score_play": (cast.y, lab_play),
        "label cast_lift ↔ label played_rate": (cast.y, lab_rate),
        "label cast_lift ↔ MV": (cast.y, mv),
        "label cast_lift ↔ total pips": (cast.y, pips),
        "pred score_play ↔ label score_play": (pred_h["score_play"], lab_play),
    }
    rows = []
    for name, (a, b) in pairs.items():
        a = np.asarray(a, float)[sel]
        b = np.asarray(b, float)[sel]
        r_u, r_w = wcorr(a, b), wcorr(a, b, cast.w[sel])
        rows.append([name, n, f"{r_u:+.4f}", f"±{pearson_fisher_se(r_u, n):.4f}",
                     f"{r_w:+.4f}"])
    return rows


def r2_of(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """In-sample and 5-fold-CV weighted R² of the WLS fit."""
    ins = pl._r2(y, X @ wls(X, y, w).beta, w)
    pred = oof_predict(LinearRegression(fit_intercept=False), X, y,
                       S.permutation_folds(len(y)), weight=w)
    return ins, pl._r2(y, pred, w)


def incremental_validity(fr: S.ProbeFrame, cast: pl.Label, sel: np.ndarray,
                         mv: np.ndarray, pips: np.ndarray) -> dict:
    """(c) what the cast_lift head adds over score_play, MV and played_rate,
    in both directions, and how much of its weight vector the other heads span."""
    p = fr.pred_h

    def design(cols):
        return np.column_stack([np.ones(int(sel.sum()))]
                               + [np.asarray(c, float)[sel] for c in cols])

    y, w = cast.y[sel], cast.w[sel]
    core = [p["score_play"], mv, mv**2, p["played_rate"]]
    blocks = {
        "MV, MV²": [mv, mv**2],
        "pred score_play": [p["score_play"]],
        "pred score_play + MV, MV²": [p["score_play"], mv, mv**2],
        "pred score_play + MV, MV² + pred played_rate": core,
        "… + pred cast_lift": core + [p["cast_lift"]],
        "… + pred score_draw too":
            [p["score_play"], p["score_draw"], mv, mv**2, p["played_rate"], p["cast_lift"]],
        "pred cast_lift alone": [p["cast_lift"]],
        "pred cast_lift + MV, MV²": [p["cast_lift"], mv, mv**2],
    }
    out: dict = {}
    rows, prev = [], None
    for name, cols in blocks.items():
        ins, cv = r2_of(design(cols), y, w)
        rows.append([name, len(cols), f"{ins:.4f}", f"{cv:.4f}",
                     "" if prev is None else f"{cv - prev:+.4f}"])
        if name.startswith("pred score_play + MV, MV² + pred played_rate"):
            prev = cv
        if name == "… + pred cast_lift":
            out["c_delta_cv_r2_from_cast_head"] = cv - prev
            out["c_full_cv_r2"] = cv
            prev = cv
    out["c_nested"] = rows

    # residual SD in label-SD units for the two key models
    for name, cols in (("without cast head", core), ("with cast head", core + [p["cast_lift"]])):
        X = design(cols)
        beta = wls(X, y, w).beta
        out[f"c_resid_sd_{name}"] = float(np.sqrt(np.average((y - X @ beta) ** 2, weights=w)))

    # the mirror question: is the *predicted* cast_lift a re-encoding?
    yp = np.asarray(p["cast_lift"], float)[sel]
    rows_m = []
    for name, cols in (("pred score_play", [p["score_play"]]),
                       ("+ MV, MV²", [p["score_play"], mv, mv**2]),
                       ("+ pred played_rate", core),
                       ("+ pred score_draw",
                        [p["score_play"], p["score_draw"], mv, mv**2, p["played_rate"]]),
                       ("+ total pips",
                        [p["score_play"], p["score_draw"], mv, mv**2, p["played_rate"], pips])):
        X = design(cols)
        rows_m.append([name, f"{pl._r2(yp, X @ wls(X, yp, w).beta, w):.4f}"])
    out["c_pred_cast_reencoding"] = rows_m

    # what fraction of the cast head's 512 weights is spanned by the other heads?
    B = np.stack([fr.ph.probes[h].coef for h in OTHER_HEADS], 1)
    c = fr.ph.probes["cast_lift"].coef
    proj = B @ np.linalg.lstsq(B, c, rcond=None)[0]
    out["c_weightvec_span_fraction"] = float((proj @ proj) / (c @ c))
    return out


def main() -> None:
    fr = S.ProbeFrame()
    mv = fr.num("mv", 0)
    pips = fr.num("pips_total", 0)
    cast = fr.label("cast_lift")
    sel = fr.val & cast.have

    out: dict = {"a_n_val": int(sel.sum()),
                 "a_corr": output_correlations(fr, cast, sel, mv, pips)}
    # (b) probe-weight cosines
    out["b_cosines"] = {
        f"{tag} cast_lift·{other}": S.cosine(ps.probes["cast_lift"].coef, ps.probes[other].coef)
        for tag, ps in (("honest", fr.ph), ("fidelity", fr.pf))
        for other in OTHER_HEADS
    }
    out |= incremental_validity(fr, cast, sel, mv, pips)

    write_json(S.OUT / "s_r4.json", out, indent=1)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
