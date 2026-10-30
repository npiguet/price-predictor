"""R18 — the nameability curve: how much of the encoder's taste is a feature list.

The curve (``s_r18.json``, ``s_r18_curve.pkl``) is 5-fold CV over every card.
The honest probe's 0.370 is the encoder's own seed-42 val split, so the top of
the curve is refitted on the encoder's train split and scored on its val split,
beside the union model, so the two numbers answer the same question
(``s_r18b.json``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import oof_predict, wls
from sklearn.linear_model import LinearRegression

HEADS = ("score_play", "played_rate")


def cv_r2(X, y, w, cat_idx, model: str) -> float:
    """5-fold CV weighted R² of the gradient-boosted or the linear fit."""
    if model == "gbm":
        est, Xd = S.gbm(250, 0.08, 0.12, cat_idx), X
    else:
        est, Xd = LinearRegression(fit_intercept=False), np.column_stack([np.ones(len(y)), X])
    return pl._r2(y, oof_predict(est, Xd, y, S.permutation_folds(len(y)), weight=w), w)


def weighted_var(v: np.ndarray, w: np.ndarray) -> float:
    return float(np.average((v - np.average(v, weights=w)) ** 2, weights=w))


def nameability_curve(fr: S.ProbeFrame) -> dict:
    """Both CV fits on the nameable blocks added one at a time, for each head's
    label and the encoder's fidelity prediction of it."""
    targets = {}
    for head in HEADS:
        y, w, have = fr.label(head)
        targets[f"{head} — label"] = (y, w, have, head)
        targets[f"{head} — encoder (fidelity pred)"] = (
            np.asarray(fr.pred_f[head], float), w, have, head)

    rows = []
    blocks = S.nameable_blocks(fr.join)
    for tname, (y, w, have, _) in targets.items():
        cols: list[tuple[str, np.ndarray]] = []
        for bname, block in blocks:
            cols += block
            cat = S.categorical_indices(cols)
            X = np.column_stack([c for _, c in cols])[have]
            r_gbm = cv_r2(X, y[have], w[have], cat, model="gbm")
            r_lin = cv_r2(X, y[have], w[have], cat, model="linear")
            rows.append({"target": tname, "block": bname, "k": len(cols),
                         "gbm_cv_r2": r_gbm, "lin_cv_r2": r_lin, "n": int(have.sum())})
            print(f"{tname:42s} {bname:28s} k={len(cols):3d} gbm={r_gbm:.4f} "
                  f"lin={r_lin:.4f}", flush=True)
    curve = pd.DataFrame(rows)
    curve.to_pickle(S.OUT / "s_r18_curve.pkl")

    out = {"curve": curve.to_dict("records")}
    pf, ph = fr.pf, fr.ph
    out["reference"] = {
        "equivalence_class_text_explainable_vs_members": 0.515,
        "honest_val_r2_score_play": ph.probes["score_play"].metrics["val_r2"],
        "honest_val_r2_played_rate": ph.probes["played_rate"].metrics["val_r2"],
        "fidelity_insample_r2_score_play": pf.probes["score_play"].metrics["in_sample_r2"],
        "fidelity_insample_r2_played_rate": pf.probes["played_rate"].metrics["in_sample_r2"],
    }
    # variance budget: how much of the *label* variance the encoder's taste itself carries
    for y, w, have, head in list(targets.values())[::2]:
        p = np.asarray(fr.pred_f[head], float)
        out[f"budget_{head}"] = {"var_label": weighted_var(y[have], w[have]),
                                 "var_encoder_taste": weighted_var(p[have], w[have])}
    return out


def same_split(fr: S.ProbeFrame) -> dict:
    """The nameable features, the embedding and their union, all fitted on the
    encoder's train split and scored on its val split."""
    emb = fr.emb
    cols = [c for _, block in S.nameable_blocks(fr.join) for c in block]
    cat = S.categorical_indices(cols)
    F = np.column_stack([c for _, c in cols])
    out: dict = {"n_features": len(cols)}
    rows = []
    for head in HEADS:
        y, w, have = fr.label(head)
        tr, va = have & fr.train, have & fr.val
        variants = {
            "nameable features (GBM)": (F, "gbm"),
            "nameable features (linear)": (F, "lin"),
            "512-d embedding (ridge probe)": (emb, "ridge"),
            "embedding + nameable features (GBM)": (np.column_stack([emb, F]), "gbm"),
        }
        for vname, (X, kind) in variants.items():
            if kind == "gbm":
                cf = cat if X.shape[1] == F.shape[1] else [c + emb.shape[1] for c in cat]
                g = S.gbm(300, 0.08, 0.12, cf)
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
    return out


def main() -> None:
    fr = S.ProbeFrame()
    out = nameability_curve(fr)
    write_json(S.OUT / "s_r18.json", out, indent=1)
    print(json.dumps(out["reference"], indent=1))
    write_json(S.OUT / "s_r18b.json", same_split(fr), indent=1)


if __name__ == "__main__":
    main()
