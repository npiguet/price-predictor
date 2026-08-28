"""R4 — cast_lift head anatomy: is it a (score_play, MV) re-encoding?"""

from __future__ import annotations

import json

import numpy as np

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
val = d["val"]
pf, ph = d["pf"], d["ph"]

pred_h = pl.predict_labels(emb, ph)
pred_f = pl.predict_labels(emb, pf)

mv = np.nan_to_num(S.num(join, "mv"))
pips = np.nan_to_num(S.num(join, "pips_total"))
lab_cast = S.num(join, "shrunk_cast_lift")
lab_play = S.num(join, "shrunk_score_play")
lab_rate = S.num(join, "shrunk_played_rate")
w_cast = S.num(join, "w_cast_lift")

out: dict = {}

# ── (a) output correlations on val cards ────────────────────────────────
sel = val & np.isfinite(lab_cast) & (w_cast > 0)
n = int(sel.sum())
out["a_n_val"] = n
pairs = {
    "pred cast_lift ↔ pred score_play": (pred_h["cast_lift"], pred_h["score_play"]),
    "pred cast_lift ↔ pred score_draw": (pred_h["cast_lift"], pred_h["score_draw"]),
    "pred cast_lift ↔ pred played_rate": (pred_h["cast_lift"], pred_h["played_rate"]),
    "pred cast_lift ↔ MV": (pred_h["cast_lift"], mv),
    "pred cast_lift ↔ total pips": (pred_h["cast_lift"], pips),
    "pred cast_lift ↔ label cast_lift": (pred_h["cast_lift"], lab_cast),
    "label cast_lift ↔ label score_play": (lab_cast, lab_play),
    "label cast_lift ↔ label played_rate": (lab_cast, lab_rate),
    "label cast_lift ↔ MV": (lab_cast, mv),
    "label cast_lift ↔ total pips": (lab_cast, pips),
    "pred score_play ↔ label score_play": (pred_h["score_play"], lab_play),
}
rows_a = []
for name, (a, b) in pairs.items():
    a = np.asarray(a, float)[sel]
    b = np.asarray(b, float)[sel]
    r_u, r_w = S.wcorr(a, b), S.wcorr(a, b, w_cast[sel])
    rows_a.append([name, n, f"{r_u:+.4f}", f"±{S.corr_se(r_u, n):.4f}", f"{r_w:+.4f}"])
out["a_corr"] = rows_a

# ── (b) probe-weight cosines ────────────────────────────────────────────
out["b_cosines"] = {
    f"{tag} cast_lift·{other}": S.cosine(ps.probes["cast_lift"].coef, ps.probes[other].coef)
    for tag, ps in (("honest", ph), ("fidelity", pf))
    for other in ("score_play", "score_draw", "played_rate", "played_rate@logit")
}

# ── (c) incremental validity of the cast_lift head ──────────────────────
def design(cols):
    return np.column_stack([np.ones(int(sel.sum()))] + [np.asarray(c, float)[sel] for c in cols])


def r2_of(X, y, w, folds=5, seed=42):
    """In-sample and 5-fold-CV weighted R²."""
    beta, _, _ = S.wls(X, y, w)
    ins = pl._r2(y, X @ beta, w)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    pred = np.empty(len(y))
    for part in np.array_split(order, folds):
        m = np.ones(len(y), bool)
        m[part] = False
        b, _, _ = S.wls(X[m], y[m], w[m])
        pred[part] = X[part] @ b
    return ins, pl._r2(y, pred, w)


y = lab_cast[sel]
w = w_cast[sel]
blocks = {
    "MV, MV²": [mv, mv**2],
    "pred score_play": [pred_h["score_play"]],
    "pred score_play + MV, MV²": [pred_h["score_play"], mv, mv**2],
    "pred score_play + MV, MV² + pred played_rate":
        [pred_h["score_play"], mv, mv**2, pred_h["played_rate"]],
    "… + pred cast_lift":
        [pred_h["score_play"], mv, mv**2, pred_h["played_rate"], pred_h["cast_lift"]],
    "… + pred score_draw too":
        [pred_h["score_play"], pred_h["score_draw"], mv, mv**2,
         pred_h["played_rate"], pred_h["cast_lift"]],
    "pred cast_lift alone": [pred_h["cast_lift"]],
    "pred cast_lift + MV, MV²": [pred_h["cast_lift"], mv, mv**2],
}
rows_c, prev = [], None
for name, cols in blocks.items():
    ins, cv = r2_of(design(cols), y, w)
    rows_c.append([name, len(cols), f"{ins:.4f}", f"{cv:.4f}",
                   "" if prev is None else f"{cv - prev:+.4f}"])
    if name.startswith("pred score_play + MV, MV² + pred played_rate"):
        prev = cv
    if name == "… + pred cast_lift":
        out["c_delta_cv_r2_from_cast_head"] = cv - prev
        out["c_full_cv_r2"] = cv
        prev = cv
out["c_nested"] = rows_c

# residual SD in label-SD units for the two key models
for name, cols in (("without cast head",
                    [pred_h["score_play"], mv, mv**2, pred_h["played_rate"]]),
                   ("with cast head",
                    [pred_h["score_play"], mv, mv**2, pred_h["played_rate"],
                     pred_h["cast_lift"]])):
    X = design(cols)
    beta, _, _ = S.wls(X, y, w)
    out[f"c_resid_sd_{name}"] = float(np.sqrt(np.average((y - X @ beta) ** 2, weights=w)))

# the mirror question: is the *predicted* cast_lift a re-encoding?
yp = np.asarray(pred_h["cast_lift"], float)[sel]
rows_m = []
for name, cols in (("pred score_play", [pred_h["score_play"]]),
                   ("+ MV, MV²", [pred_h["score_play"], mv, mv**2]),
                   ("+ pred played_rate",
                    [pred_h["score_play"], mv, mv**2, pred_h["played_rate"]]),
                   ("+ pred score_draw",
                    [pred_h["score_play"], pred_h["score_draw"], mv, mv**2,
                     pred_h["played_rate"]]),
                   ("+ total pips",
                    [pred_h["score_play"], pred_h["score_draw"], mv, mv**2,
                     pred_h["played_rate"], pips])):
    X = design(cols)
    beta, _, _ = S.wls(X, yp, w)
    rows_m.append([name, f"{pl._r2(yp, X @ beta, w):.4f}"])
out["c_pred_cast_reencoding"] = rows_m

# what fraction of the cast head's 512 weights is spanned by the other heads?
B = np.stack([ph.probes[h].coef for h in
              ("score_play", "score_draw", "played_rate", "played_rate@logit")], 1)
c = ph.probes["cast_lift"].coef
proj = B @ np.linalg.lstsq(B, c, rcond=None)[0]
out["c_weightvec_span_fraction"] = float((proj @ proj) / (c @ c))

with open(S.OUT / "s_r4.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
