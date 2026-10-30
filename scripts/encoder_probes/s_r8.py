"""R8 — played_rate, the encoder's loudest axis: cost, hostility, content.

Writes three files. ``s_r8.json`` holds the analysis: how much of played_rate
is cost, the AI-hostile classes' matched deficits and whether they share one
direction, and what loads on the played_rate direction.

``s_r8b.json`` asks whether that direction test is powered. A cosine of 0.05
between two 512-dim class directions only means "no shared axis" if the same
class split in half reproduces itself well above that; it also runs the
transfer test in prediction space, which does not depend on the direction
vector's noisy tail.

``s_r8c.json`` bootstraps standard errors for the matched deficits.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import wls

SD_PLAYED = pl.SD["played_rate"]
TYPE_FLAGS = ("is_creature", "is_instant", "is_sorcery", "is_artifact", "is_enchantment",
              "is_land", "is_planeswalker", "is_aura", "is_equipment", "is_vehicle",
              "is_legendary")


def cost_share(fr: S.ProbeFrame, rate: pl.Label, mv: np.ndarray) -> list[list]:
    """(a) nested WLS R² of played_rate on ever more of the card's nameable text."""
    def col(c):
        return fr.num(c, 0)

    cost = [mv, mv**2, col("pips_total"), col("generic"), col("n_colors"),
            col("is_colorless_cost"), col("has_x"), col("hybrid")]
    types = [col(c) for c in TYPE_FLAGS]
    pt = [col("power"), col("toughness"), np.isfinite(fr.num("power")).astype(float)]
    kws = [col(c) for c in fr.join.columns if c.startswith("kw_")]
    phs = [col(c) for c in fr.join.columns if c.startswith("ph_")]
    tr, va = rate.have & fr.train, rate.have & fr.val
    y, w = rate.y, rate.w

    rows = []
    for name, cols in (("MV, MV² only", [mv, mv**2]),
                       ("cost line (MV, MV², pips, generic, colours, X, hybrid)", cost),
                       ("cost + type flags", cost + types),
                       ("cost + type + P/T", cost + types + pt),
                       ("cost + type + P/T + keywords", cost + types + pt + kws),
                       ("cost + type + P/T + keywords + phrases",
                        cost + types + pt + kws + phs)):
        X = np.column_stack([np.ones(len(fr))] + cols)
        b = wls(X[tr], y[tr], w[tr]).beta
        rows.append([name, len(cols), f"{pl._r2(y[tr], X[tr] @ b, w[tr]):.4f}",
                     f"{pl._r2(y[va], X[va] @ b, w[va]):.4f}"])
    rows.append(["512-dim embedding, honest ridge probe", 512, "0.9009",
                 f"{fr.ph.probes['played_rate'].metrics['val_r2']:.4f}"])
    return rows


def hostility(fr: S.ProbeFrame, rate: pl.Label, classes: dict, cell: np.ndarray,
              dirs: dict[str, np.ndarray]) -> dict:
    """(b) the classes' matched deficits, whether their directions share an
    axis, and what deflating that axis does to the played_rate probe."""
    targets = {
        "label played_rate": rate.y,
        "fidelity pred played_rate": np.asarray(fr.pred_f["played_rate"], float),
        "honest pred played_rate": np.asarray(fr.pred_h["played_rate"], float),
    }
    rows = []
    for name, members in classes.items():
        row = [name, int(members.sum())]
        for vals in targets.values():
            dfc, _ = S.matched_deficit(cell, members, vals, rate.w)
            row += [f"{dfc:+.4f}", f"{dfc / SD_PLAYED:+.2f}"]
        rows.append(row)
    out: dict = {"b_matched_deficits": rows}

    names = list(dirs)
    D = np.stack([dirs[n] for n in names])
    out["b_direction_cosines"] = [
        [names[i]] + [f"{float(D[i] @ D[j]):+.3f}" for j in range(len(names))]
        for i in range(len(names))
    ]
    out["b_direction_names"] = names
    _, s_svd, vt_svd = np.linalg.svd(D, full_matrices=False)
    out["b_shared_component_var_share"] = [float(x) for x in (s_svd**2 / (s_svd**2).sum())]
    u = vt_svd[0]
    out["b_cos_shared_vs_played_probe"] = S.cosine(u, fr.ph.probes["played_rate"].coef)
    out["b_cos_shared_vs_score_probe"] = S.cosine(u, fr.ph.probes["score_play"].coef)
    out["b_loadings_on_shared"] = {n: float(dirs[n] @ u) for n in names}

    # deflate the shared direction out of the embeddings and re-read the probe;
    # also the single best-aligned direction, the played_rate probe itself
    emb = fr.emb
    probe = fr.ph.probes["played_rate"]
    pred_raw = emb @ probe.coef + probe.intercept
    pred_def = (emb - np.outer(emb @ u, u)) @ probe.coef + probe.intercept
    pc_pr = probe.coef / np.linalg.norm(probe.coef)
    pred_def2 = (emb - np.outer(emb @ pc_pr, pc_pr)) @ probe.coef + probe.intercept
    rows_def = []
    for name, members in classes.items():
        base, _ = S.matched_deficit(cell, members, pred_raw, rate.w)
        defl, _ = S.matched_deficit(cell, members, pred_def, rate.w)
        rows_def.append([name, f"{base:+.4f}", f"{defl:+.4f}",
                         f"{(1 - defl / base) * 100:+.0f}%" if base else "—"])
    out["b_deflation"] = rows_def
    out["b_pred_sd_before_after"] = [float(pred_raw.std()), float(pred_def.std()),
                                     float(pred_def2.std())]
    return out


def direction_loadings(fr: S.ProbeFrame, rate: pl.Label) -> dict:
    """(c) each card-table flag's MV/creature-controlled effect on the
    predicted and the label played_rate."""
    base = S.base_controls(fr.join)
    pred = np.asarray(fr.pred_f["played_rate"], float)
    rows = []
    for c in S.flag_columns(fr.join):
        fv = fr.num(c, 0)
        if fv.min() == fv.max() or ((fv > 0).sum() < 40):
            continue
        X = np.column_stack([base, (fv > 0).astype(float)])
        fp = wls(X[rate.have], pred[rate.have], rate.w[rate.have])
        fl = wls(X[rate.have], rate.y[rate.have], rate.w[rate.have])
        rows.append([c, int((fv > 0).sum()), fp.beta[-1] / SD_PLAYED, fp.t[-1],
                     fl.beta[-1] / SD_PLAYED, fl.t[-1]])
    rc = pd.DataFrame(rows, columns=["feature", "n", "pred_sd", "pred_t", "label_sd", "label_t"])
    rc = rc.sort_values("pred_sd")

    def cells(frame: pd.DataFrame) -> list[list]:
        return [[r.feature, int(r.n), f"{r.pred_sd:+.3f}", f"{r.pred_t:+.1f}",
                 f"{r.label_sd:+.3f}", f"{r.label_t:+.1f}"] for r in frame.itertuples()]

    return {"c_bottom": cells(rc.head(18)), "c_top": cells(rc.tail(18).iloc[::-1]),
            "c_corr_pred_label_effects": float(np.corrcoef(rc.pred_sd, rc.label_sd)[0, 1]),
            "c_n_features": int(len(rc))}


def direction_reliability(fr: S.ProbeFrame, classes: dict, cell: np.ndarray,
                          dirs: dict[str, np.ndarray]) -> dict:
    """Split-half reliability of each class direction against two random
    same-sized sets, and the transfer matrix in prediction space."""
    rng = np.random.default_rng(42)
    n = len(fr)

    def mask_of(rows) -> np.ndarray:
        m = np.zeros(n, bool)
        m[rows] = True
        return m

    rows_rel = []
    for name, members in classes.items():
        idx = np.where(members)[0]
        perm = rng.permutation(len(idx))
        v1 = S.class_direction(fr.emb, mask_of(idx[perm[: len(idx) // 2]]))
        v2 = S.class_direction(fr.emb, mask_of(idx[perm[len(idx) // 2:]]))
        rows_rel.append([name, int(members.sum()), f"{float(v1 @ v2):+.3f}"])

    null = []
    for name, members in classes.items():
        k = int(members.sum())
        va = S.class_direction(fr.emb, mask_of(rng.choice(n, k, replace=False)))
        vb = S.class_direction(fr.emb, mask_of(rng.choice(n, k, replace=False)))
        null.append([name, k, f"{float(va @ vb):+.3f}"])

    # score class b's cards with class a's direction: matched mean projection
    # difference, class b vs its MV×type controls
    rows_tr = []
    for a in classes:
        proj = fr.emb @ dirs[a]
        row = [a]
        for members in classes.values():
            diff, n_cells = S.matched_deficit(cell, members, proj)
            row.append(f"{diff / proj.std():+.2f}" if n_cells else "—")
        rows_tr.append(row)
    return {"split_half_cosine": rows_rel, "random_set_cosine": null,
            "transfer_matrix_sd_units": rows_tr, "transfer_names": list(classes)}


def deficit_errors(fr: S.ProbeFrame, rate: pl.Label, classes: dict,
                   cell: np.ndarray) -> list[list]:
    """The matched played_rate deficits with a 200-draw card bootstrap SE."""
    def deficit(members, vals, idx):
        return S.matched_deficit(cell[idx], members[idx], vals[idx], rate.w[idx])[0]

    rng = np.random.default_rng(42)
    n = len(fr)
    rows = []
    for name, members in classes.items():
        row = [name, int(members.sum())]
        for vals in (rate.y, np.asarray(fr.pred_f["played_rate"], float)):
            point = deficit(members, vals, np.arange(n))
            boots = np.array([deficit(members, vals, rng.integers(0, n, n))
                              for _ in range(200)])
            se = float(np.nanstd(boots))
            row += [f"{point:+.4f}", f"{se:.4f}", f"{point / SD_PLAYED:+.2f}",
                    f"{se / SD_PLAYED:.2f}"]
        rows.append(row)
        print(row, flush=True)
    return rows


def main() -> None:
    fr = S.ProbeFrame()
    rate = fr.label("played_rate")
    mv = fr.num("mv", 0)
    classes = S.hostile_classes(fr.join)
    cell = S.control_cells(fr.join)
    dirs = {name: S.class_direction(fr.emb, members) for name, members in classes.items()}

    out: dict = {"a_nested_r2": cost_share(fr, rate, mv)}
    out |= hostility(fr, rate, classes, cell, dirs)
    out |= direction_loadings(fr, rate)
    write_json(S.OUT / "s_r8.json", out, indent=1)
    print(json.dumps(out, indent=1, default=float))

    robust = direction_reliability(fr, classes, cell, dirs)
    write_json(S.OUT / "s_r8b.json", robust, indent=1)
    print(json.dumps(robust, indent=1, default=float))

    write_json(S.OUT / "s_r8c.json",
               {"matched_deficits_with_se": deficit_errors(fr, rate, classes, cell)}, indent=1)


if __name__ == "__main__":
    main()
