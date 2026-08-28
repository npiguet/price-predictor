"""Derived statistics for the R1/R2 report.

Three things the per-probe scripts could not compute in place:

* R1a's survival ratios (degraded val R² as a fraction of the control's).
* R1b's manifold-gated recompute (the wide table carries a NaN column
  from the sibling family, which defeats a naive all-columns gate).
* R1c's *paired null-adjusted* flip effect — the placebo families are not
  mean-zero, so the honest statistic is ``mean(Δflip − Δnull)`` over the
  cards carrying both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402

SD = 0.06181
GATE = 0.35325


def r1a() -> None:
    agg = pd.read_csv(pl.SCRATCH / "r1a_shuffle_r2_agg.csv")
    base = agg[agg["condition"] == "none"].set_index("head")["val_r2"]
    agg["survival"] = [row.val_r2 / base[row.head] for row in agg.itertuples()]
    piv = agg.pivot(index="head", columns="condition", values="val_r2").round(4)
    surv = agg.pivot(index="head", columns="condition", values="survival").round(3)
    print("=== R1a val R² by condition\n", piv.to_string())
    print("\n=== R1a survival (fraction of control val R²)\n", surv.to_string())
    piv.to_csv(pl.SCRATCH / "r1a_val_r2_table.csv")
    surv.to_csv(pl.SCRATCH / "r1a_survival_table.csv")

    shift = pd.read_csv(pl.SCRATCH / "r1a_shuffle_pred_shift.csv")
    s = (shift.groupby(["condition", "head"])
              .agg(mean_abs_delta=("mean_abs_delta", "mean"),
                   median_abs_delta=("median_abs_delta", "mean"),
                   pearson_vs_base=("pearson_vs_base", "mean"))
              .reset_index())
    s["mean_abs_delta_sd"] = np.where(
        s["head"] == "played_rate", s["mean_abs_delta"] / 0.1247,
        s["mean_abs_delta"] / SD)
    s.to_csv(pl.SCRATCH / "r1a_pred_shift_table.csv", index=False)
    print("\n=== R1a fidelity-probe prediction shift\n",
          s[s["head"].isin(["score_play", "played_rate", "cast_lift"])]
          .round(4).to_string(index=False))


def r1b() -> None:
    w = pd.read_csv(pl.SCRATCH / "r1b_wide.csv")
    A = w[w["family"] == "single"]
    cols = ["manifold_dist__i_original", "manifold_dist__ii_deleted",
            "manifold_dist__iii_grant_spell", "manifold_dist__iv_negated_static"]
    keep = (A[cols] <= GATE).all(axis=1)
    print(f"\n=== R1b family A: {len(A)} single-line fliers, "
          f"{int(keep.sum())} with every variant on-manifold")
    for tag, sub in (("all", A), ("on_manifold", A[keep])):
        dt = (sub["score_play__i_original"] - sub["score_play__ii_deleted"]).to_numpy()
        dg = (sub["score_play__iii_grant_spell"] - sub["score_play__ii_deleted"]).to_numpy()
        dn = (sub["score_play__iv_negated_static"] - sub["score_play__ii_deleted"]).to_numpy()
        print(f"  {tag:12s} n={len(sub):3d}  true={dt.mean()/SD:+.3f} SD  "
              f"grant={dg.mean()/SD:+.3f} SD  negated={dn.mean()/SD:+.3f} SD  "
              f"index_grant={dg.mean()/dt.mean():+.3f}  "
              f"index_neg={dn.mean()/dt.mean():+.3f}")


def r1c() -> None:
    v = pd.read_csv(pl.SCRATCH / "r1c_variants.csv")
    rows = []
    for family, sub in v.groupby("family"):
        base = sub[sub["variant"] == "base"].set_index("row")["score_play"]
        flip = sub[sub["variant"] == "flip"].set_index("row")["score_play"]
        d_flip = (flip - base.loc[flip.index])
        for variant in sorted(sub["variant"].unique()):
            if not variant.startswith("null_"):
                continue
            nv = sub[sub["variant"] == variant].set_index("row")["score_play"]
            common = nv.index.intersection(flip.index)
            if len(common) < 5:
                continue
            d_null = (nv.loc[common] - base.loc[common])
            adj = (d_flip.loc[common] - d_null).to_numpy()
            rng = np.random.default_rng(5)
            idx = rng.integers(0, len(adj), size=(4000, len(adj)))
            draws = adj[idx].mean(axis=1) / SD
            try:
                _, p = wilcoxon(adj, alternative="less")
            except ValueError:
                p = float("nan")
            rows.append({
                "family": family, "null": variant, "n": len(common),
                "flip_mean_sd": float(d_flip.loc[common].mean() / SD),
                "null_mean_sd": float(d_null.mean() / SD),
                "adjusted_mean_sd": float(adj.mean() / SD),
                "adj_ci_lo": float(np.percentile(draws, 2.5)),
                "adj_ci_hi": float(np.percentile(draws, 97.5)),
                "frac_adj_negative": float((adj < 0).mean()),
                "wilcoxon_p_less": float(p),
            })
    out = pd.DataFrame(rows)
    out.to_csv(pl.SCRATCH / "r1c_null_adjusted.csv", index=False)
    print("\n=== R1c null-adjusted flip effect (label SD of score_play)\n",
          out.round(4).to_string(index=False))


if __name__ == "__main__":
    r1a()
    r1b()
    r1c()
