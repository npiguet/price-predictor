"""C8 — the keyword label values against a vanilla-creature baseline.

Supersedes the keyword family of ``c7_labelside.py`` as the label side of
the keyword figure. c7 controls for total keyword count, which benchmarks
each keyword against carriers of the *other* keywords — a pool dominated
by flying, and a slightly different pool per keyword — and that baseline
made nearly every keyword read as a liability. This script fits one joint
WLS on solo-keyword carriers plus zero-keyword creatures: all sixteen
keyword flags at once, with MV, MV^2, power and toughness controls, so
every value answers "what is this keyword worth on an otherwise vanilla
creature" against one shared baseline. Demeaning the sixteen values puts
them on the same zero-mean scale as the c1 edit values.

Per keyword it also refits the three-channel decomposition (selection /
castability / contribution, linearised terms in label SD) on the same
solo-vs-vanilla design.

Writes ``c8_kw_vanilla.csv`` and prints the comparison against c1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from _lib.stats import wls  # noqa: E402
from c7_labelside import effect, mv_controls  # noqa: E402

SD = pl.SD["score_play"]
W_BAR, M_BAR = 0.4654, 0.3169  # corpus means from l_report §a
KW = cc.KEYWORD_FLAGS


def main() -> None:
    j = pl.load_frame()
    j["label_sp"] = pl.num(j, "shrunk_score_play") / SD
    med = pd.read_pickle(pl.SCRATCH / "l_mediation_table.pkl")
    mc = med["cards"][["card_name", "w", "m", "d"]].rename(
        columns={"w": "ch_w", "m": "ch_m", "d": "ch_d"})
    j = j.merge(mc, on="card_name", how="left")

    power, tough = pl.num(j, "power"), pl.num(j, "toughness")
    sub = j[pl.flag(j, "is_creature") & np.isfinite(power) & np.isfinite(tough)].reset_index(
        drop=True)
    kwn = pl.num(sub, "kw_count", 0)

    # joint fit: solo carriers of the 16 keywords + vanilla creatures
    any16 = np.column_stack([pl.flag(sub, c) for c in KW.values()]).any(axis=1)
    keep = (kwn == 0) | ((kwn == 1) & any16)
    dfk = sub[keep].reset_index(drop=True)
    X = np.column_stack(
        [np.ones(len(dfk))]
        + [pl.flag(dfk, c).astype(float) for c in KW.values()]
        + mv_controls(dfk, statline=True))
    y = dfk["label_sp"].to_numpy(float)
    w = dfk["w_score_play"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & (w > 0)
    fit = wls(X[ok], y[ok], w[ok])
    vals = dict(zip(KW, fit.beta[1:len(KW) + 1]))
    errs = dict(zip(KW, fit.se[1:len(KW) + 1]))
    mean_v = float(np.mean(list(vals.values())))

    edit = pd.read_csv(pl.SCRATCH / "c1_scale.csv").set_index("keyword")
    rows = []
    dkwn = pl.num(dfk, "kw_count", 0)
    for kw, col in KW.items():
        # channels vs vanilla only: this keyword's carriers + keywordless creatures
        dfg = dfk[(dkwn == 0) | pl.flag(dfk, col)].reset_index(drop=True)
        chan = {c: effect(dfg, pl.flag(dfg, col), c, mv_controls(dfg, statline=True))["beta"]
                for c in ("ch_w", "ch_m", "ch_d")}
        rows.append({
            "keyword": kw,
            "n_solo": int(pl.flag(sub[kwn == 1], col).sum()),
            "label_vs_vanilla": vals[kw], "se": errs[kw],
            "label_zero_mean": vals[kw] - mean_v,
            "edit_zero_mean": float(edit.loc[kw, "value_sp"]),
            "sel_term": 2 * M_BAR * chan["ch_w"] / SD,
            "cast_term": (2 * W_BAR - 1) * chan["ch_m"] / SD,
            "contrib_term": chan["ch_d"] / 2 / SD,
        })
    out = pd.DataFrame(rows).sort_values("edit_zero_mean", ascending=False)
    out.to_csv(pl.SCRATCH / "c8_kw_vanilla.csv", index=False)

    sp = out["label_zero_mean"].rank().corr(out["edit_zero_mean"].rank())
    r = float(np.corrcoef(out["label_zero_mean"], out["edit_zero_mean"])[0, 1])
    print(f"n rows in joint fit: {int(ok.sum())}; "
          f"mean keyword value vs vanilla: {mean_v:+.3f} SD")
    print(f"edit vs label (zero-mean): Spearman {sp:.3f}, Pearson {r:.3f}\n")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
