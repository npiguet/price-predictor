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
from c7_labelside import effect, wls  # noqa: E402

SD = cc.SD["score_play"]
W_BAR, M_BAR = 0.4654, 0.3169  # corpus means from l_report §a

KW = {
    "flying": "kw_flying", "lifelink": "kw_lifelink",
    "deathtouch": "kw_deathtouch", "vigilance": "kw_vigilance",
    "first strike": "kw_first_strike", "double strike": "kw_double_strike",
    "trample": "kw_trample", "haste": "kw_haste", "reach": "kw_reach",
    "menace": "kw_menace", "defender": "kw_defender",
    "hexproof": "kw_hexproof", "shroud": "kw_shroud",
    "indestructible": "kw_indestructible", "flash": "kw_flash",
    "ward {2}": "kw_ward",
}


def main() -> None:
    import pickle

    j = cc.join_table().copy()
    j["label_sp"] = pd.to_numeric(j["shrunk_score_play"], errors="coerce") / SD
    med = pickle.load(open(cc.SCRATCH / "l_mediation_table.pkl", "rb"))
    mc = med["cards"][["card_name", "w", "m", "d"]].rename(
        columns={"w": "ch_w", "m": "ch_m", "d": "ch_d"})
    j = j.merge(mc, on="card_name", how="left")

    crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    power = pd.to_numeric(j["power"], errors="coerce").to_numpy(float)
    tough = pd.to_numeric(j["toughness"], errors="coerce").to_numpy(float)
    sub = j[crea & np.isfinite(power) & np.isfinite(tough)].reset_index(drop=True)
    kwn = pd.to_numeric(sub["kw_count"], errors="coerce").fillna(0).to_numpy(float)

    def ctrls(df: pd.DataFrame) -> list[np.ndarray]:
        mv = pd.to_numeric(df["mv"], errors="coerce").fillna(0).to_numpy(float)
        p = pd.to_numeric(df["power"], errors="coerce").to_numpy(float)
        t = pd.to_numeric(df["toughness"], errors="coerce").to_numpy(float)
        return [mv, mv ** 2, p, t]

    # joint fit: solo carriers of the 16 keywords + vanilla creatures
    any16 = np.column_stack(
        [sub[c].fillna(False).astype(bool) for c in KW.values()]).any(axis=1)
    keep = (kwn == 0) | ((kwn == 1) & any16)
    dfk = sub[keep].reset_index(drop=True)
    X = np.column_stack(
        [np.ones(len(dfk))]
        + [dfk[c].fillna(False).astype(float).to_numpy() for c in KW.values()]
        + ctrls(dfk))
    y = dfk["label_sp"].to_numpy(float)
    w = dfk["w_score_play"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1) & (w > 0)
    beta, se = wls(X[ok], y[ok], w[ok])
    vals = dict(zip(KW, beta[1:len(KW) + 1]))
    errs = dict(zip(KW, se[1:len(KW) + 1]))
    mean_v = float(np.mean(list(vals.values())))

    edit = pd.read_csv(cc.SCRATCH / "c1_scale.csv").set_index("keyword")
    rows = []
    dkwn = pd.to_numeric(dfk["kw_count"], errors="coerce").fillna(0).to_numpy(float)
    for kw, col in KW.items():
        flag_all = dfk[col].fillna(False).astype(bool).to_numpy()
        # channels vs vanilla only: this keyword's carriers + keywordless creatures
        dfg = dfk[(dkwn == 0) | flag_all].reset_index(drop=True)
        flag = dfg[col].fillna(False).astype(bool).to_numpy()
        chan = {c: effect(dfg, flag, c, ctrls(dfg))["beta"]
                for c in ("ch_w", "ch_m", "ch_d")}
        rows.append({
            "keyword": kw,
            "n_solo": int(sub[(kwn == 1)][col].fillna(False).astype(bool).sum()),
            "label_vs_vanilla": vals[kw], "se": errs[kw],
            "label_zero_mean": vals[kw] - mean_v,
            "edit_zero_mean": float(edit.loc[kw, "value_sp"]),
            "sel_term": 2 * M_BAR * chan["ch_w"] / SD,
            "cast_term": (2 * W_BAR - 1) * chan["ch_m"] / SD,
            "contrib_term": chan["ch_d"] / 2 / SD,
        })
    out = pd.DataFrame(rows).sort_values("edit_zero_mean", ascending=False)
    out.to_csv(cc.SCRATCH / "c8_kw_vanilla.csv", index=False)

    sp = out["label_zero_mean"].rank().corr(out["edit_zero_mean"].rank())
    r = float(np.corrcoef(out["label_zero_mean"], out["edit_zero_mean"])[0, 1])
    print(f"n rows in joint fit: {int(ok.sum())}; "
          f"mean keyword value vs vanilla: {mean_v:+.3f} SD")
    print(f"edit vs label (zero-mean): Spearman {sp:.3f}, Pearson {r:.3f}\n")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
