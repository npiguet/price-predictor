"""R17 — the divergence table: where the encoder disagrees with its labels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import probe_lib as pl
import s_common as S
from _lib.jsonio import write_json
from _lib.stats import wcorr, wls

HEADS = ("score_play", "played_rate")
NAMED = ["ph_cycling", "ph_morph", "ph_sweeper", "ph_counterspell", "ai_remove_deck",
         "ph_kicker", "kw_defender", "ph_uncond_removal"]


def residuals(fr: S.ProbeFrame) -> dict[str, dict]:
    """Per head: the label, its weight, and both probe sets' prediction − label."""
    resid = {}
    for head in HEADS:
        lab = fr.label(head)
        resid[head] = {"y": lab.y, "w": lab.w, "have": lab.have,
                       "f": np.asarray(fr.pred_f[head], float) - lab.y,
                       "h": np.asarray(fr.pred_h[head], float) - lab.y}
    return resid


def residual_sds(fr: S.ProbeFrame, resid: dict) -> dict:
    out = {}
    for head, r in resid.items():
        hv = r["have"]
        out[f"resid_sd_{head}"] = {
            "fidelity, all": float(np.sqrt(np.average(r["f"][hv] ** 2, weights=r["w"][hv]))),
            "honest, val": float(np.sqrt(np.average(r["h"][hv & fr.val] ** 2,
                                                    weights=r["w"][hv & fr.val]))),
            "label SD": pl.SD[head],
        }
    return out


def features(fr: S.ProbeFrame, n_in: np.ndarray, dec: np.ndarray) -> dict[str, np.ndarray]:
    """The card-table flags common enough to test, plus the blacklist, two
    hostile classes, rarity, print era and n_in_deck decile."""
    join = fr.join
    hostile = S.hostile_classes(join)
    extra = {
        "ai_remove_deck": fr.num("ai_remove_deck", 0) > 0,
        "mana rock (noncreature artifact, taps for mana)": hostile["mana rock"],
        "fog / prevent damage": hostile["fog / prevent damage"],
    }
    for r_ in ("common", "uncommon", "rare", "mythic"):
        extra[f"rarity={r_}"] = (join["first_rarity"] == r_).to_numpy()
    yr = fr.num("first_year")
    for lo, hi in ((1993, 1999), (2000, 2006), (2007, 2013), (2014, 2019), (2020, 2027)):
        extra[f"first printed {lo}-{hi}"] = (yr >= lo) & (yr <= hi)
    for k in range(int(np.nanmax(dec)) + 1):
        lo, hi = n_in[dec == k].min(), n_in[dec == k].max()
        extra[f"n_in_deck decile {k + 1} ({int(lo)}–{int(hi)})"] = np.asarray(dec == k)

    feats: dict[str, np.ndarray] = {}
    for c in S.flag_columns(join):
        v = fr.num(c, 0) > 0
        if 40 <= v.sum() <= len(join) - 40:
            feats[c] = v
    feats.update(extra)
    return feats


def scopes(fr: S.ProbeFrame, r: dict) -> tuple:
    """``(tag, residual, rows)``: the fidelity fit over every card, the honest
    fit over the val cards."""
    return (("fidelity/all", r["f"], np.ones(len(fr), bool)), ("honest/val", r["h"], fr.val))


def divergence_table(fr: S.ProbeFrame, resid: dict, feats: dict,
                     base: np.ndarray) -> pd.DataFrame:
    """Each feature's raw and MV/creature-adjusted effect on each residual."""
    rows = []
    for head, r in resid.items():
        hv, w = r["have"], r["w"]
        for name, fv in feats.items():
            m = hv & np.isfinite(fv)
            if fv[m].sum() < 20:
                continue
            for tag, res, scope in scopes(fr, r):
                mm = m & scope
                if mm.sum() < 50 or fv[mm].sum() < 10:
                    continue
                ff = fv[mm].astype(float)
                raw = wls(np.column_stack([np.ones(int(mm.sum())), ff]), res[mm], w[mm])
                adj = wls(np.column_stack([base[mm], ff]), res[mm], w[mm])
                rows.append({
                    "head": head, "fit": tag, "feature": name, "n": int(fv[mm].sum()),
                    "raw_sd": raw.beta[-1] / pl.SD[head], "raw_t": raw.t[-1],
                    "adj_sd": adj.beta[-1] / pl.SD[head], "adj_t": adj.t[-1],
                })
    return pd.DataFrame(rows)


def hits(div: pd.DataFrame) -> dict:
    """The large, clear divergences, and the cells the prior named."""
    big = (div.raw_sd.abs() >= 0.10) & (div.raw_t.abs() > 3)
    hf = div[(div.fit == "fidelity/all") & big].sort_values(["head", "raw_sd"])
    hh = div[(div.fit == "honest/val") & big].sort_values(["head", "raw_sd"])
    return {
        "r17_hits": [[h.head, h.feature, h.n, f"{h.raw_sd:+.3f}", f"{h.raw_t:+.1f}",
                      f"{h.adj_sd:+.3f}", f"{h.adj_t:+.1f}"] for h in hf.itertuples()],
        "r17_n_tested": int((div.fit == "fidelity/all").sum()),
        "r17_n_hits": int(len(hf)),
        "r17_hits_honest": [[h.head, h.feature, h.n, f"{h.raw_sd:+.3f}", f"{h.raw_t:+.1f}"]
                            for h in hh.itertuples()],
        "r17_named_cells": [
            [r.head, r.fit, r.feature, r.n, f"{r.raw_sd:+.3f}", f"{r.raw_t:+.1f}",
             f"{r.adj_sd:+.3f}", f"{r.adj_t:+.1f}"]
            for r in div[div.feature.isin(NAMED)].itertuples()],
    }


def decile_profile(fr: S.ProbeFrame, resid: dict, n_in: np.ndarray,
                   dec: np.ndarray) -> list[list]:
    """Mean residual per n_in_deck decile, both heads and both fits."""
    rows = []
    for k in range(int(np.nanmax(dec)) + 1):
        m = np.asarray(dec == k)
        row = [k + 1, f"{int(n_in[m].min())}–{int(n_in[m].max())}", int(m.sum())]
        for head, r in resid.items():
            for tag in ("f", "h"):
                sub = m & r["have"] & (fr.val if tag == "h" else np.ones(len(fr), bool))
                if sub.sum() < 10:
                    row += ["—"]
                    continue
                v = np.average(r[tag][sub], weights=r["w"][sub]) / pl.SD[head]
                row += [f"{v:+.3f}"]
        rows.append(row)
    return rows


def blacklist(fr: S.ProbeFrame, resid: dict, base: np.ndarray) -> list[list]:
    """The Forge AI blacklist's residual, raw and MV/creature-adjusted."""
    bl = fr.num("ai_remove_deck", 0) > 0
    rows = []
    for head, r in resid.items():
        for tag, res, scope in scopes(fr, r):
            m = r["have"] & scope
            fit = wls(np.column_stack([base[m], bl[m].astype(float)]), res[m], r["w"][m])
            mb = np.average(res[m & bl], weights=r["w"][m & bl]) / pl.SD[head]
            mn = np.average(res[m & ~bl], weights=r["w"][m & ~bl]) / pl.SD[head]
            rows.append([head, tag, int((m & bl).sum()), f"{mb:+.3f}", f"{mn:+.3f}",
                         f"{fit.beta[-1] / pl.SD[head]:+.3f}",
                         f"{fit.se[-1] / pl.SD[head]:.3f}", f"{fit.t[-1]:+.1f}"])
    return rows


def neighbourhood(fr: S.ProbeFrame, resid: dict, n_in: np.ndarray) -> dict:
    """Is the residual pulled toward the embedding neighbourhood's labels?
    Saves the k=10 neighbour table ``s_r17b.py`` reuses."""
    nb_idx = S.top_k_cosine(fr.emb / np.linalg.norm(fr.emb, axis=1, keepdims=True), 10)
    np.save(S.OUT / "s_r17_neighbours.npy", nb_idx)

    rare = {c: fr.num(c, 0) > 0 for c in S.flag_columns(fr.join)}
    rare = {c: v for c, v in rare.items() if 20 <= v.sum() < 200}
    any_rare = (np.any(np.stack(list(rare.values())), 0) if rare
                else np.zeros(len(fr), bool))
    rows = []
    for head, r in resid.items():
        y, w, hv = r["y"], r["w"], r["have"]
        pull = S.neighbour_mean(y, nb_idx) - y
        for tag, res in (("fidelity/all", r["f"]), ("honest/val", r["h"])):
            for sub_name, sub in (("all cards", np.ones(len(fr), bool)),
                                  ("rare-mechanic cards", any_rare),
                                  ("n_in_deck < 200", n_in < 200)):
                m = hv & np.isfinite(pull) & sub
                if tag == "honest/val":
                    m = m & fr.val
                if m.sum() < 50:
                    continue
                fit = wls(np.column_stack([np.ones(int(m.sum())), pull[m]]), res[m], w[m])
                rows.append([head, tag, sub_name, int(m.sum()), f"{fit.beta[1]:+.4f}",
                             f"{fit.se[1]:.4f}", f"{fit.t[1]:+.1f}",
                             f"{wcorr(res[m], pull[m], w[m]):+.3f}"])

    # per rare class (fidelity): is each pulled toward its neighbourhood?
    r = resid["score_play"]
    sd = pl.SD["score_play"]
    rows_rc = []
    for c, v in sorted(rare.items(), key=lambda kv: -kv[1].sum()):
        m = r["have"] & v
        if m.sum() < 20:
            continue
        nb_mean = S.neighbour_mean(r["y"], nb_idx[m])
        rows_rc.append([c, int(m.sum()),
                        f"{np.average(r['f'][m], weights=r['w'][m]) / sd:+.3f}",
                        f"{np.nanmean(nb_mean - r['y'][m]) / sd:+.3f}"])
    return {"r17_rare_classes": {c: int(v.sum()) for c, v in rare.items()},
            "r17_neighbourhood": rows, "r17_rare_class_rows": rows_rc}


def main() -> None:
    fr = S.ProbeFrame()
    n_in = fr.num("n_in_deck")
    dec = pd.qcut(n_in, 10, labels=False, duplicates="drop")
    base = S.base_controls(fr.join)
    resid = residuals(fr)

    out = residual_sds(fr, resid)
    div = divergence_table(fr, resid, features(fr, n_in, dec), base)
    div.to_pickle(S.OUT / "s_r17_divergence.pkl")
    out |= hits(div)
    out["r17_decile_profile"] = decile_profile(fr, resid, n_in, dec)
    out["r17_blacklist"] = blacklist(fr, resid, base)
    out |= neighbourhood(fr, resid, n_in)

    write_json(S.OUT / "s_r17.json", out, indent=1)
    print(json.dumps(out, indent=1, default=float)[:12000])


if __name__ == "__main__":
    main()
