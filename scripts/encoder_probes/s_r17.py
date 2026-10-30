"""R17 — the divergence table: where the encoder disagrees with its labels."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import probe_lib as pl
import s_common as S
from _lib.stats import wcorr, wls

d = S.load_all()
join, emb = d["join"], d["emb"]
val, train = d["val"], d["train"]
pf, ph = d["pf"], d["ph"]
pred_f = pl.predict_labels(emb, pf)
pred_h = pl.predict_labels(emb, ph)

mv = S.flag(join, "mv")
creature = S.flag(join, "is_creature")
n_in = S.num(join, "n_in_deck")

out: dict = {}
HEADS = ("score_play", "played_rate")

resid = {}
for head in HEADS:
    y = S.num(join, f"shrunk_{head}")
    w = S.num(join, f"w_{head}")
    resid[head] = {
        "y": y, "w": w,
        "have": np.isfinite(y) & (w > 0),
        "f": np.asarray(pred_f[head], float) - y,
        "h": np.asarray(pred_h[head], float) - y,
    }
    r = resid[head]
    out[f"resid_sd_{head}"] = {
        "fidelity, all": float(np.sqrt(np.average(r["f"][r["have"]] ** 2,
                                                  weights=r["w"][r["have"]]))),
        "honest, val": float(np.sqrt(np.average(r["h"][r["have"] & val] ** 2,
                                                weights=r["w"][r["have"] & val]))),
        "label SD": S.SD[head],
    }

# ── feature sweep ───────────────────────────────────────────────────────
flag_cols = sorted({c for c in join.columns
                    if c.startswith(("kw_", "ph_", "is_")) and c != "is_primary"}
                   | {"big_tribe", "has_x", "hybrid", "phyrexian",
                      "is_colorless_cost", "has_cost_line"})
extra_flags = {
    "ai_remove_deck": S.flag(join, "ai_remove_deck") > 0,
    "mana rock (noncreature artifact, taps for mana)":
        (S.flag(join, "is_artifact") > 0) & (creature == 0)
        & (S.flag(join, "ph_tap_for_mana") > 0),
    "fog / prevent damage":
        (S.flag(join, "ph_fog") + S.flag(join, "ph_prevent_damage")) > 0,
}
for r_ in ("common", "uncommon", "rare", "mythic"):
    extra_flags[f"rarity={r_}"] = (join["first_rarity"] == r_).to_numpy()
yr = pd.to_numeric(join["first_year"], errors="coerce").to_numpy(float)
for lo, hi in ((1993, 1999), (2000, 2006), (2007, 2013), (2014, 2019), (2020, 2027)):
    extra_flags[f"first printed {lo}-{hi}"] = (yr >= lo) & (yr <= hi)
dec = pd.qcut(n_in, 10, labels=False, duplicates="drop")
for k in range(int(np.nanmax(dec)) + 1):
    lo, hi = n_in[dec == k].min(), n_in[dec == k].max()
    extra_flags[f"n_in_deck decile {k + 1} ({int(lo)}–{int(hi)})"] = np.asarray(dec == k)

feats: dict[str, np.ndarray] = {}
for c in flag_cols:
    v = S.flag(join, c) > 0
    if 40 <= v.sum() <= len(join) - 40:
        feats[c] = v
feats.update(extra_flags)

rows = []
for head in HEADS:
    r = resid[head]
    hv, w = r["have"], r["w"]
    base = np.column_stack([np.ones(len(join)), mv, mv**2, creature])
    for name, fv in feats.items():
        m = hv & np.isfinite(fv)
        if fv[m].sum() < 20:
            continue
        f = fv[m].astype(float)
        for tag, res in (("fidelity/all", r["f"]), ("honest/val", r["h"])):
            mm = m & (val if tag == "honest/val" else np.ones(len(join), bool))
            if mm.sum() < 50 or fv[mm].sum() < 10:
                continue
            ff = fv[mm].astype(float)
            X0 = np.column_stack([np.ones(int(mm.sum())), ff])
            raw = wls(X0, res[mm], w[mm])
            adj = wls(np.column_stack([base[mm], ff]), res[mm], w[mm])
            rows.append({
                "head": head, "fit": tag, "feature": name, "n": int(fv[mm].sum()),
                "raw_sd": raw.beta[-1] / S.SD[head], "raw_t": raw.t[-1],
                "adj_sd": adj.beta[-1] / S.SD[head], "adj_t": adj.t[-1],
            })
div = pd.DataFrame(rows)
div.to_pickle(S.OUT / "s_r17_divergence.pkl")

hits = div[(div.fit == "fidelity/all") & (div.raw_sd.abs() >= 0.10) & (div.raw_t.abs() > 3)]
hits = hits.sort_values(["head", "raw_sd"])
out["r17_hits"] = [[h.head, h.feature, h.n, f"{h.raw_sd:+.3f}", f"{h.raw_t:+.1f}",
                    f"{h.adj_sd:+.3f}", f"{h.adj_t:+.1f}"] for h in hits.itertuples()]
out["r17_n_tested"] = int((div.fit == "fidelity/all").sum())
out["r17_n_hits"] = int(len(hits))

hits_h = div[(div.fit == "honest/val") & (div.raw_sd.abs() >= 0.10) & (div.raw_t.abs() > 3)]
hits_h = hits_h.sort_values(["head", "raw_sd"])
out["r17_hits_honest"] = [[h.head, h.feature, h.n, f"{h.raw_sd:+.3f}", f"{h.raw_t:+.1f}"]
                          for h in hits_h.itertuples()]

# the cells the prior named explicitly
named = ["ph_cycling", "ph_morph", "ph_sweeper", "ph_counterspell", "ai_remove_deck",
         "ph_kicker", "kw_defender", "ph_uncond_removal"]
out["r17_named_cells"] = [
    [r.head, r.fit, r.feature, r.n, f"{r.raw_sd:+.3f}", f"{r.raw_t:+.1f}",
     f"{r.adj_sd:+.3f}", f"{r.adj_t:+.1f}"]
    for r in div[div.feature.isin(named)].itertuples()]

# ── n_in_deck decile profile ────────────────────────────────────────────
rows_n = []
for k in range(int(np.nanmax(dec)) + 1):
    m = np.asarray(dec == k)
    row = [k + 1, f"{int(n_in[m].min())}–{int(n_in[m].max())}", int(m.sum())]
    for head in HEADS:
        r = resid[head]
        for tag in ("f", "h"):
            sub = m & r["have"] & (val if tag == "h" else np.ones(len(join), bool))
            if sub.sum() < 10:
                row += ["—"]
                continue
            v = np.average(r[tag][sub], weights=r["w"][sub]) / S.SD[head]
            row += [f"{v:+.3f}"]
    rows_n.append(row)
out["r17_decile_profile"] = rows_n

# ── blacklist ───────────────────────────────────────────────────────────
bl = S.flag(join, "ai_remove_deck") > 0
rows_bl = []
for head in HEADS:
    r = resid[head]
    for tag, res in (("fidelity/all", r["f"]), ("honest/val", r["h"])):
        m = r["have"] & (val if tag == "honest/val" else np.ones(len(join), bool))
        X = np.column_stack([base[m], bl[m].astype(float)])
        fit = wls(X, res[m], r["w"][m])
        mb = np.average(res[m & bl], weights=r["w"][m & bl]) / S.SD[head]
        mn = np.average(res[m & ~bl], weights=r["w"][m & ~bl]) / S.SD[head]
        rows_bl.append([head, tag, int((m & bl).sum()), f"{mb:+.3f}", f"{mn:+.3f}",
                        f"{fit.beta[-1] / S.SD[head]:+.3f}",
                        f"{fit.se[-1] / S.SD[head]:.3f}", f"{fit.t[-1]:+.1f}"])
out["r17_blacklist"] = rows_bl

# ── MLM-neighbourhood test ──────────────────────────────────────────────
nb_idx = S.top_k_cosine(emb / np.linalg.norm(emb, axis=1, keepdims=True), 10)
np.save(S.OUT / "s_r17_neighbours.npy", nb_idx)

rare = {c: S.flag(join, c) > 0 for c in flag_cols}
rare = {c: v for c, v in rare.items() if 20 <= v.sum() < 200}
out["r17_rare_classes"] = {c: int(v.sum()) for c, v in rare.items()}

rows_nb = []
for head in HEADS:
    r = resid[head]
    y, w, hv = r["y"], r["w"], r["have"]
    pull = S.neighbour_mean(y, nb_idx) - y
    for tag, res in (("fidelity/all", r["f"]), ("honest/val", r["h"])):
        for sub_name, sub in (("all cards", np.ones(len(join), bool)),
                              ("rare-mechanic cards",
                               np.any(np.stack(list(rare.values())), 0) if rare
                               else np.zeros(len(join), bool)),
                              ("n_in_deck < 200", n_in < 200)):
            m = hv & np.isfinite(pull) & sub
            if tag == "honest/val":
                m = m & val
            if m.sum() < 50:
                continue
            X = np.column_stack([np.ones(int(m.sum())), pull[m]])
            fit = wls(X, res[m], w[m])
            rows_nb.append([head, tag, sub_name, int(m.sum()), f"{fit.beta[1]:+.4f}",
                            f"{fit.se[1]:.4f}", f"{fit.t[1]:+.1f}",
                            f"{wcorr(res[m], pull[m], w[m]):+.3f}"])
out["r17_neighbourhood"] = rows_nb

# per-rare-class residual (fidelity) — is each pulled toward its neighbourhood?
rows_rc = []
for c, v in sorted(rare.items(), key=lambda kv: -kv[1].sum()):
    r = resid["score_play"]
    m = r["have"] & v
    if m.sum() < 20:
        continue
    y = r["y"]
    nb_mean = S.neighbour_mean(y, nb_idx[m])
    rows_rc.append([c, int(m.sum()),
                    f"{np.average(r['f'][m], weights=r['w'][m]) / S.SD['score_play']:+.3f}",
                    f"{np.nanmean(nb_mean - y[m]) / S.SD['score_play']:+.3f}"])
out["r17_rare_class_rows"] = rows_rc

with open(S.OUT / "s_r17.json", "w") as fh:
    json.dump(out, fh, indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:12000])
