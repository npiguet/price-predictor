"""R8(b) standard errors — bootstrap the MV×type-matched played_rate deficits."""

from __future__ import annotations

import json

import numpy as np

import probe_lib as pl
import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]
pf = d["pf"]
pred_f = pl.predict_labels(emb, pf)


def col(c):
    return np.nan_to_num(S.num(join, c))


mv = col("mv")
y = S.num(join, "shrunk_played_rate")
w = S.num(join, "w_played_rate")
type_cell = np.select(
    [col("is_creature") > 0, col("is_land") > 0, col("is_instant") > 0,
     col("is_sorcery") > 0, col("is_artifact") > 0, col("is_enchantment") > 0],
    [0, 1, 2, 3, 4, 5], default=6)
cell = np.clip(mv, 0, 8).astype(int) * 10 + type_cell
classes = {
    "fog / prevent damage": (col("ph_fog") + col("ph_prevent_damage")) > 0,
    "sweeper": col("ph_sweeper") > 0,
    "counterspell": col("ph_counterspell") > 0,
    "morph": col("ph_morph") > 0,
    "mana rock": (col("is_artifact") > 0) & (col("is_creature") == 0)
                 & (col("ph_tap_for_mana") > 0),
}


def deficit(flag, vals, wts, idx):
    tot = totw = 0.0
    c = cell[idx]
    f = flag[idx]
    v = vals[idx]
    ww = wts[idx]
    for cv in np.unique(c):
        m = c == cv
        a, b = m & f, m & ~f
        if a.sum() == 0 or b.sum() < 5:
            continue
        k = min(a.sum(), b.sum())
        tot += k * (np.average(v[a], weights=ww[a]) - np.average(v[b], weights=ww[b]))
        totw += k
    return tot / totw if totw else np.nan


rng = np.random.default_rng(42)
n = len(join)
allidx = np.arange(n)
out = []
for name, flag in classes.items():
    row = [name, int(flag.sum())]
    for tag, vals in (("label", y), ("fidelity pred", np.asarray(pred_f["played_rate"], float))):
        point = deficit(flag, vals, w, allidx)
        boots = np.array([deficit(flag, vals, w, rng.integers(0, n, n)) for _ in range(200)])
        se = float(np.nanstd(boots))
        row += [f"{point:+.4f}", f"{se:.4f}", f"{point / S.SD['played_rate']:+.2f}",
                f"{se / S.SD['played_rate']:.2f}"]
    out.append(row)
    print(row, flush=True)

with open(S.OUT / "s_r8c.json", "w") as f:
    json.dump({"matched_deficits_with_se": out}, f, indent=1)
