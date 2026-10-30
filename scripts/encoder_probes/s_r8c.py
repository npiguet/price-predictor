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


y = S.num(join, "shrunk_played_rate")
w = S.num(join, "w_played_rate")
cell = S.control_cells(join)
classes = S.hostile_classes(join)


def deficit(flag, vals, wts, idx):
    return S.matched_deficit(cell[idx], flag[idx], vals[idx], wts[idx])[0]


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
