"""R8(b) robustness — is the five-class 'AI-hostile' direction test powered?

A cosine of 0.05 between two 512-dim class directions only means "no shared
axis" if the *same* class split in half reproduces itself well above that.
Also runs the transfer test in prediction space, which does not depend on
the direction vector's noisy tail.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

import s_common as S

d = S.load_all()
join, emb = d["join"], d["emb"]


def col(c):
    return np.nan_to_num(S.num(join, c))


classes = {
    "fog / prevent damage": (col("ph_fog") + col("ph_prevent_damage")) > 0,
    "sweeper": col("ph_sweeper") > 0,
    "counterspell": col("ph_counterspell") > 0,
    "morph": col("ph_morph") > 0,
    "mana rock": (col("is_artifact") > 0) & (col("is_creature") == 0)
                 & (col("ph_tap_for_mana") > 0),
}
names = list(classes)
rng = np.random.default_rng(42)
out: dict = {}


def direction(pos_mask):
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    clf.fit(emb, pos_mask.astype(int))
    v = clf.coef_[0]
    return v / np.linalg.norm(v), clf


# split-half reliability of each class direction
rows_rel = []
halves = {}
for n in names:
    idx = np.where(classes[n])[0]
    perm = rng.permutation(len(idx))
    h1, h2 = idx[perm[: len(idx) // 2]], idx[perm[len(idx) // 2:]]
    m1 = np.zeros(len(join), bool); m1[h1] = True
    m2 = np.zeros(len(join), bool); m2[h2] = True
    v1, _ = direction(m1)
    v2, _ = direction(m2)
    halves[n] = (v1, v2)
    rows_rel.append([n, int(classes[n].sum()), f"{float(v1 @ v2):+.3f}"])
out["split_half_cosine"] = rows_rel

# null: cosine between directions of two random same-sized card sets
null = []
for n in names:
    k = int(classes[n].sum())
    a = np.zeros(len(join), bool); a[rng.choice(len(join), k, replace=False)] = True
    b = np.zeros(len(join), bool); b[rng.choice(len(join), k, replace=False)] = True
    va, _ = direction(a)
    vb, _ = direction(b)
    null.append([n, k, f"{float(va @ vb):+.3f}"])
out["random_set_cosine"] = null

# transfer in prediction space: score class B's cards with class A's direction
full = {n: direction(classes[n]) for n in names}
mv = np.nan_to_num(S.num(join, "mv"))
type_cell = np.select(
    [col("is_creature") > 0, col("is_land") > 0, col("is_instant") > 0,
     col("is_sorcery") > 0, col("is_artifact") > 0, col("is_enchantment") > 0],
    [0, 1, 2, 3, 4, 5], default=6)
cell = np.clip(mv, 0, 8).astype(int) * 10 + type_cell

rows_tr = []
for a in names:
    va = full[a][0]
    proj = emb @ va
    row = [a]
    for b in names:
        # matched mean projection difference, class b vs its MV×type controls
        tot = totw = 0.0
        for cv in np.unique(cell):
            m = cell == cv
            p, q = m & classes[b], m & ~classes[b]
            if p.sum() == 0 or q.sum() < 5:
                continue
            wgt = min(p.sum(), q.sum())
            tot += wgt * (proj[p].mean() - proj[q].mean())
            totw += wgt
        row.append(f"{tot / totw / proj.std():+.2f}" if totw else "—")
    rows_tr.append(row)
out["transfer_matrix_sd_units"] = rows_tr
out["transfer_names"] = names

with open(S.OUT / "s_r8b.json", "w") as f:
    json.dump(out, f, indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
