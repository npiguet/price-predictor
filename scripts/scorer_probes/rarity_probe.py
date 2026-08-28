"""Join rarity (MTGJSON) onto the label table and test rarity gradients (read-only)."""
import json, os
import numpy as np
import pandas as pd

from pathlib import Path
SP = str(Path(__file__).resolve().parents[2] / "output" / "scorer-probes")
df = pd.read_csv(os.path.join(SP, "labels_joined.csv"))

RANK = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
best = {}
with open(str(Path(__file__).resolve().parents[2] / "resources" / "AllPrintings.json"), encoding="utf-8") as f:
    data = json.load(f)
for _set, info in data.get("data", {}).items():
    cards = info if isinstance(info, list) else info.get("cards", [])
    for c in cards:
        n = c.get("name", "")
        r = c.get("rarity", "")
        if not n or r not in RANK:
            continue
        # keep the *lowest* rarity a card was ever printed at (limited-relevant), and also max
        prev = best.get(n)
        if prev is None:
            best[n] = [RANK[r], RANK[r]]
        else:
            prev[0] = min(prev[0], RANK[r]); prev[1] = max(prev[1], RANK[r])
del data
print("mtgjson names:", len(best))

df["rar_min"] = df.name.map(lambda n: best.get(n, [np.nan, np.nan])[0])
df["rar_max"] = df.name.map(lambda n: best.get(n, [np.nan, np.nan])[1])
d = df[df.rar_min.notna() & df.sp.notna()].copy()
print("joined w/ rarity:", len(d))

print("\n--- by min-rarity (all nonland cards, n>=200) ---")
for r, lab in [(0, "common"), (1, "uncommon"), (2, "rare"), (3, "mythic")]:
    s = d[(d.rar_min == r) & (d.land == 0)]
    if len(s) > 20:
        print(f"{lab:9s} n={len(s):5d} sp={s.sp.mean():+.4f} pr={s.pr.mean():.3f} cl={s.cl.mean():+.4f} mv={s.mv.mean():.2f} creat={s.creature.mean():.2f}")

print("\n--- rarity gradient within creature x MV cells ---")
for mv in [2, 3, 4, 5, 6]:
    cells = []
    for r, lab in [(0, "C"), (1, "U"), (2, "R"), (3, "M")]:
        s = d[(d.rar_min == r) & (d.creature == 1) & (d.mv == mv)]
        cells.append(f"{lab}:{s.sp.mean():+.4f}(n={len(s)})" if len(s) > 20 else f"{lab}:--")
    print(f"creature MV{mv}: " + "  ".join(cells))

print("\n--- rarity gradient, noncreature spells ---")
for mv in [2, 3, 4]:
    cells = []
    for r, lab in [(0, "C"), (1, "U"), (2, "R"), (3, "M")]:
        s = d[(d.rar_min == r) & (d.creature == 0) & (d.land == 0) & (d.mv == mv)]
        cells.append(f"{lab}:{s.sp.mean():+.4f}(n={len(s)})" if len(s) > 20 else f"{lab}:--")
    print(f"noncreature MV{mv}: " + "  ".join(cells))

print("\n--- OLS: sp ~ mv + power + tough + creature + rarity + flying + removal (n>=200, nonland) ---")
sub = d[(d.land == 0)].dropna(subset=["sp", "mv", "creature", "rar_min"]).copy()
sub["power"] = sub.power.fillna(0); sub["tough"] = sub.tough.fillna(0)
X = sub[["mv", "power", "tough", "creature", "rar_min", "kw_flying", "removal", "counter", "draw",
         "artifact", "instant", "sorcery", "pips", "colors"]].to_numpy(float)
X = np.column_stack([np.ones(len(X)), X])
y = sub.sp.to_numpy(float)
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
names = ["const", "mv", "power", "tough", "creature", "rarity", "flying", "removal", "counter",
         "draw", "artifact", "instant", "sorcery", "pips", "colors"]
resid = y - X @ beta
se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid @ resid) / (len(y) - X.shape[1]))
print(f"N={len(y)} R2={1 - resid.var()/y.var():.3f}")
for n_, b, s_ in zip(names, beta, se):
    print(f"  {n_:10s} beta={b:+.5f}  se={s_:.5f}  t={b/s_:+.1f}")

print("\n--- same OLS for played_rate ---")
y2 = sub.pr.to_numpy(float)
beta2, *_ = np.linalg.lstsq(X, y2, rcond=None)
resid2 = y2 - X @ beta2
se2 = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * (resid2 @ resid2) / (len(y2) - X.shape[1]))
print(f"R2={1 - resid2.var()/y2.var():.3f}")
for n_, b, s_ in zip(names, beta2, se2):
    print(f"  {n_:10s} beta={b:+.5f}  se={s_:.5f}  t={b/s_:+.1f}")

print("\n--- split-half style: label reliability by observation count ---")
for lo, hi in [(200, 500), (500, 1000), (1000, 3000), (3000, 100000)]:
    s = d[(d.n >= lo) & (d.n < hi)]
    print(f"n in [{lo},{hi}): cards={len(s):5d} sd(sp)={s.sp.std():.4f} mean|sp|={s.sp.abs().mean():.4f}")
s = df[(df.n < 200) & df.sp.notna()]
print(f"n<200 (excluded above): cards={len(s):5d} sd(sp)={s.sp.std():.4f}")
