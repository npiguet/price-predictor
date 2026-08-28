"""Human draft pick-order (Forge .rnk) vs Forge-AI empirical labels (read-only)."""
import os, glob
import numpy as np
import pandas as pd
from scipy import stats

from pathlib import Path
SP = str(Path(__file__).resolve().parents[2] / "output" / "scorer-probes")
RNK = str(Path(__file__).resolve().parents[2].parent / "forge" / "forge-gui" / "res" / "draft" / "rankings")

lab = pd.read_csv(os.path.join(SP, "labels_joined.csv"))
lab["key"] = lab.name.str.lower()
lmap = lab.set_index("key")

rows = []
for path in glob.glob(os.path.join(RNK, "*.rnk")):
    setc = os.path.basename(path)[:-4].upper()
    lines = [l.strip() for l in open(path, encoding="utf-8", errors="replace") if l.strip() and not l.startswith("//")]
    n = len(lines)
    if n < 100:
        continue
    for l in lines:
        parts = l.split("|")
        if len(parts) < 3:
            continue
        try:
            rank = int(parts[0].lstrip("#"))
        except ValueError:
            continue
        rows.append({"set": setc, "rank": rank, "n_set": n, "name": parts[1].strip().lower(),
                     "rarity": parts[2].strip().upper()})
r = pd.DataFrame(rows)
r["pct"] = 1.0 - (r["rank"] - 1) / (r["n_set"] - 1)   # 1.0 = best human pick
print("rank rows:", len(r), "sets:", r.set.nunique())

j = r.join(lmap[["sp", "pr", "cl", "n", "mv", "creature", "removal", "counter", "kw_flying",
                 "instant", "sorcery", "artifact", "land", "power", "tough", "draw", "trick"]],
           on="name", how="inner")
j = j[j.n >= 200].dropna(subset=["sp"])
print("joined rows (n>=200):", len(j), "unique cards:", j.name.nunique())

print("\n--- within-set Spearman(human pick pct, label) ---")
for col in ["sp", "pr", "cl"]:
    rs = []
    for s, g in j.groupby("set"):
        if len(g) >= 80:
            rs.append(stats.spearmanr(g.pct, g[col]).statistic)
    print(f"{col}: median within-set rho = {np.median(rs):+.3f}  (sets={len(rs)}, IQR "
          f"{np.percentile(rs,25):+.3f}..{np.percentile(rs,75):+.3f})")

print("\n--- human top-decile vs bottom-half labels ---")
top = j[j.pct >= 0.9]; bot = j[j.pct <= 0.5]
print(f"human top 10%: n={len(top):5d} sp={top.sp.mean():+.4f} creature={top.creature.mean():.2f} mv={top.mv.mean():.2f}")
print(f"human bot 50%: n={len(bot):5d} sp={bot.sp.mean():+.4f} creature={bot.creature.mean():.2f} mv={bot.mv.mean():.2f}")

print("\n--- residual: which categories does the AI meta over/under-rate vs human rank? ---")
# regress sp on human pct within set (z-scored per set), inspect category residual means
j = j.copy()
j["sp_z"] = j.groupby("set").sp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
j["pct_z"] = j.groupby("set").pct.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
b = np.polyfit(j.pct_z, j.sp_z, 1)
j["resid"] = j.sp_z - np.polyval(b, j.pct_z)
print(f"slope(sp_z ~ human pct_z) = {b[0]:+.3f}   (r = {stats.pearsonr(j.pct_z, j.sp_z).statistic:+.3f})")
cats = {
    "creature": j.creature == 1,
    "noncreature spell": (j.creature == 0) & (j.land == 0),
    "creature w/ flying": (j.creature == 1) & (j.kw_flying == 1),
    "noncreature removal": (j.creature == 0) & (j.removal == 1) & (j.land == 0),
    "counterspell": j.counter == 1,
    "combat trick": j.trick == 1,
    "card draw (noncreature)": (j.creature == 0) & (j.draw == 1) & (j.land == 0),
    "artifact (noncreature)": (j.artifact == 1) & (j.creature == 0),
    "rare/mythic (rnk R/M)": j.rarity.isin(["R", "M"]),
    "common (rnk C)": j.rarity == "C",
    "MV>=6": j.mv >= 6,
    "MV<=2": j.mv <= 2,
}
for k, m in cats.items():
    s = j[m]
    if len(s) > 40:
        print(f"  {k:26s} n={len(s):5d} mean resid = {s.resid.mean():+.3f} sd-units")
