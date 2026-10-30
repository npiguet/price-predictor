"""Human draft pick-order (Forge .rnk) vs Forge-AI empirical labels (read-only).

Reads ``labels_joined.csv`` from ``label_probe.py`` and the rankings through
``forge_hints``, normalized the way Forge normalizes them.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorer_probes import forge_hints
from scorer_probes import probe_lib as pl

MIN_SET_SIZE = 100  # smaller rankings are partial lists, not a set's pick order
LABEL_COLUMNS = ["sp", "pr", "cl", "n", "mv", "creature", "removal", "counter", "kw_flying",
                 "instant", "sorcery", "artifact", "land", "power", "tough", "draw", "trick"]


def ranking_table() -> pd.DataFrame:
    """One row per ranked card per set; ``pct`` is 1 - Forge's normalized rank,
    so 1.0 is the best human pick."""
    entries, _ = forge_hints.read_ranking_entries()
    sizes = forge_hints.edition_sizes(entries)
    r = pd.DataFrame([{"set": e.edition.upper(), "rank": e.rank, "n_set": sizes[e.edition],
                       "name": e.name.lower(), "rarity": e.rarity.upper()}
                      for e in entries if sizes[e.edition] >= MIN_SET_SIZE])
    r["pct"] = 1.0 - r["rank"] / r["n_set"]
    return r


def print_within_set(j: pd.DataFrame) -> None:
    print("\n--- within-set Spearman(human pick pct, label) ---")
    for col in ["sp", "pr", "cl"]:
        rs = [stats.spearmanr(g.pct, g[col]).statistic
              for _, g in j.groupby("set") if len(g) >= 80]
        print(f"{col}: median within-set rho = {np.median(rs):+.3f}  (sets={len(rs)}, IQR "
              f"{np.percentile(rs, 25):+.3f}..{np.percentile(rs, 75):+.3f})")


def print_residuals(j: pd.DataFrame) -> None:
    """Regress sp on human pct within set (z-scored per set) and show which
    categories the AI meta over- or under-rates against human rank."""
    print("\n--- residual: which categories does the AI meta over/under-rate vs human rank? ---")
    j = j.copy()
    j["sp_z"] = j.groupby("set").sp.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    j["pct_z"] = j.groupby("set").pct.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    b = np.polyfit(j.pct_z, j.sp_z, 1)
    j["resid"] = j.sp_z - np.polyval(b, j.pct_z)
    print(f"slope(sp_z ~ human pct_z) = {b[0]:+.3f}   "
          f"(r = {stats.pearsonr(j.pct_z, j.sp_z).statistic:+.3f})")
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


def main() -> None:
    lab = pd.read_csv(pl.SCRATCH / "labels_joined.csv")
    lab["key"] = lab.name.str.lower()
    r = ranking_table()
    print("rank rows:", len(r), "sets:", r.set.nunique())

    j = r.join(lab.set_index("key")[LABEL_COLUMNS], on="name", how="inner")
    j = j[j.n >= 200].dropna(subset=["sp"])
    print("joined rows (n>=200):", len(j), "unique cards:", j.name.nunique())
    print_within_set(j)

    print("\n--- human top-decile vs bottom-half labels ---")
    top = j[j.pct >= 0.9]
    bot = j[j.pct <= 0.5]
    print(f"human top 10%: n={len(top):5d} sp={top.sp.mean():+.4f} "
          f"creature={top.creature.mean():.2f} mv={top.mv.mean():.2f}")
    print(f"human bot 50%: n={len(bot):5d} sp={bot.sp.mean():+.4f} "
          f"creature={bot.creature.mean():.2f} mv={bot.mv.mean():.2f}")
    print_residuals(j)


if __name__ == "__main__":
    main()
