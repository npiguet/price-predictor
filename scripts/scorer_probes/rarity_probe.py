"""Join rarity (MTGJSON) onto the label table and test rarity gradients (read-only).

Reads ``labels_joined.csv`` from ``label_probe.py``.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.paths import REPO
from _lib.stats import LinearFit, ols
from scorer_probes import probe_lib as pl

RANK = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}
OLS_FEATURES = ["mv", "power", "tough", "creature", "rar_min", "kw_flying", "removal",
                "counter", "draw", "artifact", "instant", "sorcery", "pips", "colors"]
OLS_NAMES = ["const", "mv", "power", "tough", "creature", "rarity", "flying", "removal",
             "counter", "draw", "artifact", "instant", "sorcery", "pips", "colors"]


def rarity_range() -> dict[str, list[int]]:
    """Per card name, the lowest and highest rarity it was ever printed at (the
    lowest is the limited-relevant one)."""
    best: dict[str, list[int]] = {}
    with open(REPO / "resources" / "AllPrintings.json", encoding="utf-8") as f:
        data = json.load(f)
    for _set, info in data.get("data", {}).items():
        cards = info if isinstance(info, list) else info.get("cards", [])
        for c in cards:
            n = c.get("name", "")
            r = c.get("rarity", "")
            if not n or r not in RANK:
                continue
            prev = best.get(n)
            if prev is None:
                best[n] = [RANK[r], RANK[r]]
            else:
                prev[0] = min(prev[0], RANK[r])
                prev[1] = max(prev[1], RANK[r])
    return best


def print_fit(fit: LinearFit, header: str) -> None:
    print(header.format(n=fit.n, r2=fit.r2))
    for n_, b, s_, t in zip(OLS_NAMES, fit.beta, fit.se, fit.t):
        print(f"  {n_:10s} beta={b:+.5f}  se={s_:.5f}  t={t:+.1f}")


def print_gradients(d: pd.DataFrame) -> None:
    print("\n--- by min-rarity (all nonland cards, n>=200) ---")
    for r, lab in [(0, "common"), (1, "uncommon"), (2, "rare"), (3, "mythic")]:
        s = d[(d.rar_min == r) & (d.land == 0)]
        if len(s) > 20:
            print(f"{lab:9s} n={len(s):5d} sp={s.sp.mean():+.4f} pr={s.pr.mean():.3f} "
                  f"cl={s.cl.mean():+.4f} mv={s.mv.mean():.2f} creat={s.creature.mean():.2f}")

    for title, kind, mvs, mask in (
            ("rarity gradient within creature x MV cells", "creature", [2, 3, 4, 5, 6],
             d.creature == 1),
            ("rarity gradient, noncreature spells", "noncreature", [2, 3, 4],
             (d.creature == 0) & (d.land == 0))):
        print(f"\n--- {title} ---")
        for mv in mvs:
            cells = []
            for r, lab in [(0, "C"), (1, "U"), (2, "R"), (3, "M")]:
                s = d[(d.rar_min == r) & mask & (d.mv == mv)]
                cells.append(f"{lab}:{s.sp.mean():+.4f}(n={len(s)})" if len(s) > 20
                             else f"{lab}:--")
            print(f"{kind} MV{mv}: " + "  ".join(cells))


def main() -> None:
    df = pd.read_csv(pl.SCRATCH / "labels_joined.csv")
    best = rarity_range()
    print("mtgjson names:", len(best))
    df["rar_min"] = df.name.map(lambda n: best.get(n, [np.nan, np.nan])[0])
    df["rar_max"] = df.name.map(lambda n: best.get(n, [np.nan, np.nan])[1])
    d = df[df.rar_min.notna() & df.sp.notna()].copy()
    print("joined w/ rarity:", len(d))
    print_gradients(d)

    print("\n--- OLS: sp ~ mv + power + tough + creature + rarity + flying + removal "
          "(n>=200, nonland) ---")
    sub = d[(d.land == 0)].dropna(subset=["sp", "mv", "creature", "rar_min"]).copy()
    sub["power"] = sub.power.fillna(0)
    sub["tough"] = sub.tough.fillna(0)
    X = np.column_stack([np.ones(len(sub)), sub[OLS_FEATURES].to_numpy(float)])
    print_fit(ols(X, sub.sp.to_numpy(float)), "N={n} R2={r2:.3f}")
    print("\n--- same OLS for played_rate ---")
    print_fit(ols(X, sub.pr.to_numpy(float)), "R2={r2:.3f}")

    print("\n--- split-half style: label reliability by observation count ---")
    for lo, hi in [(200, 500), (500, 1000), (1000, 3000), (3000, 100000)]:
        s = d[(d.n >= lo) & (d.n < hi)]
        print(f"n in [{lo},{hi}): cards={len(s):5d} sd(sp)={s.sp.std():.4f} "
              f"mean|sp|={s.sp.abs().mean():.4f}")
    s = df[(df.n < 200) & df.sp.notna()]
    print(f"n<200 (excluded above): cards={len(s):5d} sd(sp)={s.sp.std():.4f}")


if __name__ == "__main__":
    main()
