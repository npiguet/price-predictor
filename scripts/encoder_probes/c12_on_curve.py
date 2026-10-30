"""C12 — the encoder's mana curve: the best-read cost for each vanilla statline.

For every statline P/T (P 0–8, T 1–8) written onto real vanilla creature
bodies, the cost is swept from {M} to {9}{M} (one pip of the card's own color
kept, so mana value runs 1–10), and the winnability head is read at each
step. The mana value that maximizes predicted winnability for a statline is
the encoder's idea of that statline being on curve.

The best-value table is the unrestricted argmax over the full sweep; each
(statline, mana value) cell also records how many real vanilla creatures
occupy it (``n_real``), for readers who want to know which cells the game
has actually printed.

Vanilla here is strict: exactly the three core lines (mana cost, types,
power toughness). The looser ``no ability lines`` test used elsewhere in
the battery lets through cards whose converted text carries uncounted
``text:`` / ``alternate cost:`` / ``additional cost:`` lines (Myr Superion,
Scornful Egotist, Bayou Groff).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402

SEED = 17
N_BASES = 100
CORE = ("mana cost:", "types:", "power toughness:")
POWERS = list(range(0, 9))
TOUGHS = list(range(1, 9))
GENERIC = list(range(0, 10))


def main() -> None:
    j = pl.load_frame(text=True)
    C = cc.CardCorpus.of(j)
    stripped, parsed, power, tough = C.text, C.parsed_cost, C.power, C.toughness
    literal_creature = C.is_creature & C.literal_pt
    strict_vanilla = np.array([
        len(ls := cc.lines(s)) == 3
        and all(any(ln.startswith(p) for ln in ls) for p in CORE)
        for s in stripped]) & literal_creature
    loose = literal_creature & (C.n_abilities == 0)
    print(f"{int(strict_vanilla.sum())} strict-vanilla creatures "
          f"({int((loose & ~strict_vanilla).sum())} looser no-ability cards "
          "excluded for text:/alternate cost:/additional cost: lines)",
          flush=True)

    # how many real strict-vanilla creatures occupy each (P, T, MV) cell
    mv = pl.num(j, "mv")
    n_real: Counter = Counter()
    for r in np.flatnonzero(strict_vanilla):
        if parsed[r] is None or not np.isfinite(power[r]) or not np.isfinite(tough[r]):
            continue
        n_real[(int(power[r]), int(tough[r]), int(mv[r]))] += 1

    elig = cc.sample_rows(strict_vanilla & np.array([p is not None and p[1] != ""
                                                     for p in parsed]),
                          N_BASES, np.random.default_rng(SEED))
    pip = [re.match(r"\{([WUBRG])\}", parsed[r][1]).group(1) for r in elig]
    print(f"{len(elig)} vanilla bases: {list(j.loc[elig, 'name'][:8])}",
          flush=True)

    texts = []
    for r, m in zip(elig, pip):
        for p in POWERS:
            for t in TOUGHS:
                body = cc.set_pt(stripped[r], p, t)
                for g in GENERIC:
                    c = f"{{{g}}}{{{m}}}" if g else f"{{{m}}}"
                    texts.append(cc.set_cost(body, c))
    scored = pl.score_texts(texts).reshape(len(elig), len(POWERS), len(TOUGHS), len(GENERIC))

    rows = []
    for pi, p in enumerate(POWERS):
        for ti, t in enumerate(TOUGHS):
            for gi, g in enumerate(GENERIC):
                rows.append({
                    "power": p, "toughness": t, "mv": g + 1,
                    "score_play_sd": float(scored.sp[:, pi, ti, gi].mean()),
                    "off_manifold": float(scored.off[:, pi, ti, gi].mean()),
                    "n_real": n_real.get((p, t, g + 1), 0),
                })
    df = pd.DataFrame(rows)
    df.to_csv(pl.SCRATCH / "c12_on_curve.csv", index=False)

    best = (df.loc[df.groupby(["power", "toughness"])["score_play_sd"].idxmax()]
            .pivot(index="power", columns="toughness", values="mv"))
    print("best MV by statline (rows P, cols T):", flush=True)
    print(best.to_string(), flush=True)
    best.to_csv(pl.SCRATCH / "c12_best_mv.csv")


if __name__ == "__main__":
    main()
