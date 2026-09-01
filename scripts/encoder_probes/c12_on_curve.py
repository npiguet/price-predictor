"""C12 — the encoder's mana curve: the best-read cost for each vanilla statline.

For every statline P/T (P 0–5, T 1–6) written onto real vanilla creature
bodies, the cost is swept from {M} to {7}{M} (one pip of the card's own color
kept, so mana value runs 1–8), and the winnability head is read at each step.
The mana value that maximizes predicted winnability for a statline is the
encoder's idea of that statline being on curve.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
from c10_cost_sweep import parse_cost, set_cost  # noqa: E402
from c2_statlines import PT_RE, set_pt  # noqa: E402

SEED = 17
N_BASES = 30
POWERS = list(range(0, 6))
TOUGHS = list(range(1, 7))
GENERIC = list(range(0, 8))


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    cost = [cc.get_field(s, "mana cost:") for s in stripped]
    parsed = [parse_cost(c) if c else None for c in cost]
    lit_pt = np.array([bool(PT_RE.match(l))
                       for s in stripped
                       for l in [next((x for x in cc.lines(s)
                                       if x.startswith("power toughness:")), "")]])
    elig = np.flatnonzero(
        is_crea & lit_pt
        & np.array([len(cc.ability_lines(s)) == 0 for s in stripped])
        & np.array([p is not None and p[1] != "" for p in parsed]))
    rng = np.random.default_rng(SEED)
    if len(elig) > N_BASES:
        elig = np.sort(rng.choice(elig, N_BASES, replace=False))
    pip = [re.match(r"\{([WUBRG])\}", parsed[r][1]).group(1) for r in elig]
    print(f"{len(elig)} vanilla bases: {list(j.loc[elig, 'name'][:8])}",
          flush=True)

    texts = []
    for r, m in zip(elig, pip):
        for p in POWERS:
            for t in TOUGHS:
                body = set_pt(stripped[r], p, t)
                for g in GENERIC:
                    c = f"{{{g}}}{{{m}}}" if g else f"{{{m}}}"
                    texts.append(set_cost(body, c))
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    shape = (len(elig), len(POWERS), len(TOUGHS), len(GENERIC))
    S = pred["score_play"].to_numpy().reshape(shape)
    D = cc.offmanifold(emb).reshape(shape)

    rows = []
    for pi, p in enumerate(POWERS):
        for ti, t in enumerate(TOUGHS):
            for gi, g in enumerate(GENERIC):
                rows.append({
                    "power": p, "toughness": t, "mv": g + 1,
                    "score_play_sd": float(S[:, pi, ti, gi].mean()),
                    "off_manifold": float(
                        (D[:, pi, ti, gi] > cc.MANIFOLD_GATE).mean()),
                })
    df = pd.DataFrame(rows)
    df.to_csv(cc.SCRATCH / "c12_on_curve.csv", index=False)

    best = (df.loc[df.groupby(["power", "toughness"])["score_play_sd"].idxmax()]
            .pivot(index="power", columns="toughness", values="mv"))
    print("best MV by statline (rows P, cols T):", flush=True)
    print(best.to_string(), flush=True)
    best.to_csv(cc.SCRATCH / "c12_best_mv.csv")


if __name__ == "__main__":
    main()
