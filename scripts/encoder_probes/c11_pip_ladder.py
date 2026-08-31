"""C11 — the pip-intensity ladder: mana value held at 3, generic swapped for pips.

Extends C2's pairwise pip substitutions into a full ladder. For each color C,
real cards printed at {2}{C} are re-encoded at {3}, {2}{C}, {1}{C}{C} and
{C}{C}{C} — the same card at every color intensity an MV-3 cost allows — and
both heads are read as same-card deltas against the printed {2}{C} form.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
from c10_cost_sweep import set_cost  # noqa: E402

SEED = 13
CAP = 300


def arms(color: str) -> list[str]:
    c = f"{{{color}}}"
    return ["{3}", f"{{2}}{c}", f"{{1}}{c}{c}", f"{c}{c}{c}"]


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    cost = np.array([cc.get_field(s, "mana cost:") or "" for s in stripped])
    rng = np.random.default_rng(SEED)
    rows = []
    for color in "WUBRG":
        idx = np.flatnonzero(cost == f"{{2}}{{{color}}}")
        if len(idx) > CAP:
            idx = np.sort(rng.choice(idx, CAP, replace=False))
        print(f"[{color}] {len(idx)} base cards", flush=True)
        texts = [set_cost(stripped[r], a) for r in idx for a in arms(color)]
        emb = cc.encode(texts)
        pred = cc.predict_sd(emb)
        S = pred["score_play"].to_numpy().reshape(len(idx), 4)
        P = pred["played_rate"].to_numpy().reshape(len(idx), 4)
        D = cc.offmanifold(emb).reshape(len(idx), 4)
        for pips in range(4):
            ds = S[:, pips] - S[:, 1]
            dp = P[:, pips] - P[:, 1]
            rows.append({
                "color": color, "n_pips": pips, "n": len(idx),
                "d_score_play_sd": float(ds.mean()),
                "ds_ci_lo": cc.bootstrap_ci(ds)[0],
                "ds_ci_hi": cc.bootstrap_ci(ds)[1],
                "d_played_rate_sd": float(dp.mean()),
                "dp_ci_lo": cc.bootstrap_ci(dp)[0],
                "dp_ci_hi": cc.bootstrap_ci(dp)[1],
                "off_manifold": float((D[:, pips] > cc.MANIFOLD_GATE).mean()),
            })
    df = pd.DataFrame(rows)
    df.to_csv(cc.SCRATCH / "c11_pip_ladder.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
