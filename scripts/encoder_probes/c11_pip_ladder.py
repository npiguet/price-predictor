"""C11 — the pip-intensity ladder: mana value held at 3, generic swapped for pips.

Extends C2's pairwise pip substitutions into a full ladder. For each color M,
real cards printed at {2}{M} are re-encoded at {3}, {2}{M}, {1}{M}{M} and
{M}{M}{M} — the same card at every color intensity an MV-3 cost allows — and
both heads are read as same-card deltas against the printed {2}{M} form.
({M} here stands for one fixed color W–G; it is not the colorless symbol {C}.)

Rows are written once aggregated over all base cards per color (``class`` =
``all``, the document's headline numbers) and once per card class (creature /
noncreature spell / other permanent), from the same sampled bases.
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
    m = f"{{{color}}}"
    return ["{3}", f"{{2}}{m}", f"{{1}}{m}{m}", f"{m}{m}{m}"]


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    cost = np.array([cc.get_field(s, "mana cost:") or "" for s in stripped])
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    types = [cc.get_field(s, "types:") or "" for s in stripped]
    is_spell = np.array([("instant" in t or "sorcery" in t) for t in types])
    card_class = np.where(is_crea, "creature",
                          np.where(is_spell, "noncreature spell",
                                   "other permanent"))
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
        cls = card_class[idx]
        groups = [("all", np.ones(len(idx), bool))] + [
            (c, cls == c) for c in
            ("creature", "noncreature spell", "other permanent")]
        for cname, m in groups:
            if m.sum() < 20:
                continue
            for pips in range(4):
                ds = S[m, pips] - S[m, 1]
                dp = P[m, pips] - P[m, 1]
                rows.append({
                    "class": cname, "color": color, "n_pips": pips,
                    "n": int(m.sum()),
                    "d_score_play_sd": float(ds.mean()),
                    "ds_ci_lo": cc.bootstrap_ci(ds)[0],
                    "ds_ci_hi": cc.bootstrap_ci(ds)[1],
                    "d_played_rate_sd": float(dp.mean()),
                    "dp_ci_lo": cc.bootstrap_ci(dp)[0],
                    "dp_ci_hi": cc.bootstrap_ci(dp)[1],
                    "off_manifold": float((D[m][:, pips] > cc.MANIFOLD_GATE).mean()),
                })
    df = pd.DataFrame(rows)
    df.to_csv(cc.SCRATCH / "c11_pip_ladder.csv", index=False)
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
