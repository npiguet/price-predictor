"""C11 — the pip-intensity ladder: mana value held fixed, generic swapped for pips.

Extends C2's pairwise pip substitutions into a full ladder. Base cards are
real cards whose printed cost is a single colored pip plus generic {k}{M},
k 2–6 ({M} = one fixed color W–G; not the colorless symbol {C}). Each is
re-encoded at the four color intensities its own mana value allows —
{k+1}, {k}{M}, {k-1}{M}{M}, {k-2}{M}{M}{M} — and both heads are read as
same-card deltas against the printed single-pip form.

Rows are written per card class (vanilla creature / creature with text /
noncreature spell / other permanent, the generic sweep's classes) pooled
over colors, plus pooled-class rows per color (``class`` = ``all``) and one
grand row (``class`` = ``all``, ``color`` = ``all``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
from c10_cost_sweep import fmt_cost, parse_cost, set_cost  # noqa: E402

SEED = 13
CAP = 300
CLASSES = ("vanilla creature", "creature with text", "noncreature spell",
           "other permanent")


def arms(g: int, color: str) -> list[str]:
    m = f"{{{color}}}"
    return [fmt_cost(g + 1, ""), fmt_cost(g, m), fmt_cost(g - 1, m * 2),
            fmt_cost(g - 2, m * 3)]


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    types = [cc.get_field(s, "types:") or "" for s in stripped]
    is_spell = np.array([("instant" in t or "sorcery" in t) for t in types])
    n_abil = np.array([len(cc.ability_lines(s)) for s in stripped])
    cost = [cc.get_field(s, "mana cost:") for s in stripped]
    parsed = [parse_cost(c) if c else None for c in cost]
    single_pip = np.array([
        p is not None and p[1].count("{") == 1 and 2 <= p[0] <= 6
        for p in parsed])
    card_class = np.where(is_crea & (n_abil == 0), "vanilla creature",
                          np.where(is_crea, "creature with text",
                                   np.where(is_spell, "noncreature spell",
                                            "other permanent")))
    rng = np.random.default_rng(SEED)

    picked: list[np.ndarray] = []
    for cname in CLASSES:
        idx = np.flatnonzero(single_pip & (card_class == cname))
        if len(idx) > CAP:
            idx = np.sort(rng.choice(idx, CAP, replace=False))
        print(f"[{cname}] {len(idx)} base cards", flush=True)
        picked.append(idx)
    idx_all = np.concatenate(picked)
    cls = card_class[idx_all]
    color = np.array([parsed[r][1][1] for r in idx_all])

    texts = [set_cost(stripped[r], a)
             for r in idx_all
             for a in arms(parsed[r][0], parsed[r][1][1])]
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    S = pred["score_play"].to_numpy().reshape(len(idx_all), 4)
    P = pred["played_rate"].to_numpy().reshape(len(idx_all), 4)
    D = cc.offmanifold(emb).reshape(len(idx_all), 4)

    groups = [(c, "all", cls == c) for c in CLASSES]
    groups += [("all", m, color == m) for m in "WUBRG"]
    groups.append(("all", "all", np.ones(len(idx_all), bool)))
    rows = []
    for cname, cl, m in groups:
        for pips in range(4):
            ds = S[m, pips] - S[m, 1]
            dp = P[m, pips] - P[m, 1]
            rows.append({
                "class": cname, "color": cl, "n_pips": pips, "n": int(m.sum()),
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
