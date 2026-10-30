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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from c10_cost_sweep import head_ci  # noqa: E402

SEED = 13
CAP = 300
CLASSES = ("vanilla creature", "creature with text", "noncreature spell",
           "other permanent")


def arms(g: int, color: str) -> list[str]:
    m = f"{{{color}}}"
    return [cc.fmt_cost(g + 1, ""), cc.fmt_cost(g, m), cc.fmt_cost(g - 1, m * 2),
            cc.fmt_cost(g - 2, m * 3)]


def main() -> None:
    C = cc.CardCorpus.of(pl.load_frame(text=True))
    is_crea, parsed = C.is_creature, C.parsed_cost
    types = [cc.get_field(s, "types:") or "" for s in C.text]
    is_spell = np.array([("instant" in t or "sorcery" in t) for t in types])
    single_pip = np.array([
        p is not None and p[1].count("{") == 1 and 2 <= p[0] <= 6
        for p in parsed])
    card_class = np.where(is_crea & (C.n_abilities == 0), "vanilla creature",
                          np.where(is_crea, "creature with text",
                                   np.where(is_spell, "noncreature spell",
                                            "other permanent")))
    rng = np.random.default_rng(SEED)

    picked: list[np.ndarray] = []
    for cname in CLASSES:
        idx = cc.sample_rows(single_pip & (card_class == cname), CAP, rng)
        print(f"[{cname}] {len(idx)} base cards", flush=True)
        picked.append(idx)
    idx_all = np.concatenate(picked)
    cls = card_class[idx_all]
    color = np.array([parsed[r][1][1] for r in idx_all])

    # the four intensity arms differ per card, so the arm index picks from its own list
    L = pl.score_arms(idx_all, range(4), lambda r, k: cc.set_cost(
        C.text[r], arms(parsed[r][0], parsed[r][1][1])[k]))

    groups = [(c, "all", cls == c) for c in CLASSES]
    groups += [("all", m, color == m) for m in "WUBRG"]
    groups.append(("all", "all", np.ones(len(idx_all), bool)))
    rows = []
    for cname, cl, m in groups:
        for pips in range(4):
            ds = L.sp[m, pips] - L.sp[m, 1]
            dp = L.pr[m, pips] - L.pr[m, 1]
            rows.append({
                "class": cname, "color": cl, "n_pips": pips, "n": int(m.sum()),
                "d_score_play_sd": float(ds.mean()), **head_ci(ds, "ds_ci_lo", "ds_ci_hi"),
                "d_played_rate_sd": float(dp.mean()), **head_ci(dp, "dp_ci_lo", "dp_ci_hi"),
                "off_manifold": float(L.off[m][:, pips].mean()),
            })
    pl.write_table(rows, "c11_pip_ladder.csv")


if __name__ == "__main__":
    main()
