"""C5 (R16) — types and flavour, all as within-line token substitutions.

(i)   the tribal-noun pairwise matrix, reduced to an additive tribal scale
      the same way C1 reduces the keyword matrix;
(ii)  type-line swaps: instant <-> sorcery, creature -> artifact creature
      (mechanism demos — the type token is the only thing that moves);
(iii) taplands: ``tapped -> untapped`` inside the replacement line (clean),
      alongside deleting the line outright (the real-card form, which
      carries the line-deletion artifact).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from _lib.stats import bootstrap_ci  # noqa: E402
from c1_keywords import bootstrap_scales, fit_scale  # noqa: E402

MAX_PER_TRIBE = 250
TAPPED = "replacement: CARDNAME enters tapped."


def main() -> None:
    j = pl.load_frame(text=True)
    C = cc.CardCorpus.of(j)
    stripped = C.text
    rng = np.random.default_rng(13)
    out: dict = {}
    freq = pl.subtype_frequencies()

    # (i) tribal pairwise matrix
    info = [cc.sole_tribe(s) if c else None for s, c in zip(stripped, C.is_creature)]
    tribe = np.array([v[1] if v else "" for v in info])
    base = np.concatenate([
        cc.sample_rows(tribe == t, MAX_PER_TRIBE, rng)
        for t in cc.TRIBES if (tribe == t).sum() >= 15])
    own = tribe[base]
    present = [t for t in cc.TRIBES if (own == t).sum() >= 15]
    print(f"[C5] tribal base {len(base)} over {len(present)} tribes: {present}",
          flush=True)

    def swap_tribe(r: int, new: str) -> str:
        i, old = info[r]
        words = cc.lines(stripped[r])[i][6:].split()
        return cc.replace_in_line(stripped[r], i,
                                  "types: " + " ".join(new if w == old else w for w in words))

    T = pl.score_arms(base, present, swap_tribe)
    groups = [np.flatnonzero(own == t) for t in present]

    prs = []
    for gi, t in enumerate(present):
        sel = groups[gi]
        for gj, tp in enumerate(present):
            if gi == gj:
                continue
            prs.append({"from": t, "to": tp, "n": len(sel),
                        "delta_sp": float((T.sp[sel, gj] - T.sp[sel, gi]).mean()),
                        "delta_pr": float((T.pr[sel, gj] - T.pr[sel, gi]).mean())})
    pairs = pd.DataFrame(prs)
    pairs.to_csv(pl.SCRATCH / "c5_tribe_pairwise.csv", index=False)
    scale = fit_scale(pairs, present, "delta_sp")
    scale_pr = fit_scale(pairs, present, "delta_pr")
    (boot,) = bootstrap_scales([T.sp], groups, present, 400, seed=17)

    tri = pd.DataFrame({
        "tribe": present,
        "n_carriers": [len(g) for g in groups],
        "corpus_freq": [freq.get(t, 0) for t in present],
        "value_sp": [scale[t] for t in present],
        "ci_lo": np.quantile(boot, 0.025, axis=0),
        "ci_hi": np.quantile(boot, 0.975, axis=0),
        "value_pr": [scale_pr[t] for t in present],
    }).sort_values("value_sp", ascending=False)
    pl.write_table(tri, "c5_tribe_scale.csv")
    out["tribe_off_manifold"] = float(T.off.mean())
    out["tribe_spread_sd"] = float(tri["value_sp"].max() - tri["value_sp"].min())

    # the two headline directions, measured directly
    for a, b in (("dragon", "lizard"), ("angel", "bird"), ("dragon", "human"),
                 ("human", "dragon"), ("angel", "human"), ("wall", "human")):
        if a not in present or b not in present:
            continue
        sel = groups[present.index(a)]
        d = T.sp[sel, present.index(b)] - T.sp[sel, present.index(a)]
        out[f"{a}->{b}"] = {"n": int(len(sel)), "mean_sd": float(d.mean()),
                            "ci": list(bootstrap_ci(d, seed=7)),
                            "frac_neg": float((d < 0).mean())}

    # (ii) type-line swaps
    tr_rows = []

    def type_swap(mask, old_word, new_word, label):
        if not mask.any():
            return

        def swapped(r: int, edit: bool) -> str:
            if not edit:
                return stripped[r]
            i = cc.find_line(cc.lines(stripped[r]), "types:")
            words = cc.lines(stripped[r])[i][6:].split()
            if old_word == "creature" and new_word == "artifact creature":
                new_words = ["artifact"] + words
            else:
                new_words = [new_word if w == old_word else w for w in words]
            return cc.replace_in_line(stripped[r], i, "types: " + " ".join(new_words))

        S = pl.score_arms(cc.sample_rows(mask, 400, rng), [False, True], swapped)
        tr_rows.extend(pl.contrasts(S, [label]))

    is_art, is_ench = pl.flag(j, "is_artifact"), pl.flag(j, "is_enchantment")
    one_line = C.n_abilities == 1
    type_swap(pl.flag(j, "is_instant") & one_line, "instant", "sorcery", "instant -> sorcery")
    type_swap(pl.flag(j, "is_sorcery") & one_line, "sorcery", "instant", "sorcery -> instant")
    type_swap(C.is_creature & ~is_art, "creature", "artifact creature",
              "creature -> artifact creature")
    type_swap(C.is_creature & ~is_art & ~is_ench, "creature", "enchantment creature",
              "creature -> enchantment creature")
    pl.write_table(tr_rows, "c5_type_swaps.csv")

    # (iii) taplands
    tap_i = np.array([next((i for i, ln in enumerate(cc.lines(s)) if ln == TAPPED), -1)
                      for s in stripped])
    sel = np.flatnonzero(pl.flag(j, "is_land") & (tap_i >= 0))
    print(f"[C5] taplands: {len(sel)}", flush=True)
    arms = [lambda r: stripped[r],
            lambda r: cc.replace_in_line(stripped[r], tap_i[r],
                                         "replacement: CARDNAME enters untapped."),
            lambda r: cc.drop_line(stripped[r], tap_i[r])]
    L = pl.score_arms(sel, arms, lambda r, arm: arm(r))
    pl.write_table(pl.contrasts(L, ["tapped -> untapped (token sub)",
                                    "delete the enters-tapped line"]), "c5_taplands.csv")

    pl.write_summary(out, "c5_summary.json")


if __name__ == "__main__":
    main()
