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

import json
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


def main() -> None:
    j = pl.load_frame(text=True)
    stripped = list(j["text"])
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    rng = np.random.default_rng(13)
    out: dict = {}
    freq = pl.subtype_frequencies()

    # ── (i) tribal pairwise matrix ──────────────────────────────────────
    info = [cc.sole_tribe(s) if c else None for s, c in zip(stripped, is_crea)]
    rows_all = np.array([i for i, v in enumerate(info) if v is not None])
    tribe = np.array([info[i][1] for i in rows_all])

    base: list[int] = []
    for t in cc.TRIBES:
        pool = rows_all[tribe == t]
        if len(pool) < 15:
            continue
        take = pool if len(pool) <= MAX_PER_TRIBE else rng.choice(
            pool, MAX_PER_TRIBE, replace=False)
        base.extend(sorted(int(x) for x in take))
    base = np.array(base)
    own = np.array([info[r][1] for r in base])
    present = [t for t in cc.TRIBES if (own == t).sum() >= 15]
    print(f"[C5] tribal base {len(base)} over {len(present)} tribes: {present}",
          flush=True)

    def swap_tribe(r: int, new: str) -> str:
        i, old = info[r]
        words = cc.lines(stripped[r])[i][6:].split()
        return cc.replace_in_line(stripped[r], i,
                                  "types: " + " ".join(new if w == old else w for w in words))

    T = pl.score_texts([swap_tribe(r, t) for r in base for t in present])
    T = T.reshape(len(base), len(present))
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
    tri.to_csv(pl.SCRATCH / "c5_tribe_scale.csv", index=False)
    print(tri.to_string(index=False), flush=True)
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

    # ── (ii) type-line swaps ────────────────────────────────────────────
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    tr_rows = []

    def type_swap(mask, old_word, new_word, label):
        sel = np.flatnonzero(mask)
        if len(sel) == 0:
            return
        if len(sel) > 400:
            sel = np.sort(rng.choice(sel, 400, replace=False))
        txt = []
        for r in sel:
            i = cc.find_line(cc.lines(stripped[r]), "types:")
            words = cc.lines(stripped[r])[i][6:].split()
            if old_word == "creature" and new_word == "artifact creature":
                new_words = ["artifact"] + words
            else:
                new_words = [new_word if w == old_word else w for w in words]
            txt += [stripped[r], cc.replace_in_line(stripped[r], i,
                                                    "types: " + " ".join(new_words))]
        S = pl.score_texts(txt).reshape(len(sel), 2)
        tr_rows.append(pl.contrast_row(S.sp[:, 1] - S.sp[:, 0], S.pr[:, 1] - S.pr[:, 0],
                                       S.off[:, 1], contrast=label))

    def col(c):
        return j[c].fillna(False).astype(bool).to_numpy()

    is_art, is_ench = col("is_artifact"), col("is_enchantment")
    type_swap(col("is_instant") & (n_ab == 1), "instant", "sorcery", "instant -> sorcery")
    type_swap(col("is_sorcery") & (n_ab == 1), "sorcery", "instant", "sorcery -> instant")
    type_swap(is_crea & ~is_art, "creature", "artifact creature",
              "creature -> artifact creature")
    type_swap(is_crea & ~is_art & ~is_ench, "creature", "enchantment creature",
              "creature -> enchantment creature")
    tsw = pd.DataFrame(tr_rows)
    tsw.to_csv(pl.SCRATCH / "c5_type_swaps.csv", index=False)
    print(tsw.to_string(index=False), flush=True)

    # ── (iii) taplands ──────────────────────────────────────────────────
    tapped = "replacement: CARDNAME enters tapped."
    tap_i = np.array([next((i for i, ln in enumerate(cc.lines(s)) if ln == tapped), -1)
                      for s in stripped])
    sel = np.flatnonzero(col("is_land") & (tap_i >= 0))
    print(f"[C5] taplands: {len(sel)}", flush=True)
    txt = []
    for r in sel:
        txt += [stripped[r],
                cc.replace_in_line(stripped[r], tap_i[r], "replacement: CARDNAME enters untapped."),
                cc.drop_line(stripped[r], tap_i[r])]
    L = pl.score_texts(txt).reshape(len(sel), 3)
    lands = pd.DataFrame([
        pl.contrast_row(L.sp[:, i] - L.sp[:, 0], L.pr[:, i] - L.pr[:, 0], L.off[:, i],
                        contrast=lab)
        for i, lab in enumerate(["tapped -> untapped (token sub)",
                                 "delete the enters-tapped line"], start=1)])
    lands.to_csv(pl.SCRATCH / "c5_taplands.csv", index=False)
    print(lands.to_string(index=False), flush=True)

    (pl.SCRATCH / "c5_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
