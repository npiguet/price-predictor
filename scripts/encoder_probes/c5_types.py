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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from c1_keywords import fit_scale  # noqa: E402

TRIBES = [
    "dragon", "angel", "demon", "sphinx", "hydra", "wurm", "giant",
    "zombie", "goblin", "elf", "human", "soldier", "bird", "lizard",
    "spirit", "beast", "wall", "rat",
]
MAX_PER_TRIBE = 250


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    rng = np.random.default_rng(13)
    out: dict = {}
    freq = pl.subtype_frequencies()

    # ── (i) tribal pairwise matrix ──────────────────────────────────────
    def tribe_of(s: str) -> tuple[int, str] | None:
        for i, l in enumerate(cc.lines(s)):
            if not l.startswith("types:"):
                continue
            words = l[6:].split()
            if "creature" not in words:
                return None
            subs = [w for w in words if w not in pl._TYPE_WORDS
                    and w not in pl._SUPERTYPE_WORDS]
            hits = [w for w in subs if w in TRIBES]
            if len(subs) == 1 and len(hits) == 1:
                return i, hits[0]
            return None
        return None

    info = [tribe_of(s) if c else None for s, c in zip(stripped, is_crea)]
    rows_all = np.array([i for i, v in enumerate(info) if v is not None])
    tribe = np.array([info[i][1] for i in rows_all])
    line_of = {int(i): info[i][0] for i in rows_all}

    base: list[int] = []
    for t in TRIBES:
        pool = rows_all[tribe == t]
        if len(pool) < 15:
            continue
        take = pool if len(pool) <= MAX_PER_TRIBE else rng.choice(
            pool, MAX_PER_TRIBE, replace=False)
        base.extend(sorted(int(x) for x in take))
    base = np.array(base)
    own = np.array([info[r][1] for r in base])
    present = [t for t in TRIBES if (own == t).sum() >= 15]
    print(f"[C5] tribal base {len(base)} over {len(present)} tribes: {present}",
          flush=True)

    def swap_tribe(r: int, new: str) -> str:
        ls = cc.lines(stripped[r])
        i = line_of[int(r)]
        words = ls[i][6:].split()
        old = info[r][1]
        ls[i] = "types: " + " ".join(new if w == old else w for w in words)
        return "\n".join(ls)

    texts = [swap_tribe(r, t) for r in base for t in present]
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    nt = len(present)
    S = pred["score_play"].to_numpy().reshape(len(base), nt)
    P = pred["played_rate"].to_numpy().reshape(len(base), nt)
    D = dist.reshape(len(base), nt)
    groups = [np.flatnonzero(own == t) for t in present]

    prs = []
    for gi, t in enumerate(present):
        sel = groups[gi]
        for gj, tp in enumerate(present):
            if gi == gj:
                continue
            d = S[sel, gj] - S[sel, gi]
            prs.append({"from": t, "to": tp, "n": len(sel),
                        "delta_sp": float(d.mean()),
                        "delta_pr": float((P[sel, gj] - P[sel, gi]).mean())})
    pairs = pd.DataFrame(prs)
    pairs.to_csv(cc.SCRATCH / "c5_tribe_pairwise.csv", index=False)
    scale = fit_scale(pairs, present, "delta_sp")
    scale_pr = fit_scale(pairs, present, "delta_pr")

    brng = np.random.default_rng(17)
    boot = np.empty((400, nt))
    for b in range(400):
        res = [brng.choice(g, len(g), replace=True) for g in groups]
        rr = []
        for gi, t in enumerate(present):
            mm = S[res[gi]].mean(axis=0)
            for gj, tp in enumerate(present):
                if gi == gj:
                    continue
                rr.append({"from": t, "to": tp, "n": len(res[gi]),
                           "delta_sp": mm[gj] - mm[gi]})
        s = fit_scale(pd.DataFrame(rr), present, "delta_sp")
        boot[b] = [s[t] for t in present]

    tri = pd.DataFrame({
        "tribe": present,
        "n_carriers": [len(g) for g in groups],
        "corpus_freq": [freq.get(t, 0) for t in present],
        "value_sp": [scale[t] for t in present],
        "ci_lo": np.quantile(boot, 0.025, axis=0),
        "ci_hi": np.quantile(boot, 0.975, axis=0),
        "value_pr": [scale_pr[t] for t in present],
    }).sort_values("value_sp", ascending=False)
    tri.to_csv(cc.SCRATCH / "c5_tribe_scale.csv", index=False)
    print(tri.to_string(index=False), flush=True)
    out["tribe_off_manifold"] = float((D > cc.MANIFOLD_GATE).mean())
    out["tribe_spread_sd"] = float(tri["value_sp"].max() - tri["value_sp"].min())

    # the two headline directions, measured directly
    for a, b in (("dragon", "lizard"), ("angel", "bird"), ("dragon", "human"),
                 ("human", "dragon"), ("angel", "human"), ("wall", "human")):
        if a not in present or b not in present:
            continue
        sel = groups[present.index(a)]
        d = S[sel, present.index(b)] - S[sel, present.index(a)]
        lo, hi = cc.bootstrap_ci(d)
        out[f"{a}->{b}"] = {"n": int(len(sel)), "mean_sd": float(d.mean()),
                            "ci": [lo, hi], "frac_neg": float((d < 0).mean())}

    # ── (ii) type-line swaps ────────────────────────────────────────────
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    tr_rows = []

    def type_swap(mask, old_word, new_word, label):
        sel = np.flatnonzero(mask)
        if len(sel) == 0:
            return
        if len(sel) > 400:
            sel_ = np.sort(rng.choice(sel, 400, replace=False))
        else:
            sel_ = sel
        txt = []
        for r in sel_:
            ls = cc.lines(stripped[r])
            i = cc.find_line(ls, "types:")
            txt.append(stripped[r])
            words = ls[i][6:].split()
            if old_word == "creature" and new_word == "artifact creature":
                new_words = ["artifact"] + words
            else:
                new_words = [new_word if w == old_word else w for w in words]
            ls2 = list(ls)
            ls2[i] = "types: " + " ".join(new_words)
            txt.append("\n".join(ls2))
        e = cc.encode(txt)
        p = cc.predict_sd(e)
        dd = cc.offmanifold(e)
        A = p["score_play"].to_numpy().reshape(len(sel_), 2)
        B = p["played_rate"].to_numpy().reshape(len(sel_), 2)
        Dd = dd.reshape(len(sel_), 2)
        v = A[:, 1] - A[:, 0]
        lo, hi = cc.bootstrap_ci(v)
        tr_rows.append({"contrast": label, "n": len(sel_),
                        "mean_sd": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                        "frac_pos": float((v > 0).mean()),
                        "delta_pr": float((B[:, 1] - B[:, 0]).mean()),
                        "off_manifold": float((Dd[:, 1] > cc.MANIFOLD_GATE).mean())})

    is_inst = j["is_instant"].fillna(False).astype(bool).to_numpy()
    is_sorc = j["is_sorcery"].fillna(False).astype(bool).to_numpy()
    type_swap(is_inst & (n_ab == 1), "instant", "sorcery", "instant -> sorcery")
    type_swap(is_sorc & (n_ab == 1), "sorcery", "instant", "sorcery -> instant")
    is_art = j["is_artifact"].fillna(False).astype(bool).to_numpy()
    type_swap(is_crea & ~is_art, "creature", "artifact creature",
              "creature -> artifact creature")
    is_ench = j["is_enchantment"].fillna(False).astype(bool).to_numpy()
    type_swap(is_crea & ~is_art & ~is_ench, "creature", "enchantment creature",
              "creature -> enchantment creature")
    tsw = pd.DataFrame(tr_rows)
    tsw.to_csv(cc.SCRATCH / "c5_type_swaps.csv", index=False)
    print(tsw.to_string(index=False), flush=True)

    # ── (iii) taplands ──────────────────────────────────────────────────
    is_land = j["is_land"].fillna(False).astype(bool).to_numpy()
    tap_i = np.array([
        next((i for i, l in enumerate(cc.lines(s))
              if l == "replacement: CARDNAME enters tapped."), -1)
        for s in stripped])
    sel = np.flatnonzero(is_land & (tap_i >= 0))
    print(f"[C5] taplands: {len(sel)}", flush=True)
    txt = []
    for r in sel:
        txt.append(stripped[r])
        txt.append(cc.replace_in_line(stripped[r], tap_i[r],
                                      "replacement: CARDNAME enters untapped."))
        txt.append(cc.drop_line(stripped[r], tap_i[r]))
    e = cc.encode(txt)
    p = cc.predict_sd(e)
    dd = cc.offmanifold(e).reshape(len(sel), 3)
    A = p["score_play"].to_numpy().reshape(len(sel), 3)
    B = p["played_rate"].to_numpy().reshape(len(sel), 3)
    land_rows = []
    for i, lab in enumerate(["tapped -> untapped (token sub)",
                             "delete the enters-tapped line"], start=1):
        v = A[:, i] - A[:, 0]
        lo, hi = cc.bootstrap_ci(v)
        land_rows.append({"contrast": lab, "n": len(sel),
                          "mean_sd": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                          "frac_pos": float((v > 0).mean()),
                          "delta_pr": float((B[:, i] - B[:, 0]).mean()),
                          "off_manifold": float((dd[:, i] > cc.MANIFOLD_GATE).mean())})
    lands = pd.DataFrame(land_rows)
    lands.to_csv(cc.SCRATCH / "c5_taplands.csv", index=False)
    print(lands.to_string(index=False), flush=True)

    (cc.SCRATCH / "c5_summary.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps(out, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
