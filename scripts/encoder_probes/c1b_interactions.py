"""C1 interactions (R10) — flying x size, deathtouch x trample, keyword count.

All three designs are layout-matched:

* **flying x size** is read two ways — as a pure within-line substitution
  ``vigilance -> flying`` on the C1 base cards (no line added or removed at
  all), and as an add-arm contrast on keywordless creatures where both arms
  add exactly one ``static:`` line, so the line-add artifact cancels.
* **deathtouch x trample** uses a 2x2 in which *every* arm adds exactly two
  ``static:`` lines in the same two slots; the interaction term
  ``(1,1)-(1,0)-(0,1)+(0,0)`` cancels the two control keywords and the
  layout artifact simultaneously.
* **the keyword-count ladder** adds 1..4 keyword lines; there is no neutral
  static line to control against, so each rung's marginal is reported
  against the previous rung and the shape is compared across two keyword
  orderings (strong-first vs weak-first).
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

CONTROL_A, CONTROL_B = "vigilance", "reach"
LADDER_STRONG = ["flying", "deathtouch", "haste", "lifelink"]
LADDER_WEAK = ["vigilance", "trample", "first strike", "menace"]
SIZE_BUCKETS = ["<=2", "3-4", "5-6", "7-8", "9+"]


def _bucket(total: np.ndarray) -> np.ndarray:
    edges = [0, 2, 4, 6, 8, 100]
    out = np.full(len(total), "", dtype=object)
    for lo, hi, lab in zip(edges[:-1], edges[1:], SIZE_BUCKETS):
        out[(total > lo) & (total <= hi)] = lab
    return out


def main() -> None:
    j = pl.load_frame(text=True)
    stripped = list(j["text"])
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    power = pd.to_numeric(j["power"], errors="coerce").to_numpy(float)
    tough = pd.to_numeric(j["toughness"], errors="coerce").to_numpy(float)
    has_pt = np.isfinite(power) & np.isfinite(tough)
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    kw_count = pd.to_numeric(j["kw_count"], errors="coerce").fillna(0).to_numpy()
    out: dict = {}

    # ── (i.a) flying vs vigilance by body size, pure substitution ───────
    with np.load(pl.SCRATCH / "c1_pairwise_raw.npz", allow_pickle=True) as f:
        rows = f["rows"]
        sp = f["sp"]
        pr = f["pr"]
        kws = [str(x) for x in f["keywords"]]
    fi, vi = kws.index("flying"), kws.index("vigilance")
    d_sub = sp[:, fi] - sp[:, vi]
    d_sub_pr = pr[:, fi] - pr[:, vi]
    tot = power[rows] + tough[rows]
    buckets = _bucket(tot)
    sub_rows = []
    for lab in SIZE_BUCKETS:
        m = (buckets == lab) & np.isfinite(tot)
        if m.sum() < 15:
            continue
        sub_rows.append({"bucket": lab, "n": int(m.sum()),
                         **pl.mean_ci(d_sub[m], "flying_minus_vigilance_sp"),
                         "delta_pr": float(d_sub_pr[m].mean()),
                         "frac_pos": float((d_sub[m] > 0).mean())})
    sub_tab = pd.DataFrame(sub_rows)
    sub_tab.to_csv(pl.SCRATCH / "c1b_flying_size_sub.csv", index=False)
    print(sub_tab.to_string(index=False), flush=True)
    slope = np.polyfit(tot[np.isfinite(tot)], d_sub[np.isfinite(tot)], 1)[0]
    out["flying_vs_vigilance_slope_per_PT_point"] = float(slope)

    # ── (i.b) add-flying vs add-vigilance on keywordless creatures ──────
    base = np.flatnonzero(is_crea & has_pt & (kw_count == 0))
    rng = np.random.default_rng(3)
    if len(base) > 1200:
        base = np.sort(rng.choice(base, 1200, replace=False))
    arms = ["flying", "vigilance", "trample", "deathtouch", "lifelink", "defender"]
    texts = [cc.add_line(stripped[r], f"static: {a}") for r in base for a in arms]
    texts += [stripped[r] for r in base]           # the un-edited arm
    scored = pl.score_texts(texts)
    na = len(arms)
    A = scored[:len(base) * na].reshape(len(base), na)
    bare = scored[len(base) * na:]
    vig = arms.index("vigilance")
    tot_b = power[base] + tough[base]
    bk = _bucket(tot_b)

    add_rows = []
    for ai, a in enumerate(arms):
        no_line = pl.mean_ci(A.sp[:, ai] - bare.sp, "vs_no_line_sp")
        add_rows.append({
            "keyword": a, "n": len(base),
            **pl.mean_ci(A.sp[:, ai] - A.sp[:, vig], "vs_vigilance_sp"),
            "vs_no_line_sp": no_line["vs_no_line_sp"],
            "nl_ci_lo": no_line["ci_lo"], "nl_ci_hi": no_line["ci_hi"],
            "vs_vigilance_pr": float((A.pr[:, ai] - A.pr[:, vig]).mean()),
            "vs_no_line_pr": float((A.pr[:, ai] - bare.pr).mean()),
            "off_manifold": float(A.off[:, ai].mean()),
        })
    add_tab = pd.DataFrame(add_rows)
    add_tab.to_csv(pl.SCRATCH / "c1b_add_keyword.csv", index=False)
    print(add_tab.to_string(index=False), flush=True)

    size_rows = []
    for lab in SIZE_BUCKETS:
        m = bk == lab
        if m.sum() < 15:
            continue
        d = A.sp[m, arms.index("flying")] - A.sp[m, vig]
        size_rows.append({"bucket": lab, "n": int(m.sum()),
                          "mean_PT_total": float(tot_b[m].mean()),
                          **pl.mean_ci(d, "flying_minus_vigilance_sp")})
    size_tab = pd.DataFrame(size_rows)
    size_tab.to_csv(pl.SCRATCH / "c1b_flying_size_add.csv", index=False)
    print(size_tab.to_string(index=False), flush=True)

    # ── (ii) deathtouch x trample 2x2 ───────────────────────────────────
    mid = np.flatnonzero(is_crea & has_pt & (kw_count == 0) & (n_ab <= 1) &
                         (power + tough >= 3) & (power + tough <= 10))
    if len(mid) > 400:
        mid = np.sort(rng.choice(mid, 400, replace=False))
    # cells in order 00, 10, 01, 11
    cells = [(CONTROL_A, CONTROL_B), ("deathtouch", CONTROL_B),
             (CONTROL_A, "trample"), ("deathtouch", "trample")]
    C = pl.score_texts([cc.add_line(cc.add_line(stripped[r], f"static: {k1}"),
                                    f"static: {k2}")
                        for r in mid for k1, k2 in cells]).reshape(len(mid), 4)

    def terms(M: np.ndarray) -> dict[str, np.ndarray]:
        return {"deathtouch": M[:, 1] - M[:, 0], "trample": M[:, 2] - M[:, 0],
                "both": M[:, 3] - M[:, 0],
                "interaction": M[:, 3] - M[:, 1] - M[:, 2] + M[:, 0]}

    labels = {"deathtouch": "deathtouch (vs control)", "trample": "trample (vs control)",
              "both": "both (vs control)",
              "interaction": "interaction (joint - sum of singles)"}
    tt = pd.DataFrame([pl.contrast_row(v, term=labels[k])
                       for k, v in terms(C.sp).items()])
    tt.to_csv(pl.SCRATCH / "c1b_dt_trample.csv", index=False)
    print(tt.to_string(index=False), flush=True)
    out["dt_trample_off_manifold"] = float(C.off.mean())
    out["dt_trample_pr"] = {k: float(v.mean()) for k, v in terms(C.pr).items()}

    # ── (iii) keyword-count ladder ──────────────────────────────────────
    lad = np.flatnonzero(is_crea & has_pt & (kw_count == 0))
    if len(lad) > 500:
        lad = np.sort(rng.choice(lad, 500, replace=False))
    ladder_rows = []
    for order_name, order in (("strong-first", LADDER_STRONG),
                              ("weak-first", LADDER_WEAK)):
        ltexts = []
        for r in lad:
            t = stripped[r]
            ltexts.append(t)
            for k in order:
                t = cc.add_line(t, f"static: {k}")
                ltexts.append(t)
        L = pl.score_texts(ltexts).reshape(len(lad), len(order) + 1)
        for rung in range(1, len(order) + 1):
            marg = L.sp[:, rung] - L.sp[:, rung - 1]
            ladder_rows.append({
                "order": order_name, "rung": rung, "added": order[rung - 1],
                "n": len(lad), **pl.mean_ci(marg, "marginal_sp"),
                "cumulative_sp": float((L.sp[:, rung] - L.sp[:, 0]).mean()),
                "marginal_pr": float((L.pr[:, rung] - L.pr[:, rung - 1]).mean()),
                "off_manifold": float(L.off[:, rung].mean()),
            })
    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(pl.SCRATCH / "c1b_ladder.csv", index=False)
    print(ladder.to_string(index=False), flush=True)

    (pl.SCRATCH / "c1b_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
