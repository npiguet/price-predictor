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
from c1_keywords import KEYWORDS, keyword_statics  # noqa: E402

CONTROL_A, CONTROL_B = "vigilance", "reach"
LADDER_STRONG = ["flying", "deathtouch", "haste", "lifelink"]
LADDER_WEAK = ["vigilance", "trample", "first strike", "menace"]


def _pt(j: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    p = pd.to_numeric(j["power"], errors="coerce").to_numpy(float)
    t = pd.to_numeric(j["toughness"], errors="coerce").to_numpy(float)
    return p, t


def _bucket(total: np.ndarray) -> np.ndarray:
    edges = [0, 2, 4, 6, 8, 100]
    labels = ["<=2", "3-4", "5-6", "7-8", "9+"]
    out = np.full(len(total), "", dtype=object)
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        out[(total > lo) & (total <= hi)] = lab
    return out


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    power, tough = _pt(j)
    has_pt = np.isfinite(power) & np.isfinite(tough)
    n_ab = np.array([len(cc.ability_lines(s)) for s in stripped])
    kw_count = pd.to_numeric(j["kw_count"], errors="coerce").fillna(0).to_numpy()
    out: dict = {}

    # ── (i.a) flying vs vigilance by body size, pure substitution ───────
    with np.load(cc.SCRATCH / "c1_pairwise_raw.npz", allow_pickle=True) as f:
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
    for lab in ["<=2", "3-4", "5-6", "7-8", "9+"]:
        m = (buckets == lab) & np.isfinite(tot)
        if m.sum() < 15:
            continue
        lo, hi = cc.bootstrap_ci(d_sub[m])
        sub_rows.append({"bucket": lab, "n": int(m.sum()),
                         "flying_minus_vigilance_sp": float(d_sub[m].mean()),
                         "ci_lo": lo, "ci_hi": hi,
                         "delta_pr": float(d_sub_pr[m].mean()),
                         "frac_pos": float((d_sub[m] > 0).mean())})
    sub_tab = pd.DataFrame(sub_rows)
    sub_tab.to_csv(cc.SCRATCH / "c1b_flying_size_sub.csv", index=False)
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
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    na = len(arms)
    A_sp = pred["score_play"].to_numpy()[:len(base) * na].reshape(len(base), na)
    A_pr = pred["played_rate"].to_numpy()[:len(base) * na].reshape(len(base), na)
    A_d = dist[:len(base) * na].reshape(len(base), na)
    bare_sp = pred["score_play"].to_numpy()[len(base) * na:]
    bare_pr = pred["played_rate"].to_numpy()[len(base) * na:]
    tot_b = power[base] + tough[base]
    bk = _bucket(tot_b)

    add_rows = []
    for ai, a in enumerate(arms):
        d = A_sp[:, ai] - A_sp[:, arms.index("vigilance")]
        dline = A_sp[:, ai] - bare_sp
        lo, hi = cc.bootstrap_ci(d)
        lo2, hi2 = cc.bootstrap_ci(dline)
        add_rows.append({
            "keyword": a, "n": len(base),
            "vs_vigilance_sp": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
            "vs_no_line_sp": float(dline.mean()), "nl_ci_lo": lo2, "nl_ci_hi": hi2,
            "vs_vigilance_pr": float((A_pr[:, ai] - A_pr[:, arms.index("vigilance")]).mean()),
            "vs_no_line_pr": float((A_pr[:, ai] - bare_pr).mean()),
            "off_manifold": float((A_d[:, ai] > cc.MANIFOLD_GATE).mean()),
        })
    add_tab = pd.DataFrame(add_rows)
    add_tab.to_csv(cc.SCRATCH / "c1b_add_keyword.csv", index=False)
    print(add_tab.to_string(index=False), flush=True)

    size_rows = []
    for lab in ["<=2", "3-4", "5-6", "7-8", "9+"]:
        m = bk == lab
        if m.sum() < 15:
            continue
        d = A_sp[m, arms.index("flying")] - A_sp[m, arms.index("vigilance")]
        lo, hi = cc.bootstrap_ci(d)
        size_rows.append({"bucket": lab, "n": int(m.sum()),
                          "mean_PT_total": float(tot_b[m].mean()),
                          "flying_minus_vigilance_sp": float(d.mean()),
                          "ci_lo": lo, "ci_hi": hi})
    size_tab = pd.DataFrame(size_rows)
    size_tab.to_csv(cc.SCRATCH / "c1b_flying_size_add.csv", index=False)
    print(size_tab.to_string(index=False), flush=True)

    # ── (ii) deathtouch x trample 2x2 ───────────────────────────────────
    mid = np.flatnonzero(is_crea & has_pt & (kw_count == 0) & (n_ab <= 1) &
                         (power + tough >= 3) & (power + tough <= 10))
    if len(mid) > 400:
        mid = np.sort(rng.choice(mid, 400, replace=False))
    cells = {
        "00": (CONTROL_A, CONTROL_B),
        "10": ("deathtouch", CONTROL_B),
        "01": (CONTROL_A, "trample"),
        "11": ("deathtouch", "trample"),
    }
    ctexts = []
    for r in mid:
        for k1, k2 in cells.values():
            t = cc.add_line(stripped[r], f"static: {k1}")
            t = cc.add_line(t, f"static: {k2}")
            ctexts.append(t)
    cemb = cc.encode(ctexts)
    cpred = cc.predict_sd(cemb)
    cdist = cc.offmanifold(cemb)
    C = cpred["score_play"].to_numpy().reshape(len(mid), 4)
    Cpr = cpred["played_rate"].to_numpy().reshape(len(mid), 4)
    Cd = cdist.reshape(len(mid), 4)
    keys = list(cells)
    i00, i10, i01, i11 = (keys.index(k) for k in ("00", "10", "01", "11"))
    dt_main = C[:, i10] - C[:, i00]
    tr_main = C[:, i01] - C[:, i00]
    joint = C[:, i11] - C[:, i00]
    inter = C[:, i11] - C[:, i10] - C[:, i01] + C[:, i00]
    tt_rows = []
    for name, v in (("deathtouch (vs control)", dt_main),
                    ("trample (vs control)", tr_main),
                    ("both (vs control)", joint),
                    ("interaction (joint - sum of singles)", inter)):
        lo, hi = cc.bootstrap_ci(v)
        tt_rows.append({"term": name, "n": len(mid), "mean_sd": float(v.mean()),
                        "ci_lo": lo, "ci_hi": hi,
                        "frac_pos": float((v > 0).mean())})
    tt = pd.DataFrame(tt_rows)
    tt.to_csv(cc.SCRATCH / "c1b_dt_trample.csv", index=False)
    print(tt.to_string(index=False), flush=True)
    out["dt_trample_off_manifold"] = float((Cd > cc.MANIFOLD_GATE).mean())
    out["dt_trample_pr"] = {
        "deathtouch": float((Cpr[:, i10] - Cpr[:, i00]).mean()),
        "trample": float((Cpr[:, i01] - Cpr[:, i00]).mean()),
        "both": float((Cpr[:, i11] - Cpr[:, i00]).mean()),
        "interaction": float((Cpr[:, i11] - Cpr[:, i10] - Cpr[:, i01] + Cpr[:, i00]).mean()),
    }

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
        lemb = cc.encode(ltexts)
        lpred = cc.predict_sd(lemb)
        ldist = cc.offmanifold(lemb)
        L = lpred["score_play"].to_numpy().reshape(len(lad), len(order) + 1)
        Lpr = lpred["played_rate"].to_numpy().reshape(len(lad), len(order) + 1)
        Ld = ldist.reshape(len(lad), len(order) + 1)
        for rung in range(1, len(order) + 1):
            marg = L[:, rung] - L[:, rung - 1]
            cum = L[:, rung] - L[:, 0]
            lo, hi = cc.bootstrap_ci(marg)
            ladder_rows.append({
                "order": order_name, "rung": rung, "added": order[rung - 1],
                "n": len(lad), "marginal_sp": float(marg.mean()),
                "ci_lo": lo, "ci_hi": hi, "cumulative_sp": float(cum.mean()),
                "marginal_pr": float((Lpr[:, rung] - Lpr[:, rung - 1]).mean()),
                "off_manifold": float((Ld[:, rung] > cc.MANIFOLD_GATE).mean()),
            })
    ladder = pd.DataFrame(ladder_rows)
    ladder.to_csv(cc.SCRATCH / "c1b_ladder.csv", index=False)
    print(ladder.to_string(index=False), flush=True)

    (cc.SCRATCH / "c1b_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
