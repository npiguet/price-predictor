"""C2 (R11) — statlines and costs, by pure within-line token substitution.

Nothing here adds or removes a line: every arm edits the digits inside an
existing ``power toughness:`` or ``mana cost:`` line, which R1c showed is
the clean, mean-zero null family (~0.09 SD).

(i)   power-vs-toughness asymmetry at fixed P+T;
(ii)  the N/N integer sweep (also an R7-style monotonicity check);
(iii) the generic-cost sweep and the ``{X}`` question;
(iv)  pip substitutions: {1}{W} -> {W}{W} (harder) and -> {2} (colorless).
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
from _lib.stats import bootstrap_ci  # noqa: E402


def sweep_table(col: str, values: list, S: pl.Scored) -> pd.DataFrame:
    """Mean read-off at each sweep value over the fixed bodies, with its CI."""
    cis = [bootstrap_ci(S.sp[:, i], seed=7) for i in range(len(values))]
    tab = pd.DataFrame({
        col: values,
        "score_play_sd": S.sp.mean(axis=0),
        "ci_lo": [lo for lo, _ in cis],
        "ci_hi": [hi for _, hi in cis],
        "played_rate_sd": S.pr.mean(axis=0),
        "off_manifold": S.off.mean(axis=0),
    })
    tab["marginal"] = tab["score_play_sd"].diff()
    return tab


def write(tab: pd.DataFrame, name: str) -> None:
    tab.to_csv(pl.SCRATCH / name, index=False)
    print(tab.to_string(index=False), flush=True)


def main() -> None:
    j = pl.load_frame(text=True)
    stripped = list(j["text"])
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    power = pd.to_numeric(j["power"], errors="coerce").to_numpy(float)
    tough = pd.to_numeric(j["toughness"], errors="coerce").to_numpy(float)
    cost = [cc.get_field(s, "mana cost:") or "" for s in stripped]
    rng = np.random.default_rng(5)
    out: dict = {}
    lit_pt = np.array([cc.has_literal_pt(s) for s in stripped])

    # ── (i) P vs T at fixed total ───────────────────────────────────────
    tot = power + tough
    elig = np.flatnonzero(is_crea & lit_pt & np.isfinite(tot) &
                          (tot >= 4) & (tot <= 8) & (power >= 1) & (tough >= 2))
    if len(elig) > 900:
        elig = np.sort(rng.choice(elig, 900, replace=False))
    texts = []
    for r in elig:
        p, t = int(power[r]), int(tough[r])
        texts += [stripped[r], cc.set_pt(stripped[r], p + 1, t - 1),
                  cc.set_pt(stripped[r], p - 1, t + 1)]
    A = pl.score_texts(texts).reshape(len(elig), 3)

    def pt_contrasts(M: np.ndarray) -> dict[str, np.ndarray]:
        # arm 1 is +2 in P-T, arm 2 is -2; the gradient is per point of P-T
        return {"P+1/T-1 vs base (+2 in P-T)": M[:, 1] - M[:, 0],
                "P-1/T+1 vs base (-2 in P-T)": M[:, 2] - M[:, 0],
                "gradient per point of P-T": (M[:, 1] - M[:, 2]) / 4.0}

    asym = pd.DataFrame([pl.contrast_row(v, contrast=name)
                         for name, v in pt_contrasts(A.sp).items()])
    write(asym, "c2_pt_asymmetry.csv")
    pr = list(pt_contrasts(A.pr).values())
    out["pt_asymmetry_pr"] = {"up": float(pr[0].mean()), "down": float(pr[1].mean()),
                              "gradient_per_point": float(pr[2].mean())}
    out["pt_asymmetry_off_manifold"] = float(A.off.mean())

    # by total-size bucket
    grad = (A.sp[:, 1] - A.sp[:, 2]) / 4.0
    bt = tot[elig]
    br = []
    for lo_, hi_ in ((4, 5), (6, 6), (7, 8)):
        m = (bt >= lo_) & (bt <= hi_)
        if m.sum() < 20:
            continue
        br.append({"P+T": f"{lo_}-{hi_}" if lo_ != hi_ else str(lo_),
                   "n": int(m.sum()), **pl.mean_ci(grad[m], "gradient_per_point")})
    write(pd.DataFrame(br), "c2_pt_by_size.csv")

    # ── fixed bodies for the sweeps ─────────────────────────────────────
    bodies = np.flatnonzero(is_crea & lit_pt &
                            np.array([len(cc.ability_lines(s)) == 0 for s in stripped]) &
                            np.array([bool(re.fullmatch(r"(\{\d\})?(\{[WUBRG]\})*", c))
                                      for c in cost]))
    if len(bodies) > 40:
        bodies = np.sort(rng.choice(bodies, 40, replace=False))
    print(f"[sweeps] {len(bodies)} vanilla bodies: "
          f"{list(j.loc[bodies, 'name'][:8])}", flush=True)

    # ── (ii) N/N sweep ──────────────────────────────────────────────────
    Ns = list(range(0, 13))
    S = pl.score_texts([cc.set_pt(stripped[r], n, n) for r in bodies for n in Ns])
    S = S.reshape(len(bodies), len(Ns))
    write(sweep_table("N", Ns, S), "c2_nn_sweep.csv")
    out["nn_monotone_breaks"] = [
        int(n) for n, d in zip(Ns[1:], np.diff(S.sp.mean(axis=0))) if d < 0]

    # ── (iii) generic-cost sweep on the same bodies ─────────────────────
    Cs = [f"{{{k}}}" for k in range(0, 10)]
    S = pl.score_texts([cc.set_cost(stripped[r], c) for r in bodies for c in Cs])
    write(sweep_table("cost", Cs, S.reshape(len(bodies), len(Cs))), "c2_mv_sweep.csv")

    # ── (iii.b) the {X} question ────────────────────────────────────────
    has_x = np.array(["{X}" in c for c in cost])
    xr = np.flatnonzero(has_x & ~is_crea)
    if len(xr) > 250:
        xr = np.sort(rng.choice(xr, 250, replace=False))
    texts = []
    for r in xr:
        c = cost[r]
        texts += [stripped[r],
                  cc.set_cost(stripped[r], c.replace("{X}", "{3}", 1)),
                  cc.set_cost(stripped[r], c.replace("{X}", "", 1) or "{0}"),
                  cc.set_cost(stripped[r], c.replace("{X}", "{1}", 1)),
                  cc.set_cost(stripped[r], c.replace("{X}", "{6}", 1))]
    X = pl.score_texts(texts).reshape(len(xr), 5)
    xrows = [pl.contrast_row(X.sp[:, col] - X.sp[:, 0], X.pr[:, col] - X.pr[:, 0],
                             X.off[:, col], contrast=name)
             for name, col in (("{X} -> {1}", 3), ("{X} -> {3}", 1),
                               ("{X} -> {6}", 4), ("{X} deleted", 2))]

    # reverse: a generic pip -> {X} on non-X spells
    plain = np.flatnonzero(~has_x & ~is_crea &
                           np.array([bool(re.match(r"^\{[1-9]\}", c)) for c in cost]))
    if len(plain) > 250:
        plain = np.sort(rng.choice(plain, 250, replace=False))
    texts = []
    for r in plain:
        texts += [stripped[r], cc.set_cost(stripped[r], re.sub(r"^\{[1-9]\}", "{X}", cost[r]))]
    X2 = pl.score_texts(texts).reshape(len(plain), 2)
    xrows.append(pl.contrast_row(X2.sp[:, 1] - X2.sp[:, 0], X2.pr[:, 1] - X2.pr[:, 0],
                                 X2.off[:, 1], contrast="generic pip -> {X} (non-X spells)"))
    write(pd.DataFrame(xrows), "c2_x_cost.csv")

    # ── (iv) pip substitutions ──────────────────────────────────────────
    pip_rows = []
    for base_cost, arms in (
        ("{1}{W}", ["{W}{W}", "{2}"]),
        ("{1}{U}", ["{U}{U}", "{2}"]),
        ("{1}{B}", ["{B}{B}", "{2}"]),
        ("{1}{R}", ["{R}{R}", "{2}"]),
        ("{1}{G}", ["{G}{G}", "{2}"]),
        ("{2}{W}", ["{1}{W}{W}", "{3}"]),
        ("{2}{R}", ["{1}{R}{R}", "{3}"]),
    ):
        sel = np.flatnonzero(np.array(cost) == base_cost)
        if len(sel) < 30:
            continue
        if len(sel) > 400:
            sel = np.sort(rng.choice(sel, 400, replace=False))
        texts = []
        for r in sel:
            texts += [stripped[r]] + [cc.set_cost(stripped[r], a) for a in arms]
        P = pl.score_texts(texts).reshape(len(sel), 1 + len(arms))
        pip_rows += [pl.contrast_row(P.sp[:, ai] - P.sp[:, 0], P.pr[:, ai] - P.pr[:, 0],
                                     P.off[:, ai], base=base_cost, arm=a)
                     for ai, a in enumerate(arms, start=1)]
    write(pd.DataFrame(pip_rows), "c2_pips.csv")

    (pl.SCRATCH / "c2_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
