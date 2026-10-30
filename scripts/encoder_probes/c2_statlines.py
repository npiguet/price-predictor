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

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c_common as cc  # noqa: E402
import probe_lib as pl  # noqa: E402
from _lib.stats import bootstrap_ci  # noqa: E402

PIP_SWAPS = (
    ("{1}{W}", ["{W}{W}", "{2}"]),
    ("{1}{U}", ["{U}{U}", "{2}"]),
    ("{1}{B}", ["{B}{B}", "{2}"]),
    ("{1}{R}", ["{R}{R}", "{2}"]),
    ("{1}{G}", ["{G}{G}", "{2}"]),
    ("{2}{W}", ["{1}{W}{W}", "{3}"]),
    ("{2}{R}", ["{1}{R}{R}", "{3}"]),
)


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


def main() -> None:
    j = pl.load_frame(text=True)
    C = cc.CardCorpus.of(j)
    power, tough, cost = C.power, C.toughness, np.array(C.cost)
    rng = np.random.default_rng(5)
    out: dict = {}

    def costed(r: int, new_cost: str | None) -> str:
        return C.text[r] if new_cost is None else cc.set_cost(C.text[r], new_cost)

    # (i) P vs T at fixed total
    tot = power + tough
    elig = cc.sample_rows(C.is_creature & C.literal_pt & np.isfinite(tot) & (tot >= 4)
                          & (tot <= 8) & (power >= 1) & (tough >= 2), 900, rng)
    A = pl.score_arms(elig, [(0, 0), (1, -1), (-1, 1)], lambda r, d: C.text[r] if d == (0, 0)
                      else cc.set_pt(C.text[r], int(power[r]) + d[0], int(tough[r]) + d[1]))

    def pt_contrasts(M: np.ndarray) -> dict[str, np.ndarray]:
        # arm 1 is +2 in P-T, arm 2 is -2; the gradient is per point of P-T
        return {"P+1/T-1 vs base (+2 in P-T)": M[:, 1] - M[:, 0],
                "P-1/T+1 vs base (-2 in P-T)": M[:, 2] - M[:, 0],
                "gradient per point of P-T": (M[:, 1] - M[:, 2]) / 4.0}

    pl.write_table([pl.contrast_row(v, contrast=name)
                    for name, v in pt_contrasts(A.sp).items()], "c2_pt_asymmetry.csv")
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
    pl.write_table(br, "c2_pt_by_size.csv")

    # the fixed vanilla bodies the two sweeps share
    bodies = cc.sample_rows(C.is_creature & C.literal_pt & (C.n_abilities == 0)
                            & np.array([bool(re.fullmatch(r"(\{\d\})?(\{[WUBRG]\})*", c))
                                        for c in cost]), 40, rng)
    print(f"[sweeps] {len(bodies)} vanilla bodies: "
          f"{list(j.loc[bodies, 'name'][:8])}", flush=True)

    # (ii) N/N sweep
    Ns = list(range(0, 13))
    S = pl.score_arms(bodies, Ns, lambda r, n: cc.set_pt(C.text[r], n, n))
    pl.write_table(sweep_table("N", Ns, S), "c2_nn_sweep.csv")
    out["nn_monotone_breaks"] = [
        int(n) for n, d in zip(Ns[1:], np.diff(S.sp.mean(axis=0))) if d < 0]

    # (iii) generic-cost sweep on the same bodies
    Cs = [f"{{{k}}}" for k in range(0, 10)]
    pl.write_table(sweep_table("cost", Cs, pl.score_arms(bodies, Cs, costed)),
                   "c2_mv_sweep.csv")

    # (iii.b) the {X} question: X rewritten to a fixed generic, or deleted
    has_x = np.array(["{X}" in c for c in cost])
    xr = cc.sample_rows(has_x & ~C.is_creature, 250, rng)
    X = pl.score_arms(xr, [None, "{3}", "", "{1}", "{6}"], lambda r, rep: costed(
        r, None if rep is None else cost[r].replace("{X}", rep, 1) or "{0}"))
    xrows = pl.contrasts(X, ["{X} -> {1}", "{X} -> {3}", "{X} -> {6}", "{X} deleted"],
                         cols=[3, 1, 4, 2])

    # reverse: a generic pip -> {X} on non-X spells
    plain = cc.sample_rows(~has_x & ~C.is_creature
                           & np.array([bool(re.match(r"^\{[1-9]\}", c)) for c in cost]),
                           250, rng)
    X2 = pl.score_arms(plain, [False, True], lambda r, to_x: costed(
        r, re.sub(r"^\{[1-9]\}", "{X}", cost[r]) if to_x else None))
    xrows += pl.contrasts(X2, ["generic pip -> {X} (non-X spells)"])
    pl.write_table(xrows, "c2_x_cost.csv")

    # (iv) pip substitutions
    pip_rows = []
    for base_cost, arms in PIP_SWAPS:
        mask = cost == base_cost
        if mask.sum() < 30:
            continue
        P = pl.score_arms(cc.sample_rows(mask, 400, rng), [None, *arms], costed)
        pip_rows += pl.contrasts(P, arms, field="arm", base=base_cost)
    pl.write_table(pip_rows, "c2_pips.csv")

    pl.write_summary(out, "c2_summary.json")


if __name__ == "__main__":
    main()
