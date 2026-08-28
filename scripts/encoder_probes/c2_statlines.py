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

PT_RE = re.compile(r"^power toughness:\s*(-?\d+)/(-?\d+)\s*$")


def set_pt(stripped: str, p: int, t: int) -> str:
    ls = cc.lines(stripped)
    i = cc.find_line(ls, "power toughness:")
    ls[i] = f"power toughness: {p}/{t}"
    return "\n".join(ls)


def set_cost(stripped: str, cost: str) -> str:
    ls = cc.lines(stripped)
    i = cc.find_line(ls, "mana cost:")
    if i < 0:
        return stripped
    ls[i] = f"mana cost: {cost}"
    return "\n".join(ls)


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    power = pd.to_numeric(j["power"], errors="coerce").to_numpy(float)
    tough = pd.to_numeric(j["toughness"], errors="coerce").to_numpy(float)
    cost = [cc.get_field(s, "mana cost:") or "" for s in stripped]
    rng = np.random.default_rng(5)
    out: dict = {}

    # the literal "N/M" form only (skip */1+*, X/X, etc.)
    lit_pt = np.array([bool(PT_RE.match(l))
                       for s in stripped
                       for l in [next((x for x in cc.lines(s)
                                       if x.startswith("power toughness:")), "")]])

    # ── (i) P vs T at fixed total ───────────────────────────────────────
    tot = power + tough
    elig = np.flatnonzero(is_crea & lit_pt & np.isfinite(tot) &
                          (tot >= 4) & (tot <= 8) & (power >= 1) & (tough >= 2))
    if len(elig) > 900:
        elig = np.sort(rng.choice(elig, 900, replace=False))
    texts = []
    for r in elig:
        p, t = int(power[r]), int(tough[r])
        texts.append(stripped[r])
        texts.append(set_pt(stripped[r], p + 1, t - 1))
        texts.append(set_pt(stripped[r], p - 1, t + 1))
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S = pred["score_play"].to_numpy().reshape(len(elig), 3)
    P = pred["played_rate"].to_numpy().reshape(len(elig), 3)
    D = dist.reshape(len(elig), 3)
    up = S[:, 1] - S[:, 0]           # +1 point of P-T (P+1/T-1) = +2 in P-T
    dn = S[:, 2] - S[:, 0]
    grad = (S[:, 1] - S[:, 2]) / 4.0  # per point of (P - T)
    rows = []
    for name, v in (("P+1/T-1 vs base (+2 in P-T)", up),
                    ("P-1/T+1 vs base (-2 in P-T)", dn),
                    ("gradient per point of P-T", grad)):
        lo, hi = cc.bootstrap_ci(v)
        rows.append({"contrast": name, "n": len(elig), "mean_sd": float(v.mean()),
                     "ci_lo": lo, "ci_hi": hi, "frac_pos": float((v > 0).mean())})
    asym = pd.DataFrame(rows)
    asym.to_csv(cc.SCRATCH / "c2_pt_asymmetry.csv", index=False)
    print(asym.to_string(index=False), flush=True)
    out["pt_asymmetry_pr"] = {
        "up": float((P[:, 1] - P[:, 0]).mean()),
        "down": float((P[:, 2] - P[:, 0]).mean()),
        "gradient_per_point": float(((P[:, 1] - P[:, 2]) / 4.0).mean()),
    }
    out["pt_asymmetry_off_manifold"] = float((D > cc.MANIFOLD_GATE).mean())

    # by total-size bucket
    bt = tot[elig]
    br = []
    for lo_, hi_ in ((4, 5), (6, 6), (7, 8)):
        m = (bt >= lo_) & (bt <= hi_)
        if m.sum() < 20:
            continue
        lo, hi = cc.bootstrap_ci(grad[m])
        br.append({"P+T": f"{lo_}-{hi_}" if lo_ != hi_ else str(lo_),
                   "n": int(m.sum()), "gradient_per_point": float(grad[m].mean()),
                   "ci_lo": lo, "ci_hi": hi})
    bysize = pd.DataFrame(br)
    bysize.to_csv(cc.SCRATCH / "c2_pt_by_size.csv", index=False)
    print(bysize.to_string(index=False), flush=True)

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
    texts = [set_pt(stripped[r], n, n) for r in bodies for n in Ns]
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S = pred["score_play"].to_numpy().reshape(len(bodies), len(Ns))
    P = pred["played_rate"].to_numpy().reshape(len(bodies), len(Ns))
    D = dist.reshape(len(bodies), len(Ns))
    nn = pd.DataFrame({
        "N": Ns,
        "score_play_sd": S.mean(axis=0),
        "ci_lo": [cc.bootstrap_ci(S[:, i])[0] for i in range(len(Ns))],
        "ci_hi": [cc.bootstrap_ci(S[:, i])[1] for i in range(len(Ns))],
        "played_rate_sd": P.mean(axis=0),
        "off_manifold": (D > cc.MANIFOLD_GATE).mean(axis=0),
    })
    nn["marginal"] = nn["score_play_sd"].diff()
    nn.to_csv(cc.SCRATCH / "c2_nn_sweep.csv", index=False)
    print(nn.to_string(index=False), flush=True)
    out["nn_monotone_breaks"] = [
        int(n) for n, d in zip(Ns[1:], np.diff(S.mean(axis=0))) if d < 0]

    # ── (iii) generic-cost sweep on the same bodies ─────────────────────
    Cs = [f"{{{k}}}" for k in range(0, 10)]
    texts = [set_cost(stripped[r], c) for r in bodies for c in Cs]
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S = pred["score_play"].to_numpy().reshape(len(bodies), len(Cs))
    P = pred["played_rate"].to_numpy().reshape(len(bodies), len(Cs))
    D = dist.reshape(len(bodies), len(Cs))
    mv = pd.DataFrame({
        "cost": Cs,
        "score_play_sd": S.mean(axis=0),
        "ci_lo": [cc.bootstrap_ci(S[:, i])[0] for i in range(len(Cs))],
        "ci_hi": [cc.bootstrap_ci(S[:, i])[1] for i in range(len(Cs))],
        "played_rate_sd": P.mean(axis=0),
        "off_manifold": (D > cc.MANIFOLD_GATE).mean(axis=0),
    })
    mv["marginal"] = mv["score_play_sd"].diff()
    mv.to_csv(cc.SCRATCH / "c2_mv_sweep.csv", index=False)
    print(mv.to_string(index=False), flush=True)

    # ── (iii.b) the {X} question ────────────────────────────────────────
    has_x = np.array(["{X}" in c for c in cost])
    xr = np.flatnonzero(has_x & ~is_crea)
    if len(xr) > 250:
        xr = np.sort(rng.choice(xr, 250, replace=False))
    texts = []
    for r in xr:
        c = cost[r]
        texts.append(stripped[r])
        texts.append(set_cost(stripped[r], c.replace("{X}", "{3}", 1)))
        texts.append(set_cost(stripped[r], c.replace("{X}", "", 1) or "{0}"))
        texts.append(set_cost(stripped[r], c.replace("{X}", "{1}", 1)))
        texts.append(set_cost(stripped[r], c.replace("{X}", "{6}", 1)))
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S = pred["score_play"].to_numpy().reshape(len(xr), 5)
    P = pred["played_rate"].to_numpy().reshape(len(xr), 5)
    D = dist.reshape(len(xr), 5)
    xrows = []
    for i, name in enumerate(["{X} -> {1}", "{X} -> {3}", "{X} -> {6}", "{X} deleted"]):
        col = {0: 3, 1: 1, 2: 4, 3: 2}[i]
        v = S[:, col] - S[:, 0]
        lo, hi = cc.bootstrap_ci(v)
        xrows.append({"contrast": name, "n": len(xr), "mean_sd": float(v.mean()),
                      "ci_lo": lo, "ci_hi": hi, "frac_pos": float((v > 0).mean()),
                      "delta_pr": float((P[:, col] - P[:, 0]).mean()),
                      "off_manifold": float((D[:, col] > cc.MANIFOLD_GATE).mean())})

    # reverse: a generic pip -> {X} on non-X spells
    plain = np.flatnonzero(~has_x & ~is_crea &
                           np.array([bool(re.match(r"^\{[1-9]\}", c)) for c in cost]))
    if len(plain) > 250:
        plain = np.sort(rng.choice(plain, 250, replace=False))
    texts = []
    for r in plain:
        texts.append(stripped[r])
        texts.append(set_cost(stripped[r], re.sub(r"^\{[1-9]\}", "{X}", cost[r])))
    emb = cc.encode(texts)
    pred = cc.predict_sd(emb)
    dist = cc.offmanifold(emb)
    S2 = pred["score_play"].to_numpy().reshape(len(plain), 2)
    P2 = pred["played_rate"].to_numpy().reshape(len(plain), 2)
    D2 = dist.reshape(len(plain), 2)
    v = S2[:, 1] - S2[:, 0]
    lo, hi = cc.bootstrap_ci(v)
    xrows.append({"contrast": "generic pip -> {X} (non-X spells)", "n": len(plain),
                  "mean_sd": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                  "frac_pos": float((v > 0).mean()),
                  "delta_pr": float((P2[:, 1] - P2[:, 0]).mean()),
                  "off_manifold": float((D2[:, 1] > cc.MANIFOLD_GATE).mean())})
    xtab = pd.DataFrame(xrows)
    xtab.to_csv(cc.SCRATCH / "c2_x_cost.csv", index=False)
    print(xtab.to_string(index=False), flush=True)

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
            texts.append(stripped[r])
            for a in arms:
                texts.append(set_cost(stripped[r], a))
        emb = cc.encode(texts)
        pred = cc.predict_sd(emb)
        dist = cc.offmanifold(emb)
        k = 1 + len(arms)
        S = pred["score_play"].to_numpy().reshape(len(sel), k)
        P = pred["played_rate"].to_numpy().reshape(len(sel), k)
        D = dist.reshape(len(sel), k)
        for ai, a in enumerate(arms, start=1):
            v = S[:, ai] - S[:, 0]
            lo, hi = cc.bootstrap_ci(v)
            pip_rows.append({"base": base_cost, "arm": a, "n": len(sel),
                             "mean_sd": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                             "frac_pos": float((v > 0).mean()),
                             "delta_pr": float((P[:, ai] - P[:, 0]).mean()),
                             "off_manifold": float((D[:, ai] > cc.MANIFOLD_GATE).mean())})
    pips = pd.DataFrame(pip_rows)
    pips.to_csv(cc.SCRATCH / "c2_pips.csv", index=False)
    print(pips.to_string(index=False), flush=True)

    (cc.SCRATCH / "c2_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
