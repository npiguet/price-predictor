"""C10 — the price of a mana: cost sweeps at fixed effect, on both heads.

Every battery holds a card's text fixed and rewrites only a cost, so the
predicted change is the encoder's price for the mana itself. All edits are
within-line token substitutions (the clean, mean-zero null family from R1c);
readouts are the fidelity probes' ``score_play`` (winnability) and
``played_rate`` (the castability/agency head), both in label-SD units.

(i)   Mana-cost sweep: rewrite the generic component of the mana cost to
      k = 0..8, keeping colored pips, on three classes — vanilla creatures,
      creatures with rules text, and noncreature spells. Written both as
      absolute curves (mean prediction at each k) and as same-card deltas
      aligned on the card's real cost.
(ii)  Activation-cost sweep: creatures whose first activated ability costs
      mana (with or without {T}); rewrite the generic component to k = 0..6.
      Mana abilities ("add {…}") are excluded.
(iii) The {T} contrasts: drop ", {T}" from a mana-plus-tap activation cost,
      add ", {T}" to a mana-only one, and replace a bare {T} cost with {1}.
(iv)  Label-side check: families of real cards whose converted text is
      identical except for name and mana cost. Within-family WLS of the
      labels on mana value gives the corpus's own price per mana.
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
from c7_labelside import wls  # noqa: E402

SEED = 11
COST_RE = re.compile(r"^(?:\{(\d+)\})?((?:\{[WUBRG]\})*)$")
ACT_COST_RE = re.compile(
    r"^(?:\{(\d+)\})?((?:\{[WUBRG]\})*)(, \{T\})?$|^(\{T\})$")
SWEEP_K = list(range(0, 9))
ACT_K = list(range(0, 7))


def parse_cost(cost: str) -> tuple[int, str] | None:
    """(generic, pips) for a plain generic+pips cost, else None."""
    m = COST_RE.match(cost)
    if not m or (not m.group(1) and not m.group(2)):
        return None
    return int(m.group(1) or 0), m.group(2) or ""


def fmt_cost(k: int, pips: str) -> str:
    if k == 0:
        return pips if pips else "{0}"
    return f"{{{k}}}{pips}"


def set_cost(stripped: str, cost: str) -> str:
    ls = cc.lines(stripped)
    i = cc.find_line(ls, "mana cost:")
    ls[i] = f"mana cost: {cost}"
    return "\n".join(ls)


def parse_activated(stripped: str) -> tuple[int, str, str] | None:
    """(line index, cost part, effect) of the first activated ability."""
    ls = cc.lines(stripped)
    i = cc.find_line(ls, "activated[1]:")
    if i < 0:
        return None
    parts = ls[i].split(": ", 2)
    if len(parts) != 3:
        return None
    return i, parts[1], parts[2]


def parse_act_cost(cost: str) -> tuple[int, str, bool] | None:
    """(generic, pips, has_tap) for a mana / mana+{T} / bare-{T} cost."""
    m = ACT_COST_RE.match(cost)
    if not m:
        return None
    if m.group(4):  # bare {T}
        return 0, "", True
    if not m.group(1) and not m.group(2):
        return None  # empty mana part without bare-{T} form
    return int(m.group(1) or 0), m.group(2) or "", bool(m.group(3))


def fmt_act_cost(k: int, pips: str, tap: bool) -> str:
    mana = "" if (k == 0 and not pips) else fmt_cost(k, pips)
    if tap:
        return f"{mana}, {{T}}" if mana else "{T}"
    return mana if mana else "{0}"


def set_act_cost(stripped: str, line_i: int, cost: str, effect: str) -> str:
    return cc.replace_in_line(stripped, line_i, f"activated[1]: {cost}: {effect}")


def curve_rows(cls: str, ks, S, P, D) -> list[dict]:
    rows = []
    for i, k in enumerate(ks):
        rows.append({
            "class": cls, "k": k, "n": S.shape[0],
            "score_play_sd": float(S[:, i].mean()),
            "sp_ci_lo": cc.bootstrap_ci(S[:, i])[0],
            "sp_ci_hi": cc.bootstrap_ci(S[:, i])[1],
            "played_rate_sd": float(P[:, i].mean()),
            "pr_ci_lo": cc.bootstrap_ci(P[:, i])[0],
            "pr_ci_hi": cc.bootstrap_ci(P[:, i])[1],
            "off_manifold": float((D[:, i] > cc.MANIFOLD_GATE).mean()),
        })
    return rows


def delta_rows(cls: str, ks, S, P, g_real) -> list[dict]:
    """Same-card deltas aligned on the real generic cost."""
    rows = []
    k_arr = np.asarray(ks)
    in_range = (g_real >= k_arr.min()) & (g_real <= k_arr.max())
    for dlt in range(-2, 5):
        target = g_real + dlt
        ok = in_range & (target >= k_arr.min()) & (target <= k_arr.max())
        if ok.sum() < 15:
            continue
        col_t = np.searchsorted(k_arr, target[ok])
        col_0 = np.searchsorted(k_arr, g_real[ok])
        r = np.arange(len(g_real))[ok]
        ds = S[r, col_t] - S[r, col_0]
        dp = P[r, col_t] - P[r, col_0]
        rows.append({
            "class": cls, "delta_mana": dlt, "n": int(ok.sum()),
            "d_score_play_sd": float(ds.mean()),
            "ds_ci_lo": cc.bootstrap_ci(ds)[0],
            "ds_ci_hi": cc.bootstrap_ci(ds)[1],
            "d_played_rate_sd": float(dp.mean()),
            "dp_ci_lo": cc.bootstrap_ci(dp)[0],
            "dp_ci_hi": cc.bootstrap_ci(dp)[1],
        })
    return rows


def main() -> None:
    j = cc.join_table()
    stripped = [cc.strip_name(t) for t in j["text"]]
    is_crea = j["is_creature"].fillna(False).astype(bool).to_numpy()
    types = [cc.get_field(s, "types:") or "" for s in stripped]
    is_spell = np.array([("instant" in t or "sorcery" in t) for t in types])
    cost_field = [cc.get_field(s, "mana cost:") for s in stripped]
    parsed = [parse_cost(c) if c else None for c in cost_field]
    has_cost = np.array([p is not None for p in parsed])
    n_abil = np.array([len(cc.ability_lines(s)) for s in stripped])
    rng = np.random.default_rng(SEED)
    out: dict = {}

    def sample(mask: np.ndarray, cap: int) -> np.ndarray:
        idx = np.flatnonzero(mask)
        if len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        return idx

    # ── (i) mana-cost sweep ─────────────────────────────────────────────
    classes = {
        "vanilla creature": has_cost & is_crea & (n_abil == 0),
        "creature with text": has_cost & is_crea & (n_abil > 0),
        "noncreature spell": has_cost & is_spell,
    }
    all_curves, all_deltas = [], []
    for cls, mask in classes.items():
        idx = sample(mask, 150)
        print(f"[sweep] {cls}: {len(idx)} of {int(mask.sum())} eligible",
              flush=True)
        texts = [set_cost(stripped[r], fmt_cost(k, parsed[r][1]))
                 for r in idx for k in SWEEP_K]
        emb = cc.encode(texts)
        pred = cc.predict_sd(emb)
        S = pred["score_play"].to_numpy().reshape(len(idx), len(SWEEP_K))
        P = pred["played_rate"].to_numpy().reshape(len(idx), len(SWEEP_K))
        D = cc.offmanifold(emb).reshape(len(idx), len(SWEEP_K))
        all_curves += curve_rows(cls, SWEEP_K, S, P, D)
        g_real = np.array([parsed[r][0] for r in idx])
        all_deltas += delta_rows(cls, SWEEP_K, S, P, g_real)
    curves = pd.DataFrame(all_curves)
    curves.to_csv(cc.SCRATCH / "c10_cost_curves.csv", index=False)
    print(curves.to_string(index=False), flush=True)
    deltas = pd.DataFrame(all_deltas)
    deltas.to_csv(cc.SCRATCH / "c10_cost_deltas.csv", index=False)
    print(deltas.to_string(index=False), flush=True)

    # ── (ii) activation-cost sweep on creatures ─────────────────────────
    act = [parse_activated(s) for s in stripped]
    act_cost = [parse_act_cost(a[1]) if a else None for a in act]
    is_mana_ability = np.array([bool(a) and "add {" in a[2] for a in act])
    act_ok = np.array([a is not None for a in act_cost]) & ~is_mana_ability
    groups = {
        "activation mana + {T}": act_ok & is_crea
        & np.array([bool(a) and a[2] for a in act_cost], dtype=bool),
        "activation mana only": act_ok & is_crea
        & np.array([bool(a) and not a[2] for a in act_cost], dtype=bool),
    }
    act_curves, act_deltas = [], []
    for cls, mask in groups.items():
        idx = sample(mask, 150)
        print(f"[activation] {cls}: {len(idx)} of {int(mask.sum())} eligible",
              flush=True)
        texts = []
        for r in idx:
            line_i, _, effect = act[r]
            _, pips, tap = act_cost[r]
            for k in ACT_K:
                texts.append(set_act_cost(
                    stripped[r], line_i, fmt_act_cost(k, pips, tap), effect))
        emb = cc.encode(texts)
        pred = cc.predict_sd(emb)
        S = pred["score_play"].to_numpy().reshape(len(idx), len(ACT_K))
        P = pred["played_rate"].to_numpy().reshape(len(idx), len(ACT_K))
        D = cc.offmanifold(emb).reshape(len(idx), len(ACT_K))
        act_curves += curve_rows(cls, ACT_K, S, P, D)
        g_real = np.array([act_cost[r][0] for r in idx])
        act_deltas += delta_rows(cls, ACT_K, S, P, g_real)
    acurves = pd.DataFrame(act_curves)
    acurves.to_csv(cc.SCRATCH / "c10_ability_curves.csv", index=False)
    print(acurves.to_string(index=False), flush=True)
    adeltas = pd.DataFrame(act_deltas)
    adeltas.to_csv(cc.SCRATCH / "c10_ability_deltas.csv", index=False)
    print(adeltas.to_string(index=False), flush=True)

    # ── (iii) the {T} contrasts ─────────────────────────────────────────
    tap_rows = []
    specs = [
        ("drop {T} from mana+{T} cost", groups["activation mana + {T}"]
         & np.array([bool(a) and (a[0] > 0 or a[1]) for a in act_cost],
                    dtype=bool),
         lambda g, p, t: (fmt_act_cost(g, p, True), fmt_act_cost(g, p, False))),
        ("add {T} to mana-only cost", groups["activation mana only"],
         lambda g, p, t: (fmt_act_cost(g, p, False), fmt_act_cost(g, p, True))),
        ("bare {T} cost -> {1}", act_ok & is_crea
         & np.array([a == (0, "", True) for a in act_cost], dtype=bool),
         lambda g, p, t: ("{T}", "{1}")),
    ]
    for label, mask, arms in specs:
        idx = sample(mask, 400)
        print(f"[tap] {label}: {len(idx)} of {int(mask.sum())} eligible",
              flush=True)
        texts = []
        for r in idx:
            line_i, _, effect = act[r]
            g, pips, tap = act_cost[r]
            a0, a1 = arms(g, pips, tap)
            texts.append(set_act_cost(stripped[r], line_i, a0, effect))
            texts.append(set_act_cost(stripped[r], line_i, a1, effect))
        emb = cc.encode(texts)
        pred = cc.predict_sd(emb)
        S = pred["score_play"].to_numpy().reshape(len(idx), 2)
        P = pred["played_rate"].to_numpy().reshape(len(idx), 2)
        D = cc.offmanifold(emb).reshape(len(idx), 2)
        for head, M in (("score_play", S), ("played_rate", P)):
            v = M[:, 1] - M[:, 0]
            lo, hi = cc.bootstrap_ci(v)
            tap_rows.append({
                "contrast": label, "head": head, "n": len(idx),
                "mean_sd": float(v.mean()), "ci_lo": lo, "ci_hi": hi,
                "frac_pos": float((v > 0).mean()),
                "off_manifold": float((D > cc.MANIFOLD_GATE).mean()),
            })
    taps = pd.DataFrame(tap_rows)
    taps.to_csv(cc.SCRATCH / "c10_tap_contrasts.csv", index=False)
    print(taps.to_string(index=False), flush=True)

    # ── (iv) label-side same-text families ──────────────────────────────
    key = ["\n".join(l for l in cc.lines(s) if not l.startswith("mana cost:"))
           for s in stripped]
    mv = np.array([(p[0] + p[1].count("{")) if p else np.nan for p in parsed])
    fam = pd.DataFrame({
        "key": key, "mv": mv, "is_crea": is_crea,
        "generic": [p[0] if p else np.nan for p in parsed],
        "pips": [p[1] if p else None for p in parsed],
        "sp": j["shrunk_score_play"].to_numpy(float) / cc.SD["score_play"],
        "pr": j["shrunk_played_rate"].to_numpy(float) / cc.SD["played_rate"],
        "w": j["w_score_play"].to_numpy(float),
    })
    fam = fam[np.isfinite(fam["mv"]) & np.isfinite(fam["sp"])
              & (fam["w"] > 0)].copy()

    def family_slope(sub: pd.DataFrame, group_cols: list[str]) -> list[dict]:
        """Within-family WLS slope of each head on mana value.

        Families are groups identical on ``group_cols``; keep only groups
        where mana value actually varies, demean with the WLS weights so
        the slope is a pure within-family contrast, and fit without an
        intercept on the demeaned columns.
        """
        varies = sub.groupby(group_cols, dropna=False)["mv"].transform("nunique")
        sub = sub[varies >= 2]
        if len(sub) < 10:
            return []
        g = sub.groupby(group_cols, dropna=False)
        wsum = g["w"].transform("sum")

        def wdemean(col: str) -> np.ndarray:
            wm = (sub[col] * sub["w"]).groupby(
                [sub[c] for c in group_cols], dropna=False).transform("sum") / wsum
            return (sub[col] - wm).to_numpy()

        x = wdemean("mv")[:, None]
        rows = []
        for head, col in (("score_play", "sp"), ("played_rate", "pr")):
            beta, se = wls(x, wdemean(col), sub["w"].to_numpy())
            rows.append({
                "head": head, "n_families": int(g.ngroups), "n_cards": len(sub),
                "slope_sd_per_mana": float(beta[0]), "se": float(se[0]),
            })
        return rows

    fam_rows = []
    for label, sub, gcols in (
            ("all families", fam, ["key"]),
            ("creature families", fam[fam["is_crea"]], ["key"]),
            ("noncreature families", fam[~fam["is_crea"]], ["key"]),
            ("same pips, generic differs", fam, ["key", "pips"])):
        for row in family_slope(sub, gcols):
            fam_rows.append({"families": label, **row})
    fams = pd.DataFrame(fam_rows)
    fams.to_csv(cc.SCRATCH / "c10_label_families.csv", index=False)
    print(fams.to_string(index=False), flush=True)

    (cc.SCRATCH / "c10_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
