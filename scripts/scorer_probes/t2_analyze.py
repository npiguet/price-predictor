"""T2 analysis: what card properties explain the scorer's marginal card value?

Reads t2_card_values.csv (produced by t2_marginal_values.py) and prints a
markdown report to stdout. Pure numpy/pandas -- OLS is done by hand via lstsq
on standardized inputs so the coefficients are directly comparable and the
t-stats come from the classic sigma^2 (X'X)^-1 formula.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRATCH = Path(__file__).resolve().parents[2] / "output" / "scorer-probes"
DEFAULT_CSV = SCRATCH / "t2_card_values.csv"

RARITIES = ["common", "uncommon", "rare", "mythic"]
MV_BUCKETS = [("0-1", 0, 1), ("2", 2, 2), ("3", 3, 3), ("4", 4, 4),
              ("5", 5, 5), ("6+", 6, 99)]


# ---------------------------------------------------------------- formatting

def fmt(v, nd=4):
    if v is None:
        return "--"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "--"
    return f"{f:+.{nd}f}" if abs(f) < 1e6 else f"{f:.3g}"


def md_table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- statistics

def _pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def pearson(x, y):
    x, y = _pair(x, y)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan, len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def spearman(x, y):
    x, y = _pair(x, y)
    if len(x) < 3:
        return np.nan, len(x)
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, len(x)
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def ols_standardized(X, y, names):
    """OLS on z-scored X and y. Returns dict with betas, t-stats, R^2."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    n = len(y)
    if n < len(names) + 5:
        return None

    sd = X.std(axis=0)
    keep = sd > 1e-12
    dropped = [nm for nm, k in zip(names, keep) if not k]
    Xk = X[:, keep]
    kept = [nm for nm, k in zip(names, keep) if k]
    Z = (Xk - Xk.mean(axis=0)) / Xk.std(axis=0)
    ysd = y.std()
    if ysd == 0:
        return None
    yz = (y - y.mean()) / ysd

    A = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
    resid = yz - A @ beta
    dof = max(n - A.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = np.linalg.pinv(A.T @ A) * s2
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    r2 = 1.0 - float(resid @ resid) / float(yz @ yz)
    return {"names": ["const"] + kept, "beta": beta, "t": t, "r2": r2,
            "n": n, "dropped": dropped, "y_sd": float(ysd)}


def print_ols(title, res):
    print(f"\n**{title}**\n")
    if res is None:
        print("_too few usable rows to fit._")
        return
    print(f"N = {res['n']}, R^2 = {res['r2']:.4f}, sd(y) = {res['y_sd']:.4f}"
          + (f", dropped (no variance): {', '.join(res['dropped'])}" if res["dropped"] else ""))
    print()
    rows = []
    order = np.argsort(-np.abs(np.nan_to_num(res["t"])))
    for i in order:
        if res["names"][i] == "const":
            continue
        rows.append([res["names"][i], fmt(res["beta"][i], 4), fmt(res["t"][i], 1)])
    print(md_table(["term", "beta (std)", "t"], rows))


def group_stats(df, col, mask, label):
    s = pd.to_numeric(df.loc[mask, col], errors="coerce").dropna()
    if len(s) == 0:
        return [label, 0, "--", "--"]
    se = s.std() / np.sqrt(len(s)) if len(s) > 1 else np.nan
    return [label, len(s), fmt(s.mean()), fmt(se)]


# ---------------------------------------------------------------- data prep

def prepare(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_raw = len(df)
    if "v_swap" not in df.columns:
        sys.exit(f"{path} has no v_swap column -- wrong input file?")
    keep = pd.to_numeric(df["v_swap"], errors="coerce").notna()
    loo_only = int((~keep & pd.to_numeric(df.get("v_loo_rel"), errors="coerce").notna()).sum())
    df = df[keep].copy()

    numeric = ["v_swap", "v_loo_rel", "v_loo_raw", "mean_deck_score", "mv", "colors",
               "power", "toughness", "det_power", "det_toughness", "combat_kw_count",
               "shrunk_score_play", "shrunk_score_draw", "shrunk_played_rate",
               "shrunk_cast_lift", "n_obs", "n_loo_contexts", "n_swap_contexts",
               "release_year", "printings_count"]
    flags = ["is_creature", "is_instant", "is_sorcery", "is_artifact", "is_enchantment",
             "is_planeswalker", "is_aura", "is_equipment", "has_evasion", "has_flying",
             "is_removal", "draws_cards", "vanilla"]
    for c in numeric + flags:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in flags + ["combat_kw_count", "n_loo_contexts", "n_swap_contexts"]:
        df[c] = df[c].fillna(0.0)

    # power/toughness: text value, falling back to the deterministic block, then 0
    for c, d in (("power", "det_power"), ("toughness", "det_toughness")):
        df[c] = df[c].fillna(df[d]).fillna(0.0)

    df["n_colors"] = df["colors"].fillna(0.0)
    df["mv"] = df["mv"].fillna(0.0)
    df["mv2"] = df["mv"] ** 2
    df["evasion_nonflying"] = ((df["has_evasion"] > 0.5) & (df["has_flying"] < 0.5)).astype(float)
    df["pt_per_mv"] = (df["power"] + df["toughness"]) / np.maximum(df["mv"], 1.0)
    df["rarity"] = df.get("rarity", pd.Series([""] * len(df))).fillna("").astype(str)
    for r in RARITIES[1:]:
        df[f"rar_{r}"] = (df["rarity"] == r).astype(float)
    df["rar_unknown"] = (~df["rarity"].isin(RARITIES)).astype(float)

    def bucket(mv):
        for lab, lo, hi in MV_BUCKETS:
            if lo <= mv <= hi:
                return lab
        return "6+"

    df["mv_bucket"] = df["mv"].map(bucket)

    def cat(row):
        if row["is_creature"] > 0.5:
            return "creature"
        if row["is_removal"] > 0.5:
            return "removal"
        if row["draws_cards"] > 0.5:
            return "card draw"
        return "other noncreature"

    df["category"] = df.apply(cat, axis=1)
    print(f"_loaded {n_raw} rows; {len(df)} with a usable v_swap; {loo_only} further rows "
          f"carry only a LOO value (lands and 3+ colour cards have no eligible "
          f"two-colour swap context) and are dropped._\n")
    return df


# ---------------------------------------------------------------- sections

CORR_VARS = [
    ("shrunk_score_play", "empirical score_play label"),
    ("shrunk_score_draw", "empirical score_draw label"),
    ("shrunk_played_rate", "empirical played_rate label"),
    ("shrunk_cast_lift", "empirical cast_lift label"),
    ("mv", "mana value"),
    ("power", "power"),
    ("toughness", "toughness"),
    ("pt_per_mv", "(P+T)/max(mv,1)  [creatures only]"),
]


def section_correlations(df):
    print("\n## 1. Correlations of the two marginal-value measures\n")
    both = df[df["v_loo_rel"].notna()]
    r, n = pearson(both["v_swap"], both["v_loo_rel"])
    rho, _ = spearman(both["v_swap"], both["v_loo_rel"])
    print(f"Agreement between the measures: pearson r = {fmt(r)}, "
          f"spearman rho = {fmt(rho)} (n = {n}).\n")

    rows = []
    for col, label in CORR_VARS:
        if col not in df.columns:
            continue
        sub = df[df["is_creature"] > 0.5] if col == "pt_per_mv" else df
        cells = [label]
        for target in ("v_swap", "v_loo_rel"):
            r, n = pearson(sub[target], sub[col])
            rho, _ = spearman(sub[target], sub[col])
            cells += [fmt(r, 3), fmt(rho, 3), n]
        rows.append(cells)
    print(md_table(
        ["vs", "r(v_swap)", "rho(v_swap)", "n", "r(v_loo_rel)", "rho(v_loo_rel)", "n"],
        rows))


OLS_TERMS = ["mv", "mv2", "is_creature", "has_flying", "evasion_nonflying",
             "combat_kw_count", "is_removal", "draws_cards", "vanilla", "is_aura",
             "is_equipment", "is_instant", "is_sorcery", "power", "toughness",
             "n_colors", "rar_uncommon", "rar_rare", "rar_mythic", "rar_unknown"]


def section_ols(df):
    print("\n\n## 2. OLS on v_swap (standardized coefficients)\n")
    terms = [t for t in OLS_TERMS if t in df.columns]
    X = df[terms].to_numpy(float)
    res_a = ols_standardized(X, df["v_swap"].to_numpy(float), terms)
    print_ols("Model A -- card properties only", res_a)

    sub = df[df["shrunk_score_play"].notna()]
    if len(sub) < len(terms) + 10:
        print("\n_too few rows with a score_play label for model B._")
        return
    res_a2 = ols_standardized(sub[terms].to_numpy(float),
                              sub["v_swap"].to_numpy(float), terms)
    terms_b = terms + ["shrunk_score_play"]
    res_b = ols_standardized(sub[terms_b].to_numpy(float),
                             sub["v_swap"].to_numpy(float), terms_b)
    if res_a and res_a2 and res_a["n"] == res_a2["n"]:
        print("\n**Model A' -- same terms, restricted to rows with a score_play label**\n")
        print("_identical to model A: every scored card carries an empirical label._")
    else:
        print_ols("Model A' -- same terms, restricted to rows with a score_play label",
                  res_a2)
    print_ols("Model B -- A' + empirical shrunk_score_play", res_b)
    if res_a2 and res_b:
        print(f"\n**delta R^2 from adding shrunk_score_play: "
              f"{res_b['r2'] - res_a2['r2']:+.4f}** "
              f"({res_a2['r2']:.4f} -> {res_b['r2']:.4f}, N = {res_b['n']})\n")
        ba = dict(zip(res_a2["names"], res_a2["beta"]))
        bb = dict(zip(res_b["names"], res_b["beta"]))
        rows = []
        for t in terms:
            if t in ba and t in bb:
                shrink = 1 - abs(bb[t]) / abs(ba[t]) if abs(ba[t]) > 1e-9 else np.nan
                rows.append([t, fmt(ba[t]), fmt(bb[t]),
                             "--" if not np.isfinite(shrink) else f"{100 * shrink:+.0f}%"])
        rows.sort(key=lambda r: -abs(float(r[1])) if r[1] != "--" else 0)
        print("Does the label eat the flags' coefficients?\n")
        print(md_table(["term", "beta A'", "beta B", "shrinkage"], rows))


def section_groups(df):
    print("\n\n## 3. Group means of v_swap\n")

    print("### by rarity\n")
    rows = [group_stats(df, "v_swap", df["rarity"] == r, r) for r in RARITIES]
    rows.append(group_stats(df, "v_swap", ~df["rarity"].isin(RARITIES), "unknown"))
    print(md_table(["rarity", "n", "mean v_swap", "se"], rows))

    print("\n### by mana value\n")
    rows = [group_stats(df, "v_swap", df["mv_bucket"] == lab, f"MV {lab}")
            for lab, _lo, _hi in MV_BUCKETS]
    print(md_table(["mv bucket", "n", "mean v_swap", "se"], rows))

    print("\n### by card category\n")
    rows = [group_stats(df, "v_swap", df["category"] == c, c)
            for c in ("creature", "removal", "card draw", "other noncreature")]
    print(md_table(["category", "n", "mean v_swap", "se"], rows))

    print("\n### vanilla vs keyworded creatures\n")
    cre = df["is_creature"] > 0.5
    rows = [
        group_stats(df, "v_swap", cre & (df["vanilla"] > 0.5), "vanilla creature"),
        group_stats(df, "v_swap", cre & (df["vanilla"] < 0.5) & (df["combat_kw_count"] > 0),
                    "creature w/ combat kw"),
        group_stats(df, "v_swap", cre & (df["vanilla"] < 0.5) & (df["combat_kw_count"] == 0),
                    "creature, no combat kw (text only)"),
    ]
    print(md_table(["group", "n", "mean v_swap", "se"], rows))

    print("\n### flying vs ground creatures, matched by MV bucket\n")
    rows = []
    for lab, _lo, _hi in MV_BUCKETS:
        cell = cre & (df["mv_bucket"] == lab)
        f = group_stats(df, "v_swap", cell & (df["has_flying"] > 0.5), lab)
        g = group_stats(df, "v_swap", cell & (df["has_flying"] < 0.5), lab)
        try:
            diff = fmt(float(f[2]) - float(g[2]))
        except ValueError:
            diff = "--"
        rows.append([lab, f[1], f[2], g[1], g[2], diff])
    print(md_table(["mv bucket", "n flying", "mean (flying)", "n ground",
                    "mean (ground)", "flying - ground"], rows))


def section_extremes(df, k):
    print(f"\n\n## 4. Extremes: top/bottom {k} cards by v_swap\n")
    cols = ["name", "v_swap", "n_swap_contexts", "mv", "rarity",
            "shrunk_score_play", "category"]
    cols = [c for c in cols if c in df.columns]
    d = df.sort_values("v_swap", ascending=False)

    def block(sub, title):
        print(f"### {title}\n")
        rows = []
        for _i, r in sub.iterrows():
            ctx = r.get("n_swap_contexts")
            ctx = int(ctx) if pd.notna(ctx) else 0
            rows.append([r["name"], fmt(r["v_swap"]), ctx,
                         fmt(r.get("mv"), 1), r.get("rarity", "") or "--",
                         fmt(r.get("shrunk_score_play"), 4), r.get("category", "")])
        print(md_table(["name", "v_swap", "ctx", "mv", "rarity", "score_play", "category"],
                       rows))

    block(d.head(k), f"top {k}")
    print()
    block(d.tail(k).iloc[::-1], f"bottom {k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    if not args.csv.exists():
        sys.exit(f"missing input: {args.csv}")

    print(f"# T2 -- marginal card value under the scorer\n")
    print(f"_source: `{args.csv.name}`_\n")
    df = prepare(args.csv)
    if len(df) < 10:
        print("**Too few rows with v_swap for a meaningful report.**")
        return

    has_loo = df["n_loo_contexts"] > 0
    med_loo = df.loc[has_loo, "n_loo_contexts"].median() if has_loo.any() else np.nan
    print("Coverage (rows with a v_swap value): "
          f"{int(has_loo.sum())} of them also have LOO contexts "
          f"(median {med_loo:.0f} contexts each), "
          f"{len(df)} have swap-in values "
          f"(median {df['n_swap_contexts'].median():.0f} contexts). "
          f"sd(v_swap) = {df['v_swap'].std():.4f}, "
          f"sd(v_loo_rel) = {df['v_loo_rel'].std():.4f}.")

    section_correlations(df)
    section_ols(df)
    section_groups(df)
    section_extremes(df, min(args.top, len(df)))


if __name__ == "__main__":
    main()
