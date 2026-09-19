"""What each principal direction of the ability-embedding space correlates with.

For one space — ``full`` (the shipping cache), ``residual`` (the shipping
cache with each script API type's mean vector subtracted) or ``taxonomy`` (the
hash baseline) — over one vector per unique script-surface text:

1. the variance share of each principal component (the centred PCA the
   gate-3 canary reads);
2. the ten texts at each end of the top ten components, with prose and
   script;
3. for every candidate feature, the share of each component's variance it
   explains on its own — one-way η² (between-group variance over total) for
   a categorical feature, squared Pearson correlation for a 0/1 or numeric
   one — and the share of the **whole** space's variance it explains
   (multivariate η², the between-group trace over the total trace; for a
   numeric feature the R² of a linear fit of every dimension on it);
4. cross-validated R² of nested feature sets (API type; + line kind;
   + every script feature; + card facts; + corpus facts) for each component
   and for the whole space, with folds grouped by carrying card so one card's
   texts never sit on both sides;
5. each API type's mean position along PC1 in standard deviations, and
   within-type splits by target;
6. when ``effect_profiles.csv`` exists, the observed-effect statistics most
   correlated with each component (weighted by ``n/(n+5)`` resolutions).

Run: ``python scripts/effect_embedding_probes/pca_directions.py --space full``
(then ``--space residual`` and ``--space taxonomy``). CPU only, ~2 min each.
Writes ``pca_<space>_*.md|csv`` to the report directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BINARY_SCRIPT,
    CARD,
    CORPUS,
    NUMERIC_SCRIPT,
    OUT,
    design,
    eta_squared,
    group_means_residual,
    matrix,
    multivariate_eta_squared,
    pca,
    prepared,
    weighted_corr,
    write_markdown,
)

N_PCS = 10
CATEGORICAL_FEATURES = {
    "API type": "api_c",
    "line kind": "line_kind",
    "trigger / static / replacement mode": "mode_c",
    "target type": "target",
    "script head key": "head_c",
    "keyword name": "keyword_c",
}


def space_matrix(table: pd.DataFrame, space: str) -> np.ndarray:
    if space == "full":
        return matrix(table, "e_full")
    if space == "taxonomy":
        return matrix(table, "e_tax")
    if space == "residual":
        return group_means_residual(matrix(table, "e_full"), table["api"])
    raise ValueError(space)


def r2_numeric(y: np.ndarray, x: np.ndarray) -> float:
    ok = np.isfinite(x)
    if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1] ** 2)


def space_r2_numeric(E: np.ndarray, x: np.ndarray) -> float:
    ok = np.isfinite(x)
    if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
        return float("nan")
    Ec = E[ok] - E[ok].mean(axis=0)
    xc = x[ok] - x[ok].mean()
    beta = (xc @ Ec) / (xc @ xc)
    return float(((np.outer(xc, beta)) ** 2).sum() / (Ec ** 2).sum())


def cv_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-column out-of-fold R² of a ridge fit, plus the pooled R²."""
    pred = np.zeros_like(Y)
    for train, test in GroupKFold(n_splits=5).split(X, Y, groups):
        model = Ridge(alpha=10.0).fit(X[train], Y[train])
        pred[test] = model.predict(X[test])
    ss_res = ((Y - pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    return np.append(1 - ss_res / ss_tot, 1 - ss_res.sum() / ss_tot.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", choices=("full", "residual", "taxonomy"),
                        default="full")
    args = parser.parse_args()
    space = args.space
    table, key_columns = prepared()
    E = space_matrix(table, space)
    scores, _, share = pca(E)
    sd = scores.std(axis=0)
    z = scores / sd

    variance = pd.DataFrame({
        "component": np.arange(1, 21),
        "share": share[:20],
        "cumulative": np.cumsum(share)[:20],
    })
    write_markdown(variance, OUT / f"pca_{space}_variance.md",
                   f"Variance share by principal component ({space}, "
                   f"{len(table)} unique texts)")

    # ── extremes ──
    lines = [f"# Extreme texts along each principal component ({space})", ""]
    for k in range(N_PCS):
        order = np.argsort(z[:, k])
        lines.append(f"## PC{k + 1} ({share[k]:.1%} of variance)")
        for label, idx in (("negative end", order[:10]),
                           ("positive end", order[::-1][:10])):
            lines.append(f"\n**{label}**\n")
            lines.append("| SD | card | API | kind | prose | script |")
            lines.append("|---:|---|---|---|---|---|")
            for i in idx:
                row = table.iloc[i]
                prose = str(row.prose)[:110].replace("|", "/")
                script = str(row.text)[:150].replace("|", "/")
                lines.append(f"| {z[i, k]:+.2f} | {row.card} | {row.api} | "
                             f"{row.line_kind} | {prose} | `{script}` |")
        lines.append("")
    (OUT / f"pca_{space}_extremes.md").write_text("\n".join(lines),
                                                   encoding="utf-8")

    # ── single-feature attribution ──
    rows = []
    for name, column in CATEGORICAL_FEATURES.items():
        row = {"feature": name, "kind": "categorical"}
        for k in range(N_PCS):
            row[f"PC{k + 1}"] = eta_squared(scores[:, k], table[column])
        row["whole space"] = multivariate_eta_squared(E, table[column])
        rows.append(row)
    numeric = list(BINARY_SCRIPT) + list(NUMERIC_SCRIPT) + list(CARD) + \
        list(CORPUS) + key_columns
    for column in numeric:
        x = table[column].to_numpy(dtype=float)
        row = {"feature": column,
               "kind": "key" if column.startswith("key_") else "numeric/binary"}
        for k in range(N_PCS):
            row[f"PC{k + 1}"] = r2_numeric(scores[:, k], x)
        row["whole space"] = space_r2_numeric(E, x)
        rows.append(row)
    attribution = pd.DataFrame(rows)
    attribution.to_csv(OUT / f"pca_{space}_feature_r2.csv", index=False)
    top = []
    for k in range(N_PCS):
        column = f"PC{k + 1}"
        best = attribution.nlargest(8, column)
        top.append({
            "component": column, "share": share[k],
            "top features (variance explained alone)": "; ".join(
                f"{f} {v:.2f}" for f, v in zip(best.feature, best[column])),
        })
    write_markdown(pd.DataFrame(top), OUT / f"pca_{space}_top_features.md",
                   f"Features explaining the most variance per component ({space})",
                   "η² for categoricals, r² for numeric and 0/1 features. "
                   "Full table: pca_" + space + "_feature_r2.csv.")

    # ── nested feature sets, cross-validated ──
    groups = table["group"].to_numpy()
    script_cats = ["api_c", "line_kind", "mode_c", "target", "head_c", "keyword_c"]
    script_nums = list(BINARY_SCRIPT) + list(NUMERIC_SCRIPT) + key_columns
    sets = {
        "API type": design(table, categorical=["api_c"]),
        "+ line kind": design(table, categorical=["api_c", "line_kind"]),
        "+ every script feature": design(table, categorical=script_cats,
                                         numeric=script_nums),
        "+ card facts": design(table, categorical=script_cats,
                               numeric=script_nums + list(CARD)),
        "+ corpus facts": design(table, categorical=script_cats,
                                 numeric=script_nums + list(CARD) + list(CORPUS)),
    }
    Y = scores[:, :N_PCS]
    nested = []
    for name, X in sets.items():
        per_pc = cv_r2(X, Y, groups)
        whole = cv_r2(X, E - E.mean(axis=0), groups)[-1]
        row = {"feature set": name, "columns": X.shape[1]}
        for k in range(N_PCS):
            row[f"PC{k + 1}"] = per_pc[k]
        row["whole space"] = whole
        nested.append(row)
    write_markdown(pd.DataFrame(nested), OUT / f"pca_{space}_nested_r2.md",
                   f"Cross-validated R² of nested feature sets ({space})",
                   "Ridge (alpha 10), 5 folds grouped by carrying card. "
                   "'whole space' is the pooled R² over all 64 dimensions.")

    # ── API types along PC1 and within-type target splits ──
    by_type = (pd.DataFrame({"api": table.api, "z": z[:, 0]})
               .groupby("api").z.agg(["count", "mean"]).reset_index())
    by_type = by_type[by_type["count"] >= 150].sort_values("mean")
    write_markdown(by_type, OUT / f"pca_{space}_pc1_by_api.md",
                   f"Mean PC1 position by API type, SD units ({space}; types "
                   "with at least 150 texts)", floatfmt=".2f")
    split = (pd.DataFrame({"api": table.api, "target": table.target,
                           "z1": z[:, 0], "z2": z[:, 1], "z3": z[:, 2]})
             .groupby(["api", "target"])
             .agg(count=("z1", "size"), pc1=("z1", "mean"), pc2=("z2", "mean"),
                  pc3=("z3", "mean"))
             .reset_index())
    split = split[split["count"] >= 30].sort_values(["api", "pc1"])
    write_markdown(split, OUT / f"pca_{space}_by_api_target.md",
                   f"Mean position on PCs 1-3 by API type and target ({space})",
                   floatfmt=".2f")

    # ── observed effects vs components ──
    if "res_n" in table:
        weight = table["res_n"].fillna(0).to_numpy(float)
        weight = weight / (weight + 5.0)
        stats = [c for c in table.columns
                 if c.startswith(("p_", "mean_", "share_opp", "kind_share_"))]
        corr_rows = []
        for column in stats:
            values = table[column].to_numpy(float)
            w = weight
            if column.startswith(("p_cost", "mean_cost")):
                n = table["cost_n"].fillna(0).to_numpy(float)
                w = n / (n + 5.0)
            elif column.startswith("mean_cont"):
                n = table["cont_n"].fillna(0).to_numpy(float)
                w = n / (n + 5.0)
            elif column.startswith("kind_share_"):
                n = table["n_records"].fillna(0).to_numpy(float)
                w = n / (n + 5.0)
            row = {"statistic": column,
                   "texts": int(np.isfinite(values).sum())}
            for k in range(N_PCS):
                row[f"PC{k + 1}"] = weighted_corr(values, scores[:, k], w)
            corr_rows.append(row)
        corr = pd.DataFrame(corr_rows)
        corr.to_csv(OUT / f"pca_{space}_effect_corr.csv", index=False)
        summary = []
        for k in range(N_PCS):
            column = f"PC{k + 1}"
            ordered = corr.dropna(subset=[column]).sort_values(column)
            summary.append({
                "component": column,
                "most negative": "; ".join(
                    f"{s} {v:+.2f}" for s, v in
                    zip(ordered.statistic[:5], ordered[column][:5])),
                "most positive": "; ".join(
                    f"{s} {v:+.2f}" for s, v in
                    zip(ordered.statistic[::-1][:5], ordered[column][::-1][:5])),
            })
        write_markdown(pd.DataFrame(summary), OUT / f"pca_{space}_effect_corr.md",
                       f"Observed-effect statistics most correlated with each "
                       f"component ({space})",
                       "Weighted Pearson correlation over acting texts, weight "
                       "n/(n+5) with n the text's records of the statistic's kind. "
                       "Full table: pca_" + space + "_effect_corr.csv.")
    print(variance.head(10).to_string(index=False))
    print(pd.DataFrame(nested).to_string(index=False))


if __name__ == "__main__":
    main()
