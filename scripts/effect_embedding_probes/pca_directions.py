"""What each principal direction of the ability-embedding space correlates with.

For one space — ``full`` (the shipping cache), ``residual`` (the shipping
cache with each script API type's mean vector subtracted) or ``taxonomy`` (the
hash baseline) — over one vector per unique script-surface text:

1. the variance share of each principal component (the centred PCA the
   gate-3 canary reads), and the participation ratio ``(Σλ)² / Σλ²`` over the
   eigenvalues: how many dimensions the space effectively uses (FR-073);
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

Run: ``python scripts/effect_embedding_probes/pca_directions.py --checkpoint PATH
--abilities-root DIR --space full`` (then ``--space residual`` and
``--space taxonomy``). CPU only, ~2 min each.
Writes ``pca_<space>_*.md|csv`` to the report directory.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _lib.markdown import md_table
from _lib.stats import participation_ratio, pearson, ridge_cv_r2, wcorr  # noqa: E402

from effect_embedding_probes.common import (  # noqa: E402
    BINARY_SCRIPT,
    CARD,
    CORPUS,
    NUMERIC_SCRIPT,
    SCRIPT_CATS,
    MarkdownStyle,
    add_probe_arguments,
    design,
    eta_squared,
    group_means_residual,
    has_taxonomy,
    matrix,
    multivariate_eta_squared,
    pca,
    prepared,
    profile_weight,
    resolve_paths,
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


def space_r2_numeric(E: np.ndarray, x: np.ndarray) -> float:
    ok = np.isfinite(x)
    if ok.sum() < 30 or np.nanstd(x[ok]) == 0:
        return float("nan")
    Ec = E[ok] - E[ok].mean(axis=0)
    xc = x[ok] - x[ok].mean()
    beta = (xc @ Ec) / (xc @ xc)
    return float(((np.outer(xc, beta)) ** 2).sum() / (Ec ** 2).sum())


@dataclass(frozen=True)
class Decomposition:
    """One space's principal components of the probed texts, and where they are written."""

    paths: object
    space: str
    table: pd.DataFrame
    E: np.ndarray
    scores: np.ndarray
    share: np.ndarray
    key_columns: list

    @property
    def z(self) -> np.ndarray:
        """The component scores in standard-deviation units."""
        return self.scores / self.scores.std(axis=0)


def per_pc(values) -> dict:
    """``PC1``, ``PC2``, ... columns from one value per component."""
    return {f"PC{k + 1}": value for k, value in enumerate(values)}


def write_variance(d: Decomposition) -> pd.DataFrame:
    shown = min(20, len(d.share))
    variance = pd.DataFrame({
        "component": np.arange(1, shown + 1),
        "share": d.share[:shown],
        "cumulative": np.cumsum(d.share)[:shown],
    })
    ratio = participation_ratio(d.share)
    write_markdown(variance, d.paths.out / f"pca_{d.space}_variance.md",
                   f"Variance share by principal component ({d.space}, "
                   f"{len(d.table)} unique texts)",
                   MarkdownStyle(note=f"Participation ratio (Σλ)² / Σλ² over all "
                                      f"{len(d.share)} eigenvalues: {ratio:.2f}."))
    print(f"participation ratio ({d.space}): {ratio:.2f} of {len(d.share)} dimensions")
    return variance


def extreme_rows(d: Decomposition, k: int, idx) -> list:
    """The table rows of the texts ``idx`` along component ``k``."""
    z = d.z
    return [[f"{z[i, k]:+.2f}", row.card, row.api, row.line_kind,
             str(row.prose)[:110], f"`{str(row.text)[:150]}`"]
            for i, row in ((i, d.table.iloc[i]) for i in idx)]


def write_extremes(d: Decomposition) -> None:
    lines = [f"# Extreme texts along each principal component ({d.space})", ""]
    for k in range(N_PCS):
        order = np.argsort(d.z[:, k])
        lines.append(f"## PC{k + 1} ({d.share[k]:.1%} of variance)")
        for label, idx in (("negative end", order[:10]),
                           ("positive end", order[::-1][:10])):
            lines.append(f"\n**{label}**\n")
            lines.append(md_table(["SD", "card", "API", "kind", "prose", "script"],
                                  extreme_rows(d, k, idx), "rlllll"))
        lines.append("")
    (d.paths.out / f"pca_{d.space}_extremes.md").write_text("\n".join(lines),
                                                          encoding="utf-8")


def categorical_attribution(d: Decomposition) -> list:
    """The η² rows: how much of each component's variance a categorical feature explains."""
    return [{"feature": name, "kind": "categorical",
             **per_pc(eta_squared(d.scores[:, k], d.table[column]) for k in range(N_PCS)),
             "whole space": multivariate_eta_squared(d.E, d.table[column])}
            for name, column in CATEGORICAL_FEATURES.items()]


def numeric_attribution_row(d: Decomposition, column: str) -> dict:
    """The r² row of one numeric or 0/1 feature."""
    x = d.table[column].to_numpy(dtype=float)
    return {"feature": column,
            "kind": "key" if column.startswith("key_") else "numeric/binary",
            **per_pc(pearson(x, d.scores[:, k], min_n=30) ** 2 for k in range(N_PCS)),
            "whole space": space_r2_numeric(d.E, x)}


def top_features(attribution: pd.DataFrame, share: np.ndarray) -> pd.DataFrame:
    """The eight features explaining the most of each component, one row per component."""
    top = []
    for k in range(N_PCS):
        column = f"PC{k + 1}"
        best = attribution.nlargest(8, column)
        top.append({
            "component": column, "share": share[k],
            "top features (variance explained alone)": "; ".join(
                f"{f} {v:.2f}" for f, v in zip(best.feature, best[column])),
        })
    return pd.DataFrame(top)


def write_feature_attribution(d: Decomposition) -> None:
    """The share of each component's variance, and of the whole space's, one feature explains."""
    numeric = (list(BINARY_SCRIPT) + list(NUMERIC_SCRIPT) + list(CARD) + list(CORPUS)
               + d.key_columns)
    rows = categorical_attribution(d) + [numeric_attribution_row(d, c) for c in numeric]
    attribution = pd.DataFrame(rows)
    attribution.to_csv(d.paths.out / f"pca_{d.space}_feature_r2.csv", index=False)
    write_markdown(top_features(attribution, d.share),
                   d.paths.out / f"pca_{d.space}_top_features.md",
                   f"Features explaining the most variance per component ({d.space})",
                   MarkdownStyle(note="η² for categoricals, r² for numeric and 0/1 features. "
                                      "Full table: pca_" + d.space + "_feature_r2.csv."))


def nested_feature_sets(d: Decomposition) -> dict:
    """The design matrices of the nested feature sets, each adding to the one before."""
    table = d.table
    script_nums = list(BINARY_SCRIPT) + list(NUMERIC_SCRIPT) + d.key_columns
    return {
        "API type": design(table, categorical=["api_c"]),
        "+ line kind": design(table, categorical=["api_c", "line_kind"]),
        "+ every script feature": design(table, categorical=SCRIPT_CATS,
                                         numeric=script_nums),
        "+ card facts": design(table, categorical=SCRIPT_CATS,
                               numeric=script_nums + list(CARD)),
        "+ corpus facts": design(table, categorical=SCRIPT_CATS,
                                 numeric=script_nums + list(CARD) + list(CORPUS)),
    }


def write_nested_r2(d: Decomposition) -> pd.DataFrame:
    groups = d.table["group"].to_numpy()
    Y = d.scores[:, :N_PCS]
    nested = []
    for name, X in nested_feature_sets(d).items():
        scored = ridge_cv_r2(X, Y, groups, alpha=10.0)
        whole = ridge_cv_r2(X, d.E - d.E.mean(axis=0), groups, alpha=10.0)[-1]
        nested.append({"feature set": name, "columns": X.shape[1],
                       **per_pc(scored[:N_PCS]), "whole space": whole})
    frame = pd.DataFrame(nested)
    write_markdown(frame, d.paths.out / f"pca_{d.space}_nested_r2.md",
                   f"Cross-validated R² of nested feature sets ({d.space})",
                   MarkdownStyle(note="Ridge (alpha 10), 5 folds grouped by carrying card. "
                                      f"'whole space' is the pooled R² over all {d.E.shape[1]} "
                                      "dimensions."))
    return frame


def write_api_positions(d: Decomposition) -> None:
    """Each API type's mean PC1 position, and the within-type target splits."""
    paths, space, table, z = d.paths, d.space, d.table, d.z
    by_type = (pd.DataFrame({"api": table.api, "z": z[:, 0]})
               .groupby("api").z.agg(["count", "mean"]).reset_index())
    by_type = by_type[by_type["count"] >= 150].sort_values("mean")
    write_markdown(by_type, paths.out / f"pca_{space}_pc1_by_api.md",
                   f"Mean PC1 position by API type, SD units ({space}; types "
                   "with at least 150 texts)", style=MarkdownStyle(floatfmt=".2f"))
    split = (pd.DataFrame({"api": table.api, "target": table.target,
                           "z1": z[:, 0], "z2": z[:, 1], "z3": z[:, 2]})
             .groupby(["api", "target"])
             .agg(count=("z1", "size"), pc1=("z1", "mean"), pc2=("z2", "mean"),
                  pc3=("z3", "mean"))
             .reset_index())
    split = split[split["count"] >= 30].sort_values(["api", "pc1"])
    write_markdown(split, paths.out / f"pca_{space}_by_api_target.md",
                   f"Mean position on PCs 1-3 by API type and target ({space})",
                   style=MarkdownStyle(floatfmt=".2f"))


def write_effect_correlations(d: Decomposition) -> None:
    """The observed-effect statistics most correlated with each component."""
    paths, space, table, scores = d.paths, d.space, d.table, d.scores
    stats = [c for c in table.columns
             if c.startswith(("p_", "mean_", "share_opp", "kind_share_"))]
    corr_rows = []
    for column in stats:
        values = table[column].to_numpy(float)
        w = profile_weight(table, column)
        row = {"statistic": column,
               "texts": int(np.isfinite(values).sum())}
        for k in range(N_PCS):
            row[f"PC{k + 1}"] = wcorr(values, scores[:, k], w)
        corr_rows.append(row)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(paths.out / f"pca_{space}_effect_corr.csv", index=False)
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
    write_markdown(pd.DataFrame(summary), paths.out / f"pca_{space}_effect_corr.md",
                   f"Observed-effect statistics most correlated with each "
                   f"component ({space})",
                   MarkdownStyle(note="Weighted Pearson correlation over acting texts, weight "
                                      "n/(n+5) with n the text's records of the statistic's "
                                      "kind. Full table: pca_" + space + "_effect_corr.csv."))


def main() -> None:
    parser = add_probe_arguments(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
    )
    parser.add_argument("--space", choices=("full", "residual", "taxonomy"),
                        default="full")
    args = parser.parse_args()
    paths = resolve_paths(args)
    space = args.space
    table, key_columns = prepared(paths)
    if space == "taxonomy" and not has_taxonomy(table):
        print(f"no taxonomy cache under {paths.abilities_root}; nothing to "
              "decompose in the taxonomy space")
        return
    E = space_matrix(table, space)
    scores, _, share = pca(E)
    d = Decomposition(paths, space, table, E, scores, share, key_columns)

    variance = write_variance(d)
    write_extremes(d)
    write_feature_attribution(d)
    nested = write_nested_r2(d)
    write_api_positions(d)
    if "res_n" in table:
        write_effect_correlations(d)
    print(variance.head(10).to_string(index=False))
    print(nested.to_string(index=False))


if __name__ == "__main__":
    from effect_embedding_probes.common import main_with_manifest

    main_with_manifest(main, __file__)
