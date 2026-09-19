"""Which properties are linearly decodable from the whole 64-d embedding.

Fits one cross-validated linear probe per property, from each of several
input spaces, over every unique script-surface text (nothing is filtered to a
subset of cards: a text that lacks a property is labelled 0 / "none" for it):

- ``full`` — the shipping cache's ``e``;
- ``taxonomy`` — the taxonomy baseline's ``e``, a hash of the API type and the
  set of parameter keys and nothing else;
- ``residual`` — ``full`` with each API type's mean vector subtracted, so
  whatever it decodes is carried *within* an API type;
- ``api one-hot`` — the API type alone, as indicator columns (the ceiling of
  what knowing only the API type gives);
- ``script features`` and ``full + script features`` (effect-profile targets
  only) — the hand-parsed script features of ``common.py`` as a design
  matrix, alone and appended to ``full``, to ask whether the embedding holds
  anything about observed effects beyond what a parser reads off the script.

Two target families (``--targets``):

- ``script`` — properties of the text and its card: API type, line kind,
  mode, target type (multi-class: accuracy against the majority-class rate);
  each 0/1 script, card and corpus feature and each common parameter key's
  presence (ROC AUC, the probability that a random positive scores above a
  random negative; 0.5 is chance); numeric amounts and counts (R², after
  clipping each at its 0.1st/99.9th percentiles).
- ``profile`` — the observed-effect statistics of ``effect_profiles.csv``,
  over the texts that have records of the relevant kind, by weighted ridge
  with weight ``n/(n+5)`` (weighted R²); ``mean_*`` statistics are
  winsorized at their 0.5th/99.5th percentiles first. Because the effect
  head reads ``e`` non-linearly, the ``full``, ``taxonomy`` and ``script
  features`` spaces also get two non-linear probes: 20-nearest-neighbour
  regression (not on script features) and gradient-boosted trees.

Every fold split is ``GroupKFold(5)`` over the alphabetically first card
carrying the text, so one card's texts never sit on both sides.

Run: ``python scripts/effect_embedding_probes/linear_probes.py --targets script``
then ``--targets profile``. CPU only (joblib, 8 processes), ~10 min each.
Writes ``probes_<targets>.csv`` and ``probes_<targets>.md``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    BINARY_SCRIPT,
    CARD,
    NUMERIC_SCRIPT,
    OUT,
    design,
    group_means_residual,
    matrix,
    prepared,
    winsorized,
    write_markdown,
)

FOLDS = 5
MULTICLASS = {"API type": "api_c", "line kind": "line_kind",
              "mode": "mode_c", "target type": "target"}
NUMERIC_TARGETS = list(NUMERIC_SCRIPT) + ["card_cmc", "card_n_colours",
                                          "log_rarity", "log_carriers",
                                          "log_train_records"]
BINARY_CARD = [c for c in CARD if c not in ("card_cmc", "card_n_colours")]
#: Spaces that also get a non-linear probe on profile targets: a linear probe
#: cannot carve out the API-type groups that a one-hot hands a ridge for free,
#: while the effect head reading ``e`` is itself non-linear.
NONLINEAR_SPACES = ("full", "taxonomy", "script features")
SCRIPT_CATS = ["api_c", "line_kind", "mode_c", "target", "head_c", "keyword_c"]


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def spaces_for(table: pd.DataFrame, key_columns: list[str],
               with_features: bool) -> dict[str, np.ndarray]:
    full = matrix(table, "e_full")
    spaces = {
        "full": standardize(full),
        "taxonomy": standardize(matrix(table, "e_tax")),
        "residual": standardize(group_means_residual(full, table["api"])),
        "api one-hot": design(table, categorical=["api_c"]),
    }
    if with_features:
        features = design(table, categorical=SCRIPT_CATS,
                          numeric=list(BINARY_SCRIPT) + list(NUMERIC_SCRIPT)
                          + key_columns)
        spaces["script features"] = features
        spaces["full + script features"] = np.hstack([spaces["full"], features])
    return spaces


def probe_binary(X, y, groups) -> float:
    pred = np.zeros(len(y))
    for train, test in GroupKFold(FOLDS).split(X, y, groups):
        if len(np.unique(y[train])) < 2:
            return float("nan")
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(X[train], y[train])
        pred[test] = model.decision_function(X[test])
    return float(roc_auc_score(y, pred))


def probe_multiclass(X, y, groups) -> float:
    pred = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(FOLDS).split(X, y, groups):
        model = LogisticRegression(C=1.0, max_iter=500)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return float((pred == y).mean())


def probe_numeric(X, y, groups, w=None) -> float:
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]
    w = np.ones(len(y)) if w is None else w[ok]
    pred = np.zeros(len(y))
    for train, test in GroupKFold(FOLDS).split(X, y, groups):
        model = Ridge(alpha=10.0)
        model.fit(X[train], y[train], sample_weight=w[train])
        pred[test] = model.predict(X[test])
    mean = np.average(y, weights=w)
    ss_tot = float((w * (y - mean) ** 2).sum())
    return 1.0 - float((w * (y - pred) ** 2).sum()) / ss_tot if ss_tot else float("nan")


def probe_nonlinear(X, y, groups, w, method: str) -> float:
    """Out-of-fold weighted R² of a k-nearest-neighbour or boosted-tree fit."""
    pred = np.zeros(len(y))
    for train, test in GroupKFold(FOLDS).split(X, y, groups):
        if method == "knn":
            model = KNeighborsRegressor(n_neighbors=20)
            model.fit(X[train], y[train])
        else:
            model = HistGradientBoostingRegressor(max_iter=200, random_state=0)
            model.fit(X[train], y[train], sample_weight=w[train])
        pred[test] = model.predict(X[test])
    mean = np.average(y, weights=w)
    ss_tot = float((w * (y - mean) ** 2).sum())
    return 1.0 - float((w * (y - pred) ** 2).sum()) / ss_tot if ss_tot else float("nan")


def run_one(task, spaces, groups):
    kind, name, y, w = task
    warnings.simplefilter("ignore", ConvergenceWarning)
    out = {"target": name, "metric": {"binary": "AUC", "multiclass": "accuracy",
                                       "numeric": "R²", "profile": "weighted R²"}[kind]}
    if kind == "binary":
        out["positives"] = int(y.sum())
    elif kind == "multiclass":
        out["baseline"] = float(pd.Series(y).value_counts(normalize=True).iloc[0])
        out["classes"] = int(pd.Series(y).nunique())
    else:
        out["texts"] = int(np.isfinite(y).sum() if w is None
                           else (np.isfinite(y) & (w > 0)).sum())
    for space, X in spaces.items():
        if kind == "binary":
            out[space] = probe_binary(X, y, groups)
        elif kind == "multiclass":
            out[space] = probe_multiclass(X, y, groups)
        elif kind == "numeric":
            out[space] = probe_numeric(X, y, groups)
        else:
            keep = w > 0
            out[space] = probe_numeric(X[keep], y[keep], groups[keep], w[keep])
            if space in NONLINEAR_SPACES:
                for method in ("knn", "boosted trees"):
                    if method == "knn" and space == "script features":
                        continue
                    out[f"{space} ({method})"] = probe_nonlinear(
                        X[keep], y[keep], groups[keep], w[keep],
                        "knn" if method == "knn" else "hgb")
    return out


def script_tasks(table: pd.DataFrame, key_columns: list[str]) -> list:
    tasks = []
    for name, column in MULTICLASS.items():
        tasks.append(("multiclass", name, table[column].to_numpy(object), None))
    for column in list(BINARY_SCRIPT) + BINARY_CARD + ["held_out"] + key_columns:
        y = table[column].to_numpy(float)
        if 30 <= y.sum() <= len(y) - 30:
            tasks.append(("binary", column, y, None))
    for column in NUMERIC_TARGETS:
        tasks.append(("numeric", column,
                      winsorized(table[column].to_numpy(float)), None))
    return tasks


def weight_for(table: pd.DataFrame, column: str) -> np.ndarray:
    if column.startswith(("p_cost", "mean_cost")):
        n = table["cost_n"]
    elif column.startswith("mean_cont"):
        n = table["cont_n"]
    elif column.startswith("p_trigger"):
        n = table["n_trigger"]
    elif column.startswith("kind_share"):
        n = table["n_records"]
    else:
        n = table["res_n"]
    n = n.fillna(0).to_numpy(float)
    return n / (n + 5.0)


def profile_tasks(table: pd.DataFrame) -> list:
    tasks = []
    for column in table.columns:
        if not column.startswith(("p_", "mean_", "share_opp", "kind_share_")):
            continue
        y = table[column].to_numpy(float)
        w = weight_for(table, column)
        w = np.where(np.isfinite(y), w, 0.0)
        have = w > 0
        if have.sum() < 1000:
            continue
        if column.startswith("p_") and np.average(y[have], weights=w[have]) < 0.005:
            continue
        if not column.startswith("p_"):
            # A handful of texts reach extreme means (one X-counter ability
            # averages 522 counters); winsorizing keeps one text from
            # deciding a weighted R².
            low, high = np.quantile(y[have], [0.005, 0.995])
            y = np.clip(y, low, high)
            if high <= low:
                continue
        tasks.append(("profile", column, np.nan_to_num(y), w))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--targets", choices=("script", "profile"), default="script")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    table, key_columns = prepared()
    groups = table["group"].to_numpy()
    spaces = spaces_for(table, key_columns, with_features=args.targets == "profile")
    tasks = (script_tasks(table, key_columns) if args.targets == "script"
             else profile_tasks(table))
    print(f"{len(tasks)} targets x {len(spaces)} spaces", flush=True)
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_one)(task, spaces, groups) for task in tasks)
    frame = pd.DataFrame(rows)
    if args.targets == "script":
        frame["full - taxonomy"] = frame["full"] - frame["taxonomy"]
    else:
        frame["full - script features"] = frame["full"] - frame["script features"]
        frame["gain of adding e"] = (frame["full + script features"]
                                     - frame["script features"])
    frame.to_csv(OUT / f"probes_{args.targets}.csv", index=False)
    notes = {
        "script": "Out-of-fold, GroupKFold(5) by carrying card. AUC for 0/1 "
                  "targets, accuracy for multi-class (baseline = majority "
                  "class), R² for numeric ones.",
        "profile": "Weighted out-of-fold R² (weight n/(n+5), n the text's "
                   "records of the statistic's kind), GroupKFold(5) by "
                   "carrying card.",
    }
    for metric, part in frame.groupby("metric", sort=False):
        part = part.dropna(axis=1, how="all")
        slug = metric.replace("²", "2").replace(" ", "_")
        write_markdown(part, OUT / f"probes_{args.targets}_{slug}.md",
                       f"Linear probes, {args.targets} targets, {metric}",
                       notes[args.targets])
    print(frame.to_string(index=False, max_colwidth=30))


if __name__ == "__main__":
    main()
