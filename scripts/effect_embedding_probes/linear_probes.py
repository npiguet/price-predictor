"""Which properties are linearly decodable from the whole embedding.

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from _lib.stats import auc, oof_predict, weighted_r2
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from effect_embedding_probes.common import (  # noqa: E402
    BINARY_SCRIPT,
    CARD,
    NUMERIC_SCRIPT,
    SCRIPT_CATS,
    MarkdownStyle,
    add_probe_arguments,
    design,
    group_means_residual,
    has_taxonomy,
    matrix,
    prepared,
    profile_weight,
    resolve_paths,
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


def standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


def spaces_for(table: pd.DataFrame, key_columns: list[str],
               with_features: bool) -> dict[str, np.ndarray]:
    full = matrix(table, "e_full")
    spaces = {"full": standardize(full)}
    # The taxonomy column only where that baseline's cache exists (FR-072).
    if has_taxonomy(table):
        spaces["taxonomy"] = standardize(matrix(table, "e_tax"))
    spaces["residual"] = standardize(group_means_residual(full, table["api"]))
    spaces["api one-hot"] = design(table, categorical=["api_c"])
    if with_features:
        features = design(table, categorical=SCRIPT_CATS,
                          numeric=list(BINARY_SCRIPT) + list(NUMERIC_SCRIPT)
                          + key_columns)
        spaces["script features"] = features
        spaces["full + script features"] = np.hstack([spaces["full"], features])
    return spaces


def probe_binary(X, y, groups) -> float:
    try:
        pred = oof_predict(LogisticRegression(C=1.0, max_iter=1000), X, y, groups,
                           folds=FOLDS, method="decision_function")
    except ValueError:
        # A training fold holding one class only: scikit-learn refuses to fit it.
        return float("nan")
    return auc(y, pred)


def probe_multiclass(X, y, groups) -> float:
    pred = oof_predict(LogisticRegression(C=1.0, max_iter=500), X, y, groups, folds=FOLDS)
    return float((pred == y).mean())


def probe_numeric(X, y, groups, w=None) -> float:
    ok = np.isfinite(y)
    X, y, groups = X[ok], y[ok], groups[ok]
    w = np.ones(len(y)) if w is None else w[ok]
    pred = oof_predict(Ridge(alpha=10.0), X, y, groups, folds=FOLDS, weight=w)
    return weighted_r2(y, pred, w)


@dataclass(frozen=True)
class Fit:
    """The rows one profile probe fits: the space, the target, the card groups and the weights."""

    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    w: np.ndarray

    def where(self, keep: np.ndarray) -> "Fit":
        """The rows ``keep`` selects."""
        return Fit(self.X[keep], self.y[keep], self.groups[keep], self.w[keep])


def probe_nonlinear(fit: Fit, method: str) -> float:
    """Out-of-fold weighted R² of a k-nearest-neighbour or boosted-tree fit."""
    if method == "knn":
        # k-nearest-neighbour takes no sample weights: its fit is unweighted, only the score is.
        pred = oof_predict(KNeighborsRegressor(n_neighbors=20), fit.X, fit.y, fit.groups,
                           folds=FOLDS)
    else:
        pred = oof_predict(HistGradientBoostingRegressor(max_iter=200, random_state=0),
                           fit.X, fit.y, fit.groups, folds=FOLDS, weight=fit.w)
    return weighted_r2(fit.y, pred, fit.w)


METRICS = {"binary": "AUC", "multiclass": "accuracy", "numeric": "R²", "profile": "weighted R²"}
#: The non-linear probes by label, per space: k-nearest-neighbour is not run on script features.
NONLINEAR_METHODS = {False: {"knn": "knn", "boosted trees": "hgb"},
                     True: {"boosted trees": "hgb"}}
SIMPLE_PROBES = {"binary": probe_binary, "multiclass": probe_multiclass, "numeric": probe_numeric}


def describe(kind: str, y: np.ndarray, w) -> dict:
    """The columns that say what a target is made of: its positives, classes or texts."""
    if kind == "binary":
        return {"positives": int(y.sum())}
    if kind == "multiclass":
        return {"baseline": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
                "classes": int(pd.Series(y).nunique())}
    return {"texts": int(np.isfinite(y).sum() if w is None
                         else (np.isfinite(y) & (w > 0)).sum())}


def profile_scores(space: str, everything: Fit) -> dict:
    """The weighted ridge score of a profile target in one space, and its non-linear probes."""
    fit = everything.where(everything.w > 0)
    scores = {space: probe_numeric(fit.X, fit.y, fit.groups, fit.w)}
    if space in NONLINEAR_SPACES:
        methods = NONLINEAR_METHODS[space == "script features"]
        for label, method in methods.items():
            scores[f"{space} ({label})"] = probe_nonlinear(fit, method)
    return scores


def run_one(task, spaces, groups):
    kind, name, y, w = task
    warnings.simplefilter("ignore", ConvergenceWarning)
    out = {"target": name, "metric": METRICS[kind], **describe(kind, y, w)}
    for space, X in spaces.items():
        if kind == "profile":
            out.update(profile_scores(space, Fit(X, y, groups, w)))
        else:
            out[space] = SIMPLE_PROBES[kind](X, y, groups)
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


def profile_tasks(table: pd.DataFrame) -> list:
    tasks = [profile_task(table, column) for column in table.columns
             if column.startswith(("p_", "mean_", "share_opp", "kind_share_"))]
    return [task for task in tasks if task is not None]


def profile_task(table: pd.DataFrame, column: str):
    """The weighted-ridge task of one statistic column, or ``None`` when too few texts carry it."""
    y = table[column].to_numpy(float)
    w = profile_weight(table, column)
    w = np.where(np.isfinite(y), w, 0.0)
    have = w > 0
    if have.sum() < 1000 or is_rare_share(column, y, w, have):
        return None
    if not column.startswith("p_"):
        y = clipped_mean(y, have)
        if y is None:
            return None
    return ("profile", column, np.nan_to_num(y), w)


def is_rare_share(column: str, y: np.ndarray, w: np.ndarray, have: np.ndarray) -> bool:
    """Whether a ``p_`` statistic is almost never nonzero among the texts that have it."""
    return column.startswith("p_") and np.average(y[have], weights=w[have]) < 0.005


def clipped_mean(y: np.ndarray, have: np.ndarray):
    """``y`` winsorized at its 0.5th/99.5th percentiles, ``None`` when it does not vary.

    A handful of texts reach extreme means (one X-counter ability averages 522
    counters); winsorizing keeps one text from deciding a weighted R².
    """
    low, high = np.quantile(y[have], [0.005, 0.995])
    return np.clip(y, low, high) if high > low else None


def main() -> None:
    parser = add_probe_arguments(
        argparse.ArgumentParser(description=__doc__.splitlines()[0]),
    )
    parser.add_argument("--targets", choices=("script", "profile"), default="script")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    paths = resolve_paths(args)
    table, key_columns = prepared(paths)
    groups = table["group"].to_numpy()
    spaces = spaces_for(table, key_columns, with_features=args.targets == "profile")
    tasks = (script_tasks(table, key_columns) if args.targets == "script"
             else profile_tasks(table))
    print(f"{len(tasks)} targets x {len(spaces)} spaces", flush=True)
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_one)(task, spaces, groups) for task in tasks)
    frame = pd.DataFrame(rows)
    if args.targets == "script":
        if "taxonomy" in frame:
            frame["full - taxonomy"] = frame["full"] - frame["taxonomy"]
    else:
        frame["full - script features"] = frame["full"] - frame["script features"]
        frame["gain of adding e"] = (frame["full + script features"]
                                     - frame["script features"])
    frame.to_csv(paths.out / f"probes_{args.targets}.csv", index=False)
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
        write_markdown(part, paths.out / f"probes_{args.targets}_{slug}.md",
                       f"Linear probes, {args.targets} targets, {metric}",
                       MarkdownStyle(note=notes[args.targets]))
    print(frame.to_string(index=False, max_colwidth=30))


if __name__ == "__main__":
    from effect_embedding_probes.common import main_with_manifest

    main_with_manifest(main, __file__)
