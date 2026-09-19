"""Which API types the embedding merges, and whether it reads amounts within one.

Two follow-ups to ``linear_probes.py`` on the shipping embedding:

1. **Merged API types.** A multinomial logistic probe from ``e`` to the API
   type (types under 20 texts pooled), out of fold, grouped by carrying card.
   Reports the API-type pairs the probe confuses most often, as the share of
   the first type's texts it assigns to the second — two types the embedding
   places together read as one effect. Also reports the same accuracy from a
   20-nearest-neighbour classifier, which is not limited to linear
   boundaries.
2. **Amounts within an API type.** For each numeric amount (damage, counters,
   power, toughness, cards, life, tokens, activation mana), the out-of-fold
   R² of a ridge probe after subtracting each API type's mean from both the
   embedding and the amount, so only variation *within* a type is scored.
   Every text is used; a text without the amount has 0, and a variable
   amount (``X``) is left out, and each amount is clipped at its 0.1st/99.9th
   percentiles. Run on the full and taxonomy spaces.

Run: ``python scripts/effect_embedding_probes/type_merging.py`` (CPU, ~3 min).
Writes ``api_confusion.md`` and ``amounts_within_type.md``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    OUT,
    group_means_residual,
    matrix,
    prepared,
    winsorized,
    write_markdown,
)

AMOUNTS = ("amt_damage", "amt_counters", "amt_power", "amt_toughness",
           "amt_cards", "amt_life", "amt_tokens", "cost_mana")


def oof_predict(model_factory, X, y, groups) -> np.ndarray:
    pred = np.empty(len(y), dtype=object)
    for train, test in GroupKFold(5).split(X, y, groups):
        model = model_factory().fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred


def within_r2(X, y, api, groups) -> float:
    ok = np.isfinite(y)
    frame = pd.DataFrame({"y": y[ok], "api": api[ok]})
    yr = (frame.y - frame.groupby("api").y.transform("mean")).to_numpy()
    Xr = group_means_residual(X[ok], pd.Series(api[ok]))
    pred = np.zeros(len(yr))
    for train, test in GroupKFold(5).split(Xr, yr, groups[ok]):
        pred[test] = Ridge(alpha=10.0).fit(Xr[train], yr[train]).predict(Xr[test])
    ss = float((yr ** 2).sum())
    return 1.0 - float(((yr - pred) ** 2).sum()) / ss if ss else float("nan")


def main() -> None:
    warnings.simplefilter("ignore", ConvergenceWarning)
    table, _ = prepared()
    groups = table["group"].to_numpy()
    X = StandardScaler().fit_transform(matrix(table, "e_full"))
    y = table["api_c"].to_numpy(object)
    linear = oof_predict(lambda: LogisticRegression(max_iter=500), X, y, groups)
    knn = oof_predict(lambda: KNeighborsClassifier(n_neighbors=20), X, y, groups)
    counts = pd.Series(y).value_counts()
    confusion = pd.crosstab(pd.Series(y, name="true"), pd.Series(linear, name="pred"))
    pairs = []
    for true in confusion.index:
        for pred in confusion.columns:
            if true != pred and counts[true] >= 100:
                pairs.append({"API type": true, "texts": int(counts[true]),
                              "probe says": pred,
                              "share": confusion.loc[true, pred] / counts[true]})
    pairs = pd.DataFrame(pairs).sort_values("share", ascending=False).head(30)
    per_type = pd.DataFrame({
        "API type": counts.index,
        "texts": counts.to_numpy(),
        "linear recall": [float((linear[y == t] == t).mean()) for t in counts.index],
        "20-NN recall": [float((knn[y == t] == t).mean()) for t in counts.index],
    }).head(40)
    note = (f"Out-of-fold over {len(y)} texts, GroupKFold(5) by carrying card. "
            f"Accuracy: linear {np.mean(linear == y):.3f}, 20-nearest-neighbour "
            f"{np.mean(knn == y):.3f}, majority class "
            f"{counts.iloc[0] / len(y):.3f}.")
    write_markdown(pairs, OUT / "api_confusion.md",
                   "API-type pairs the linear probe confuses most", note)
    write_markdown(per_type, OUT / "api_recall.md",
                   "Per-type recall of the API-type probes (40 largest types)", note)
    print(note)
    print(pairs.head(15).to_string(index=False))

    rows = []
    api = table["api"].to_numpy(object)
    for column in AMOUNTS:
        values = winsorized(table[column].to_numpy(float))
        row = {"amount": column,
               "texts with a nonzero value": int((np.nan_to_num(values) != 0).sum())}
        for space, source in (("full", "e_full"), ("taxonomy", "e_tax")):
            row[f"{space}, within type"] = within_r2(
                matrix(table, source), values, api, groups)
        rows.append(row)
    frame = pd.DataFrame(rows)
    write_markdown(frame, OUT / "amounts_within_type.md",
                   "Amounts read within an API type (out-of-fold R²)",
                   "Both the embedding and the amount have their API-type mean "
                   "subtracted; ridge alpha 10, GroupKFold(5) by carrying card.")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
