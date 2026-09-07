"""Ridge-probe harness for linear decodability of a card embedding.

Fits one weighted ridge regression per winnability label from an embedding
matrix, cross-validating the penalty over a fixed grid. It answers "how much of
each label is linearly readable from this vector", which is what makes two
encoders comparable on the same feature table.

This is the harness
[`experiments/2026-08-28-encoder-preferences.md`](../../../experiments/2026-08-28-encoder-preferences.md)
was run with, moved out of `scripts/encoder_probes/probe_lib.py` so it can be
imported: `scripts/` is not a package, and importing that module runs
side effects and resolves a NAS path at module scope. Every path here is a
parameter and nothing runs at import time.

`fit_probes` takes an arbitrary embedding matrix and a label table row-aligned
with it, so the caller decides which encoder and which cards are in scope.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Head order matches ``sealed.application.train_encoder``'s head names and the
# column order of ``cards-win-rates.txt``.
HEADS: tuple[str, ...] = (
    "score_play", "score_draw", "played_rate", "cast_lift",
    "color_lift_W", "color_lift_U", "color_lift_B", "color_lift_R",
    "color_lift_G",
)
COUNTER_COLUMNS: tuple[str, ...] = (
    "wins_when_played", "wins_when_in_deck",
    "losses_when_played", "losses_when_in_deck",
)

ALPHA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LOGIT_CLIP = 1e-3
SHRINKAGE_K = 20.0  # train-encoder --shrinkage-k default, used for the labels


# ── labels ──────────────────────────────────────────────────────────────


def load_labels(path: Path) -> dict[str, dict]:
    """``cards-win-rates.txt`` → ``name -> {column: value}``.

    The four counters come back as ``int``; the eighteen raw/shrunk label cells
    as ``float`` or ``None`` (an empty cell means "no signal", not "neutral
    signal" — see the file-format contract in CLAUDE.md). The file is UTF-8;
    reading it as cp1252 crashes on accented card names.

    The path is a required argument: the table lives wherever the operator's
    training data does, which for this project is a NAS mount that a test run
    must not depend on.
    """
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split(";")
        for line in handle:
            parts = line.rstrip("\n").split(";")
            if len(parts) != len(header):
                continue
            record: dict = {}
            for key, value in zip(header[1:], parts[1:]):
                if key in COUNTER_COLUMNS:
                    record[key] = int(value)
                else:
                    record[key] = float(value) if value != "" else None
            rows[parts[0]] = record
    return rows


def head_effective_n(record: dict, head: str, k: float = SHRINKAGE_K) -> float | None:
    """Effective observation count behind one head's cell, or None.

    ``played_rate`` and ``cast_lift`` read straight off the four counters. The
    remaining seven heads use denominators (@play / @draw / per-color slices)
    that ``cards-win-rates.txt`` does not carry, but every one of them is
    recoverable from the raw/shrunk pair: both share a numerator, so
    ``shrunk / raw == n / (n + k)`` and ``n = k · shrunk / (raw − shrunk)``.
    That identity degrades when the numerator is near zero (the cells are
    rounded to five decimals), so it is only used when ``|raw|`` is comfortably
    above the rounding floor; the fallback is ``n_in_deck / 2`` for the two
    @play/@draw heads (each game puts exactly one side on the play) and
    ``n_in_deck`` for a color lift.
    """
    in_deck = record["wins_when_in_deck"] + record["losses_when_in_deck"]
    if head == "played_rate":
        return float(in_deck)
    if head == "cast_lift":
        played = record["wins_when_played"] + record["losses_when_played"]
        return float(min(played, in_deck - played))

    if head in ("score_play", "score_draw"):
        raw, shrunk = record[f"raw_{head}"], record[f"shrunk_{head}"]
        fallback = in_deck / 2.0
    else:
        color = head[-1]
        raw = record[f"raw_color_lift_{color}"]
        shrunk = record[f"shrunk_color_lift_{color}"]
        # color_lift subtracts the card's overall score, so undo that first.
        overall_numerator = record["wins_when_played"] - record["losses_when_played"]
        if in_deck == 0:
            return None
        if raw is not None:
            raw = raw + overall_numerator / in_deck
        if shrunk is not None:
            shrunk = shrunk + overall_numerator / (in_deck + k)
        fallback = float(in_deck)
    if raw is None or shrunk is None:
        return None
    if abs(raw) < 5e-4 or raw == shrunk:
        return fallback
    ratio = shrunk / raw
    if not 0.0 < ratio < 1.0:
        return fallback
    return float(k * ratio / (1.0 - ratio))


def head_weight(n: float | None, k: float = SHRINKAGE_K) -> float:
    """Per-head sample weight ``n / (n + k)``; 0.0 for a dead cell."""
    if n is None or n <= 0:
        return 0.0
    return float(n / (n + k))


def build_label_table(
    names: Sequence[str],
    *,
    win_rates_path: Path,
    val_names: Collection[str] = (),
    shrinkage_k: float = SHRINKAGE_K,
) -> pd.DataFrame:
    """One label row per entry of ``names``, in that order.

    Row *i* describes ``names[i]``, so the table stays aligned with whatever
    embedding matrix the caller built from the same name list. A name with no
    row in the win-rate table gets NaN labels and zero weights, which is
    exactly the mask :func:`fit_probes` drops — the caller does not have to
    pre-filter its cards.

    Columns are the ones :func:`fit_probes` reads (``split``, ``is_primary``,
    and a ``shrunk_<head>`` / ``w_<head>`` pair per head) plus the raw cells and
    counters for callers that want them. ``val_names`` names the validation
    split; every other name is ``train``. The split is a parameter rather than
    something recomputed here, because the split that makes a comparison honest
    belongs to the model being probed.
    """
    labels = load_labels(win_rates_path)
    val = set(val_names)
    rows: list[dict] = []
    for name in names:
        record = labels.get(name)
        row: dict = {
            "name": name,
            "split": "val" if name in val else "train",
            "has_label": record is not None,
        }
        if record is None:
            for column in COUNTER_COLUMNS:
                row[column] = 0
            row["n_in_deck"] = 0
            row["n_played"] = 0
            for head in HEADS:
                row[f"raw_{head}"] = float("nan")
                row[f"shrunk_{head}"] = float("nan")
                row[f"n_{head}"] = 0.0
                row[f"w_{head}"] = 0.0
            rows.append(row)
            continue
        row.update(record)
        row["n_in_deck"] = record["wins_when_in_deck"] + record["losses_when_in_deck"]
        row["n_played"] = record["wins_when_played"] + record["losses_when_played"]
        for head in HEADS:
            n = head_effective_n(record, head, k=shrinkage_k)
            row[f"n_{head}"] = n if n is not None else 0.0
            row[f"w_{head}"] = head_weight(n, k=shrinkage_k)
        rows.append(row)

    table = pd.DataFrame(rows)
    # A name repeated in ``names`` would fit the same card several times and
    # weight it accordingly; keep the busiest occurrence as primary.
    table["is_primary"] = True
    duplicated = table["name"].duplicated(keep=False)
    if duplicated.any():
        for _, group in table[duplicated].groupby("name"):
            keep = group["n_in_deck"].idxmax()
            table.loc[group.index.difference([keep]), "is_primary"] = False
    return table


# ── ridge probes ────────────────────────────────────────────────────────


def to_logit(p: np.ndarray, clip: float = LOGIT_CLIP) -> np.ndarray:
    q = np.clip(p, clip, 1.0 - clip)
    return np.log(q / (1.0 - q))


@dataclass
class HeadProbe:
    """One fitted ridge probe: its weights, an intercept, and its metrics."""

    head: str
    space: str            # "linear" or "logit" (played_rate only)
    coef: np.ndarray      # (embedding_dim,)
    intercept: float
    alpha: float
    n_fit: int
    metrics: dict = field(default_factory=dict)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings @ self.coef + self.intercept


@dataclass
class ProbeSet:
    mode: str             # "fidelity" (all cards) or "honest" (train only)
    weighted: bool
    probes: dict[str, HeadProbe]

    @property
    def key(self) -> str:
        return f"{self.mode}_{'w' if self.weighted else 'u'}"


def _ridge_solve(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, alphas: Sequence[float],
) -> dict[float, tuple[np.ndarray, float]]:
    """Weighted ridge for every alpha at once (one eigendecomposition).

    Centres X and y under the sample weights so the intercept is exact and
    unpenalised, forms the weighted Gram matrix once, and reuses its
    eigendecomposition across the alpha grid.
    """
    sw = w / w.sum()
    xm = sw @ X
    ym = float(sw @ y)
    Xc = X - xm
    yc = y - ym
    Xw = Xc * w[:, None]
    gram = Xc.T @ Xw
    rhs = Xw.T @ yc
    evals, evecs = np.linalg.eigh(gram)
    proj = evecs.T @ rhs
    out: dict[float, tuple[np.ndarray, float]] = {}
    for alpha in alphas:
        coef = evecs @ (proj / (evals + alpha))
        out[float(alpha)] = (coef, ym - float(xm @ coef))
    return out


def _r2(y: np.ndarray, pred: np.ndarray, w: np.ndarray | None = None) -> float:
    if w is None:
        w = np.ones_like(y)
    mean = float((w * y).sum() / w.sum())
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - mean) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _choose_alpha(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, folds: int = 5, seed: int = 42,
    alphas: Sequence[float] = ALPHA_GRID,
) -> tuple[float, float]:
    """K-fold CV over ``alphas`` (:data:`ALPHA_GRID` by default).

    Returns ``(alpha, cv_r2)``.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    scores = {float(a): 0.0 for a in alphas}
    for part in parts:
        mask = np.ones(len(y), dtype=bool)
        mask[part] = False
        fits = _ridge_solve(X[mask], y[mask], w[mask], alphas)
        for alpha, (coef, b) in fits.items():
            pred = X[part] @ coef + b
            scores[alpha] += _r2(y[part], pred, w[part]) / folds
    best = max(scores, key=lambda a: scores[a])
    return best, scores[best]


def fit_probes(
    join: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    mode: str = "fidelity",
    weighted: bool = True,
    heads: Sequence[str] = HEADS,
    folds: int = 5,
    alphas: Sequence[float] = ALPHA_GRID,
) -> ProbeSet:
    """Ridge probes from an embedding matrix to each shrunk label.

    ``mode='fidelity'`` fits on every joined card — the read-off model for
    counterfactual edits, where generalization is not the claim.
    ``mode='honest'`` fits only on the train split, so its metrics on the val
    split are an unrecycled generalization number.

    ``weighted`` applies the training objective's ``n/(n+k)`` per-head sample
    weight. ``played_rate`` is fitted twice — linearly and in logit space
    (``played_rate@logit``), since a rate bounded in [0, 1] with mass near both
    ends is not a linear target.

    ``embeddings`` must be row-aligned with ``join``, which needs the columns
    :func:`build_label_table` produces.
    """
    if mode not in ("fidelity", "honest"):
        raise ValueError(f"mode must be 'fidelity' or 'honest', got {mode!r}")
    primary = join["is_primary"].to_numpy()
    is_train = (join["split"] == "train").to_numpy()
    is_val = (join["split"] == "val").to_numpy()
    fit_base = primary & (is_train if mode == "honest" else np.ones_like(primary))

    probes: dict[str, HeadProbe] = {}
    for head in heads:
        y_all = pd.to_numeric(join[f"shrunk_{head}"], errors="coerce").to_numpy(float)
        w_all = join[f"w_{head}"].to_numpy(float)
        have = np.isfinite(y_all) & (w_all > 0)
        spaces = [("linear", y_all)]
        if head == "played_rate":
            spaces.append(("logit", to_logit(y_all)))
        for space, target in spaces:
            fit_mask = fit_base & have
            X, y = embeddings[fit_mask], target[fit_mask]
            w = w_all[fit_mask] if weighted else np.ones(fit_mask.sum())
            alpha, cv_r2 = _choose_alpha(X, y, w, folds=folds, alphas=alphas)
            coef, b = _ridge_solve(X, y, w, [alpha])[alpha]
            metrics = {"cv_r2": cv_r2, "in_sample_r2": _r2(y, X @ coef + b, w)}
            for split_name, split_mask in (("train", is_train), ("val", is_val)):
                m = primary & have & split_mask
                if m.sum() < 2:
                    continue
                pred = embeddings[m] @ coef + b
                mw = w_all[m] if weighted else np.ones(int(m.sum()))
                metrics[f"{split_name}_r2"] = _r2(target[m], pred, mw)
                metrics[f"{split_name}_pearson"] = _pearson(target[m], pred)
                metrics[f"{split_name}_n"] = int(m.sum())
            name = head if space == "linear" else f"{head}@logit"
            probes[name] = HeadProbe(
                head=head, space=space, coef=coef, intercept=b, alpha=alpha,
                n_fit=int(fit_mask.sum()), metrics=metrics,
            )
    return ProbeSet(mode=mode, weighted=weighted, probes=probes)
