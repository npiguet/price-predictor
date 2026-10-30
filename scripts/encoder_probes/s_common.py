"""Shared CPU-only scaffolding for the axis-structure probes (R3/R4/R5/R8/R17/R18).

Never loads the encoder and never touches the GPU: everything here reads the
cached ``.npz`` embeddings and the fitted ridge probes that ``p0_build`` left
in ``output/encoder-probes/``.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pandas as pd
import probe_lib as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

OUT = pl.SCRATCH

COLORS = tuple("WUBRG")
ALLIED = {frozenset(p) for p in (("W", "U"), ("U", "B"), ("B", "R"), ("R", "G"), ("G", "W"))}
ENEMY = {frozenset(p) for p in (("W", "B"), ("U", "R"), ("B", "G"), ("R", "W"), ("G", "U"))}


class ProbeFrame:
    """What every ``s_*`` probe reads: ``join`` (primary rows, card-table
    merged, with ``text`` if asked), the row-aligned ``emb``, both weighted
    probe sets with their predictions, and the encoder's split masks."""

    def __init__(self, text: bool = False) -> None:
        self.join = pl.load_frame(text=text)
        self.emb = pl.load_embedding_matrix(list(self.join["name"]), self.join)
        self.pf = pl.load_probes("fidelity", True)
        self.ph = pl.load_probes("honest", True)
        self.val = (self.join["split"] == "val").to_numpy()
        self.train = (self.join["split"] == "train").to_numpy()

    def __len__(self) -> int:
        return len(self.join)

    @functools.cached_property
    def pred_f(self) -> pd.DataFrame:
        return pl.predict_labels(self.emb, self.pf)

    @functools.cached_property
    def pred_h(self) -> pd.DataFrame:
        return pl.predict_labels(self.emb, self.ph)

    def label(self, head: str) -> pl.Label:
        return pl.label(self.join, head)

    def num(self, col: str, fill: float | None = None) -> np.ndarray:
        return pl.num(self.join, col, fill)


def cosine(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def top_k_cosine(rows: np.ndarray, k: int, chunk: int = 1024) -> np.ndarray:
    """(N, k) indices of each row's k most cosine-similar other rows, nearest
    first. ``rows`` must already be L2-normalised."""
    nb = np.empty((len(rows), k), dtype=np.int64)
    for lo in range(0, len(rows), chunk):
        sims = rows[lo:lo + chunk] @ rows.T
        sims[np.arange(sims.shape[0]), np.arange(lo, min(lo + chunk, len(rows)))] = -np.inf
        part = np.argpartition(-sims, k, axis=1)[:, :k]
        order = np.take_along_axis(sims, part, 1).argsort(axis=1)[:, ::-1]
        nb[lo:lo + chunk] = np.take_along_axis(part, order, 1)
    return nb


def neighbour_mean(y: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """Mean of ``y`` over each row's neighbours, NaN where none has a value."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(y[nb], axis=1)


def card_colors(join: pd.DataFrame) -> np.ndarray:
    """(N, 5) bool: does the card's mana cost contain a pip of this colour."""
    return np.stack([join[f"pip_{c.lower()}"].fillna(0).to_numpy() > 0 for c in COLORS], 1)


def base_controls(join: pd.DataFrame) -> np.ndarray:
    """The ``[1, MV, MV², creature]`` design every feature regression extends
    (missing values read as 0)."""
    mv = pl.num(join, "mv", 0)
    return np.column_stack([np.ones(len(join)), mv, mv ** 2, pl.num(join, "is_creature", 0)])


def flag_columns(join: pd.DataFrame) -> list[str]:
    """The card table's yes/no feature columns the feature sweeps scan."""
    return sorted({c for c in join.columns
                   if c.startswith(("kw_", "ph_", "is_")) and c != "is_primary"}
                  | {"big_tribe", "has_x", "hybrid", "phyrexian", "is_colorless_cost",
                     "has_cost_line"})


def permutation_folds(n: int, folds: int = 5, seed: int = 42) -> np.ndarray:
    """A fold label per row: a seeded permutation cut into ``folds`` near-equal
    parts. Passed to ``oof_predict`` as its groups, each part is one fold."""
    fold = np.empty(n, dtype=np.int64)
    for k, part in enumerate(np.array_split(np.random.default_rng(seed).permutation(n), folds)):
        fold[part] = k
    return fold


def gbm(max_iter: int, learning_rate: float, validation_fraction: float,
        categorical: list[int] | None = None) -> HistGradientBoostingRegressor:
    """The early-stopped gradient-boosting regressor the nonlinear bounds use."""
    return HistGradientBoostingRegressor(
        max_iter=max_iter, learning_rate=learning_rate, early_stopping=True,
        validation_fraction=validation_fraction, random_state=42,
        categorical_features=categorical or None)


# ── R8: the AI-hostile classes against MV × broad-type controls ─────────


def hostile_classes(join: pd.DataFrame) -> dict[str, np.ndarray]:
    def col(c):
        return pl.num(join, c, 0)

    return {
        "fog / prevent damage": (col("ph_fog") + col("ph_prevent_damage")) > 0,
        "sweeper": col("ph_sweeper") > 0,
        "counterspell": col("ph_counterspell") > 0,
        "morph": col("ph_morph") > 0,
        "mana rock": (col("is_artifact") > 0) & (col("is_creature") == 0)
                     & (col("ph_tap_for_mana") > 0),
    }


def class_direction(emb: np.ndarray, members: np.ndarray) -> np.ndarray:
    """Unit normal of a class-balanced logistic split of ``members`` from the rest."""
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
    clf.fit(emb, members.astype(int))
    v = clf.coef_[0]
    return v / np.linalg.norm(v)


def control_cells(join: pd.DataFrame) -> np.ndarray:
    """Matching cell per card: mana value (clipped to 8) × first matching broad type."""
    def col(c):
        return pl.num(join, c, 0)

    type_cell = np.select(
        [col("is_creature") > 0, col("is_land") > 0, col("is_instant") > 0,
         col("is_sorcery") > 0, col("is_artifact") > 0, col("is_enchantment") > 0],
        [0, 1, 2, 3, 4, 5], default=6)
    return np.clip(col("mv"), 0, 8).astype(int) * 10 + type_cell


def matched_deficit(cell, is_class, values, weights=None) -> tuple[float, int]:
    """Cell-weighted mean difference, class − control, over the cells holding
    at least one class card and five controls; each cell counts
    ``min(n_class, n_control)``. Means are weighted by ``weights`` when given.
    Returns ``(deficit, cells used)``."""
    usable = np.isfinite(values) if weights is None else np.isfinite(values) & (weights > 0)

    def mean(m):
        return values[m].mean() if weights is None else np.average(values[m], weights=weights[m])

    tot = tot_w = 0.0
    n_cells = 0
    for c in np.unique(cell):
        m = (cell == c) & usable
        a, b = m & is_class, m & ~is_class
        if a.sum() == 0 or b.sum() < 5:
            continue
        k = min(a.sum(), b.sum())
        tot += k * (mean(a) - mean(b))
        tot_w += k
        n_cells += 1
    return (tot / tot_w if tot_w else np.nan), n_cells


# ── R18: the hand-built nameable feature table ──────────────────────────

CATEGORICAL = ("first_rarity", "first_set_code")


def nameable_blocks(join: pd.DataFrame) -> list[tuple[str, list[tuple[str, np.ndarray]]]]:
    """The nameable feature table as the named blocks the nameability curve
    adds in order, each a list of ``(column, values)``. Every column appears
    once: the flying flags and ``kw_count`` have their own slots and are not
    repeated among the keyword and phrase flags."""
    def cols(names):
        return [(c, pl.num(join, c, 0)) for c in names]

    mv = pl.num(join, "mv", 0)
    fly = ["kw_flying", "kw_flying_anywhere", "ph_flying_grant"]
    kw = sorted(c for c in join.columns if c.startswith("kw_") and c not in fly + ["kw_count"])
    ph = sorted(c for c in join.columns if c.startswith("ph_") and c not in fly)
    year = pl.num(join, "first_year")
    year = np.where(np.isfinite(year), year, np.nanmedian(year))
    return [
        ("MV", [("mv", mv), ("mv2", mv**2)]),
        ("+ type class", cols(("is_creature", "is_instant", "is_sorcery", "is_artifact",
                               "is_enchantment", "is_land", "is_planeswalker", "is_aura",
                               "is_equipment", "is_vehicle", "is_legendary", "is_basic"))),
        ("+ P/T", cols(("power", "toughness"))
         + [("has_pt", np.isfinite(pl.num(join, "power")).astype(float))]),
        ("+ flying", cols(c for c in fly if c in join.columns)),
        ("+ keyword flags", cols(kw + ["kw_count"])),
        ("+ phrase / archetype flags", cols(ph)),
        ("+ pips", cols(("pip_w", "pip_u", "pip_b", "pip_r", "pip_g", "pip_c", "pips_total",
                         "generic", "n_colors", "is_gold", "hybrid", "has_x",
                         "is_colorless_cost", "phyrexian"))),
        ("+ ability-line count", cols(("n_abilities", "n_lines", "n_body_lines", "n_words",
                                       "n_body_words", "n_sym", "n_types", "big_tribe"))),
        ("+ era / rarity", [("first_year", year)] + [
            (c, pd.Categorical(join[c].fillna("unknown")).codes.astype(float))
            for c in CATEGORICAL]),
    ]


def categorical_indices(columns: list[tuple[str, np.ndarray]]) -> list[int]:
    return [i for i, (name, _) in enumerate(columns) if name in CATEGORICAL]
