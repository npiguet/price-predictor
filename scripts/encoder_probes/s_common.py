"""Shared CPU-only scaffolding for the axis-structure probes (R3/R4/R5/R8/R17/R18).

Never loads the encoder and never touches the GPU: everything here reads the
cached ``.npz`` embeddings and the fitted ridge probes that ``p0_build`` left
in ``output/encoder-probes/``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import probe_lib as pl

OUT = pl.SCRATCH

SD = {
    "score_play": 0.0618,
    "score_draw": 0.0618,
    "played_rate": 0.1247,
    "cast_lift": 0.0845,
    "color_lift": 0.024,
}

COLORS = tuple("WUBRG")
ALLIED = {frozenset(p) for p in (("W", "U"), ("U", "B"), ("B", "R"), ("R", "G"), ("G", "W"))}
ENEMY = {frozenset(p) for p in (("W", "B"), ("U", "R"), ("B", "G"), ("R", "W"), ("G", "U"))}


def load_all(text: bool = False) -> dict:
    """``join`` (primary rows, card-table merged, with ``text`` if asked),
    row-aligned ``emb``, both weighted probe sets and the split masks."""
    join = pl.load_frame(text=text)
    return {
        "join": join,
        "emb": pl.load_embedding_matrix(list(join["name"]), join),
        "pf": pl.load_probes("fidelity", True),
        "ph": pl.load_probes("honest", True),
        "val": (join["split"] == "val").to_numpy(),
        "train": (join["split"] == "train").to_numpy(),
    }


def num(join: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(join[col], errors="coerce").to_numpy(float)


def flag(join: pd.DataFrame, col: str) -> np.ndarray:
    """A numeric column with missing values read as 0."""
    return np.nan_to_num(num(join, col))


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


# ── R8: the AI-hostile classes against MV × broad-type controls ─────────


def hostile_classes(join: pd.DataFrame) -> dict[str, np.ndarray]:
    def col(c):
        return flag(join, c)

    return {
        "fog / prevent damage": (col("ph_fog") + col("ph_prevent_damage")) > 0,
        "sweeper": col("ph_sweeper") > 0,
        "counterspell": col("ph_counterspell") > 0,
        "morph": col("ph_morph") > 0,
        "mana rock": (col("is_artifact") > 0) & (col("is_creature") == 0)
                     & (col("ph_tap_for_mana") > 0),
    }


def control_cells(join: pd.DataFrame) -> np.ndarray:
    """Matching cell per card: mana value (clipped to 8) × first matching broad type."""
    def col(c):
        return flag(join, c)

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
        return [(c, flag(join, c)) for c in names]

    mv = flag(join, "mv")
    fly = ["kw_flying", "kw_flying_anywhere", "ph_flying_grant"]
    kw = sorted(c for c in join.columns if c.startswith("kw_") and c not in fly + ["kw_count"])
    ph = sorted(c for c in join.columns if c.startswith("ph_") and c not in fly)
    year = num(join, "first_year")
    year = np.where(np.isfinite(year), year, np.nanmedian(year))
    return [
        ("MV", [("mv", mv), ("mv2", mv**2)]),
        ("+ type class", cols(("is_creature", "is_instant", "is_sorcery", "is_artifact",
                               "is_enchantment", "is_land", "is_planeswalker", "is_aura",
                               "is_equipment", "is_vehicle", "is_legendary", "is_basic"))),
        ("+ P/T", cols(("power", "toughness"))
         + [("has_pt", np.isfinite(num(join, "power")).astype(float))]),
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
