"""Shared harness for gen-4 sealed card-encoder interpretability probes.

Loads the production sealed encoder once, joins the per-card winnability
labels to the converted card text and its cached ``.npz`` embedding,
reconstructs the encoder's own seed-42 train/val split, and fits ridge
probes from the 512-dim text vector to each of the nine shrunk labels.

Also provides the counterfactual scaffolding the later probes need:
re-encoding of edited card text (bit-exact against the cache in ``exact``
mode), a placebo-edit family that changes text without changing meaning,
a nearest-neighbour manifold gate over the full embedding cache, and the
read-off of edited texts in label-SD units.

All probe scripts import from this module; importing it also puts
``scripts/`` on the path, so a probe can reach ``_lib`` after it. Read-only
with respect to the repo and the Y: drive; generated artifacts go to
``output/encoder-probes/``. The paths, the join, the encoder, the embedding
cache, the equivalence classes and the placebo edits live in the
``probe_*`` modules beside this one; this module imports their names back,
so a probe reaches all of them as ``probe_lib.<name>``.
"""

from __future__ import annotations

import functools
import json
import pickle
import sys
from dataclasses import dataclass, fields
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from price_predictor.application.ridge_probes import (  # noqa: F401
    ALPHA_GRID,
    HEADS,
    AlphaSearch,
    HeadProbe,
    ProbeConfig,
    ProbeSet,
    _choose_alpha,
    _pearson,
    _r2,
    _ridge_solve,
    fit_probes,
    head_effective_n,
    head_weight,
    to_logit,
)
from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.tokenizer_store import load_tokenizer  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.jsonio import plain, write_json  # noqa: E402
from _lib.paths import REPO  # noqa: E402, F401
from _lib.stats import bootstrap_ci  # noqa: E402
from probe_classes import (  # noqa: E402, F401
    equivalence_classes,
    token_key,
    variance_decomposition,
)
from probe_embeddings import (  # noqa: E402, F401
    CORPUS_MATRIX,
    corpus_embedding_matrix,
    corpus_key,
    load_embedding_matrix,
    manifold_distance,
)
from probe_encoder import TEXT_DIM, EncoderRunner, read_text  # noqa: E402, F401
from probe_join import (  # noqa: E402, F401
    JOIN_TABLE,
    build_join,
    build_name_index,
    load_forge_hints,
    load_labels,
    load_printing_metadata,
    read_name_line,
    reconstruct_split,
    resolve_card_file,
)
from probe_paths import (  # noqa: E402, F401
    ALLPRINTINGS,
    CARD_TABLE,
    CARDS_PATH,
    ENCODER_CKPT,
    FORGE_HINTS,
    SCRATCH,
    VOCAB_PATH,
    WIN_RATES,
)
from probe_placebo import placebo_edits, subtype_frequencies  # noqa: E402, F401

# The ridge harness, the label reader, and the per-head weighting live in
# ``price_predictor.application.ridge_probes`` so the effects decodability
# battery can import them too; the probe scripts reach them through here.

# Label SDs every probe quotes effects in (one shared figure for the five
# color_lift heads), and the real-card p95 nearest-neighbour cosine distance
# (p0_report) beyond which an edit is off-manifold.
SD = {"score_play": 0.06181, "played_rate": 0.1247, "color_lift": 0.024}
MANIFOLD_GATE = 0.35325




def load_frame(*, features: bool = True, text: bool = False) -> pd.DataFrame:
    """The primary rows of the join table — the frame every probe analyses.

    ``features`` merges in the ``card_table.pkl`` columns the join table
    lacks, keyed by the table's ``card_name`` (kept as a column). ``text``
    adds ``text``: the card's converted text without its ``name:`` line,
    which is what the encoder reads.
    """
    frame = build_join()
    frame = frame[frame["is_primary"]].reset_index(drop=True)
    if features:
        table = pd.read_pickle(CARD_TABLE)
        extra = [c for c in table.columns if c not in set(frame.columns) | {"card_name"}]
        table = table[["card_name", *extra]].drop_duplicates("card_name")
        frame = frame.merge(table, left_on="name", right_on="card_name", how="left")
    if text:
        frame["text"] = [ConvertedCardText(read_text(p)).without_name_line()
                         for p in frame["txt_path"]]
    return frame


def num(frame: pd.DataFrame, col: str, fill: float | None = None) -> np.ndarray:
    """A frame column as floats: unparseable cells NaN, or ``fill`` when given."""
    v = pd.to_numeric(frame[col], errors="coerce")
    return (v if fill is None else v.fillna(fill)).to_numpy(float)


def flag(frame: pd.DataFrame, col: str) -> np.ndarray:
    """A yes/no frame column as bools, a missing cell read as no."""
    return frame[col].fillna(False).astype(bool).to_numpy()


@dataclass(frozen=True)
class Label:
    """One head's shrunk label, its sample weight, and the rows carrying both."""

    y: np.ndarray
    w: np.ndarray
    have: np.ndarray

    def __iter__(self):
        return iter((self.y, self.w, self.have))


def label(frame: pd.DataFrame, head: str) -> Label:
    y = num(frame, f"shrunk_{head}")
    w = num(frame, f"w_{head}")
    return Label(y, w, np.isfinite(y) & (w > 0))


def save_probes(probe_set: ProbeSet, directory: Path = SCRATCH) -> Path:
    path = directory / f"probes_{probe_set.key}.pkl"
    with open(path, "wb") as f:
        pickle.dump(probe_set, f)
    return path


def load_probes(mode: str, weighted: bool, directory: Path = SCRATCH) -> ProbeSet:
    key = f"{mode}_{'w' if weighted else 'u'}"
    with open(directory / f"probes_{key}.pkl", "rb") as f:
        return pickle.load(f)


def predict_labels(
    embeddings: np.ndarray, probe_set: ProbeSet,
) -> pd.DataFrame:
    """Every probe in ``probe_set`` applied to ``embeddings``.

    Returns one column per probe (``played_rate@logit`` stays in logit
    space — invert with a sigmoid if you want a rate).
    """
    return pd.DataFrame({
        name: probe.predict(embeddings) for name, probe in probe_set.probes.items()
    })


def probe_metrics_frame(probe_sets: Iterable[ProbeSet]) -> pd.DataFrame:
    rows = []
    for ps in probe_sets:
        for name, probe in ps.probes.items():
            row = {"mode": ps.mode, "weighted": ps.weighted, "probe": name,
                   "alpha": probe.alpha, "n_fit": probe.n_fit}
            row.update(probe.metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def ridge_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """Ridge with alpha chosen by 5-fold CV, as ``fit_probes`` chooses it.

    Returns ``(alpha, cv_r2, coef, intercept)``.
    """
    w = np.ones(len(y)) if w is None else w
    alpha, cv_r2 = _choose_alpha(X, y, w, AlphaSearch(folds=5))
    coef, b = _ridge_solve(X, y, w, [alpha])[alpha]
    return alpha, cv_r2, coef, b


@dataclass(frozen=True)
class RidgeSplit:
    """A train part and a test part of one ridge fit; ``w_tr`` weights the train rows."""

    X_tr: np.ndarray
    y_tr: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    w_tr: np.ndarray | None = None


def ridge_fit_eval(split: RidgeSplit):
    """Ridge fitted on the train part; returns ``(r2_te, alpha, coef, b)``."""
    alpha, _, coef, b = ridge_fit(split.X_tr, split.y_tr, split.w_tr)
    return _r2(split.y_te, split.X_te @ coef + b), alpha, coef, b


@dataclass(frozen=True)
class HonestSplit:
    """The encoder's train and val masks over the rows, and the rows ``keep``
    admits (all of them when ``None``)."""

    is_train: np.ndarray
    is_val: np.ndarray
    keep: np.ndarray | None = None


def honest_ridge(X: np.ndarray, y: np.ndarray, split: HonestSplit) -> dict:
    """Unweighted ridge fitted on the encoder's train split, scored on its val split.

    Comparable to the p0 honest/unweighted table; ``val_r2`` is the number
    to report.
    """
    keep = np.ones(len(y), dtype=bool) if split.keep is None else split.keep
    good = keep & np.isfinite(y)
    tr = good & split.is_train
    va = good & split.is_val
    alpha, cv_r2, coef, b = ridge_fit(X[tr], y[tr])
    pred_va = X[va] @ coef + b
    return {
        "alpha": alpha,
        "cv_r2": cv_r2,
        "train_r2": _r2(y[tr], X[tr] @ coef + b),
        "val_r2": _r2(y[va], pred_va),
        "val_pearson": _pearson(y[va], pred_va),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "coef": coef,
        "intercept": b,
        "val_mask": va,
        "val_pred": pred_va,
    }


# ── counterfactual read-off ─────────────────────────────────────────────


@functools.cache
def runner() -> EncoderRunner:
    return EncoderRunner()


@functools.cache
def fidelity_probes() -> ProbeSet:
    return load_probes("fidelity", True)


@functools.cache
def reference_cloud() -> tuple[list[str], np.ndarray]:
    return corpus_embedding_matrix()


def predict_sd(embeddings: np.ndarray) -> pd.DataFrame:
    """Fidelity-probe predictions, ``score_play``/``played_rate`` in label-SD units."""
    pred = predict_labels(embeddings, fidelity_probes())
    return pd.DataFrame({
        "score_play": pred["score_play"] / SD["score_play"],
        "played_rate": pred["played_rate"] / SD["played_rate"],
        "cast_lift": pred["cast_lift"],
    })


@dataclass(frozen=True)
class Scored:
    """Read-off of a batch of texts, index-aligned: both headline heads in
    label-SD units, the manifold distance and the nearest real card's row."""

    sp: np.ndarray
    pr: np.ndarray
    dist: np.ndarray
    nearest: np.ndarray

    def _map(self, fn) -> Scored:
        return Scored(*(fn(getattr(self, f.name)) for f in fields(self)))

    def reshape(self, *shape: int) -> Scored:
        return self._map(lambda a: a.reshape(*shape))

    def __getitem__(self, key) -> Scored:
        return self._map(lambda a: a[key])

    @property
    def off(self) -> np.ndarray:
        return self.dist > MANIFOLD_GATE


def score_texts(texts: Sequence[str]) -> Scored:
    """Encode edited card texts in one batched pass and read both heads off them."""
    emb = runner().encode_texts(list(texts), batch_size=96)
    pred = predict_sd(emb)
    dist, nearest = manifold_distance(emb, reference_cloud()[1])
    return Scored(pred["score_play"].to_numpy(), pred["played_rate"].to_numpy(),
                  dist, nearest)


@dataclass(frozen=True)
class Variant:
    """One edited text of a frame card: ``family`` groups the edits, ``row`` is
    the card's frame row and ``variant`` names the edit."""

    family: str
    row: int
    variant: str
    text: str


def score_variants(variants: Sequence[Variant], frame: pd.DataFrame, *,
                   distance: bool = True) -> pd.DataFrame:
    """Encode the variants in one batched pass and read the fidelity probes off them.

    One row per variant: ``family``, ``row``, ``variant``, the card's ``name``,
    ``score_play`` and ``played_rate`` in label units, and with ``distance``
    the ``manifold_dist`` of the edited embedding.
    """
    emb = runner().encode_texts([v.text for v in variants], batch_size=96)
    print(f"encoded {len(variants)} variants on {runner().device}", flush=True)
    pred = predict_labels(emb, fidelity_probes())
    rows = [v.row for v in variants]
    out = pd.DataFrame({
        "family": [v.family for v in variants], "row": rows,
        "variant": [v.variant for v in variants],
        "name": frame["name"].iloc[rows].to_numpy(),
        "score_play": pred["score_play"].to_numpy(),
        "played_rate": pred["played_rate"].to_numpy(),
    })
    if distance:
        out["manifold_dist"] = manifold_distance(emb, reference_cloud()[1])[0]
    return out


def mean_ci(delta: np.ndarray, key: str = "mean_sd", n_boot: int = 4000) -> dict:
    """``{key: mean, ci_lo, ci_hi}`` of a paired per-card difference."""
    lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=7)
    return {key: float(np.mean(delta)), "ci_lo": lo, "ci_hi": hi}


@dataclass(frozen=True)
class ArmDelta:
    """A paired arm-minus-arm difference per base card: the headline ``delta``,
    the played_rate shift ``pr_delta`` and the edited arm's off-manifold mask ``off``."""

    delta: np.ndarray
    pr_delta: np.ndarray | None = None
    off: np.ndarray | None = None


@dataclass(frozen=True)
class ContrastStyle:
    """The column names of a result row and the bootstrap size of its interval."""

    key: str = "mean_sd"
    pr_key: str = "delta_pr"
    n_boot: int = 4000


def contrast_row(arm: ArmDelta, style: ContrastStyle = ContrastStyle(),
                 lead: Mapping[str, object] | None = None) -> dict:
    """One paired arm-minus-arm result row over the base cards.

    ``lead`` names the contrast; ``arm.pr_delta`` and ``arm.off`` add the
    played_rate shift and the off-manifold share.
    """
    delta = arm.delta
    row = {**(lead or {}), "n": len(delta), **mean_ci(delta, style.key, style.n_boot),
           "frac_pos": float((delta > 0).mean())}
    if arm.pr_delta is not None:
        row[style.pr_key] = float(arm.pr_delta.mean())
    if arm.off is not None:
        row["off_manifold"] = float(np.mean(arm.off))
    return row


def score_arms(bases: Sequence[int], arms: Sequence, edit) -> Scored:
    """Every base card under every arm, read in one batch: ``edit(base, arm)``
    gives a text, and the result is shaped ``(len(bases), len(arms))``."""
    return score_texts([edit(b, a) for b in bases for a in arms]).reshape(
        len(bases), len(arms))


@dataclass(frozen=True)
class ContrastPlan:
    """Which arm columns :func:`contrasts` reads against which reference column.

    ``cols`` defaults to the columns after the first, one per label; each row
    names its arm under ``field``, after any extra ``lead`` columns.
    """

    labels: Sequence
    cols: Sequence[int] | None = None
    ref: int = 0
    field: str = "contrast"
    style: ContrastStyle = ContrastStyle()
    lead: Mapping[str, object] = dataclass_field(default_factory=dict)


def contrasts(S: Scored, plan: ContrastPlan) -> list[dict]:
    """One :func:`contrast_row` per arm column against the ``plan.ref`` column."""
    cols = range(1, len(plan.labels) + 1) if plan.cols is None else plan.cols
    ref = plan.ref
    return [contrast_row(ArmDelta(S.sp[:, c] - S.sp[:, ref], S.pr[:, c] - S.pr[:, ref],
                                  S.off[:, c]),
                         plan.style, {**plan.lead, plan.field: label})
            for c, label in zip(cols, plan.labels)]


def write_table(tab: pd.DataFrame | Sequence[dict], name: str) -> None:
    """Write a result table to ``SCRATCH/name`` and print it."""
    tab = pd.DataFrame(tab)
    tab.to_csv(SCRATCH / name, index=False)
    print(tab.to_string(index=False), flush=True)


def write_summary(summary: dict, name: str, *, indent: int = 2) -> None:
    """Write a summary JSON to ``SCRATCH/name`` and print it."""
    write_json(SCRATCH / name, summary, indent=indent)
    print(json.dumps(plain(summary), indent=indent), flush=True)
