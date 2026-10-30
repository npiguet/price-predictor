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
``output/encoder-probes/``.
"""

from __future__ import annotations

import functools
import json
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from price_predictor.application.ridge_probes import (  # noqa: F401
    ALPHA_GRID,
    HEADS,
    HeadProbe,
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
from price_predictor.application.ridge_probes import load_labels as _load_labels
from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.tokenizer_store import load_tokenizer
from sealed.infrastructure.encoder_store import SealedEncoderStore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.jsonio import plain, write_json  # noqa: E402
from _lib.paths import REPO, SEALED_TRAINING_DATA  # noqa: E402
from _lib.stats import bootstrap_ci  # noqa: E402
from c_common import ABILITY_PREFIXES, subtypes  # noqa: E402

ENCODER_CKPT = REPO / "models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt"
VOCAB_PATH = REPO / "models/sealed/encoder/vocab.txt"
CARDS_PATH = REPO / "output/cardsfolder-512"
ALLPRINTINGS = REPO / "resources/AllPrintings.json"
FORGE_HINTS = REPO / "output/scorer-probes/forge_hints.csv"
WIN_RATES = SEALED_TRAINING_DATA / "matches-bo1" / "cards-win-rates.txt"
SCRATCH = REPO / "output" / "encoder-probes"  # generated artifacts (gitignored)
SCRATCH.mkdir(parents=True, exist_ok=True)
CARD_TABLE = SCRATCH / "card_table.pkl"  # written by p0b_card_table.py

# The ridge harness, the label reader, and the per-head weighting live in
# ``price_predictor.application.ridge_probes`` so the effects decodability
# battery can import them too; the probe scripts reach them through here.

TEXT_DIM = 512          # pooled text vector; the cached .npz is 512 + 32 features

# Label SDs every probe quotes effects in (one shared figure for the five
# color_lift heads), and the real-card p95 nearest-neighbour cosine distance
# (p0_report) beyond which an edit is off-manifold.
SD = {"score_play": 0.06181, "played_rate": 0.1247, "color_lift": 0.024}
MANIFOLD_GATE = 0.35325


# ── labels ──────────────────────────────────────────────────────────────


def load_labels(path: Path = WIN_RATES) -> dict[str, dict]:
    """``cards-win-rates.txt`` → ``name -> {column: value}``, from the NAS.

    Default-path wrapper over ``ridge_probes.load_labels``, which takes the
    path as a required argument so it can run without this drive mounted.
    """
    return _load_labels(path)


# ── name → converted-file join ──────────────────────────────────────────
#
# The repo's ConvertedCardLocator resolves names through filename munging
# plus a prefix search, which silently mis-joins a handful of cards (the
# prefix search maps "Undercity" onto ``undercity_dire_rat.txt``, and a
# stale FILENAME_CORRECTIONS entry maps "Village Watch" onto
# ``scorned_villager_moonscarred_werewolf.txt``). The converted corpus
# carries the authoritative name on its ``name:`` line — 32,606 files,
# 32,606 distinct names — so the join here is built from that reverse
# index instead, with a punctuation-insensitive fallback.

def _norm_name(name: str) -> str:
    """Accent-folded, lowercased, whitespace-collapsed name key."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(ascii_name.lower().split())


def _aggressive_name(name: str) -> str:
    """``_norm_name`` with every non-alphanumeric run collapsed to a space.

    Recovers ``"With Great Power..."`` ↔ ``"with great power . . ."`` and
    ``"Start Your Engines!"`` ↔ ``"start your engines"``.
    """
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", _norm_name(name)).split())


def read_name_line(path: Path) -> str:
    """First ``name:`` value of a converted card file (already lowercase)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("name:"):
                return line[5:].strip()
    return ""


def build_name_index(cards_path: Path = CARDS_PATH):
    """Scan the converted corpus once → ``(exact_index, aggressive_index)``.

    ``exact_index`` maps ``_norm_name`` → path (the corpus is 1:1 on this
    key). ``aggressive_index`` maps ``_aggressive_name`` → list of paths.
    """
    exact: dict[str, Path] = {}
    aggressive: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(cards_path.rglob("*.txt")):
        name = read_name_line(path)
        exact[_norm_name(name)] = path
        aggressive[_aggressive_name(name)].append(path)
    return exact, dict(aggressive)


def resolve_card_file(
    name: str, exact: dict[str, Path], aggressive: dict[str, list[Path]],
) -> tuple[Path | None, str]:
    """Resolve one label name to its converted ``.txt``.

    Returns ``(path, method)``; ``method`` is one of ``name-exact``,
    ``front-face`` (``"A // B"`` rows, whose file carries the front face
    on its ``name:`` line), ``punct-norm``, ``front-punct``, or
    ``unresolved``.
    """
    key = _norm_name(name)
    if key in exact:
        return exact[key], "name-exact"
    front = name.split(" // ", 1)[0].strip() if " // " in name else None
    if front:
        if _norm_name(front) in exact:
            return exact[_norm_name(front)], "front-face"
    hits = aggressive.get(_aggressive_name(name), ())
    if len(hits) == 1:
        return hits[0], "punct-norm"
    if front:
        hits = aggressive.get(_aggressive_name(front), ())
        if len(hits) == 1:
            return hits[0], "front-punct"
    return None, "unresolved"


# ── the encoder's own train/val split ───────────────────────────────────


def reconstruct_split(labels: dict[str, dict]) -> tuple[set[str], set[str]]:
    """Rebuild ``train_encoder._split_cards`` from the label snapshot.

    The snapshot ``cards-win-rates.txt`` *is* the label map the trainer
    split (it is written immediately before the split, after the
    missing-text drop), and the split only reads the shrunk signed cells,
    so importing the real function on reconstructed ``CardLabels`` gives
    the run's own partition. ``p0_build`` asserts the sizes against the
    training log.
    """
    from sealed.application.train_encoder import (  # noqa: I001 — local: heavy import
        CardCounters,
        CardLabels,
        _split_cards,
    )

    label_map = {}
    for name, rec in labels.items():
        label_map[name] = CardLabels(
            card_name=name,
            counters=CardCounters(),
            raw_score_play=rec["raw_score_play"],
            shrunk_score_play=rec["shrunk_score_play"],
            raw_score_draw=rec["raw_score_draw"],
            shrunk_score_draw=rec["shrunk_score_draw"],
            raw_played_rate=rec["raw_played_rate"],
            shrunk_played_rate=rec["shrunk_played_rate"],
            raw_cast_lift=rec["raw_cast_lift"],
            shrunk_cast_lift=rec["shrunk_cast_lift"],
            raw_color_lift={c: rec[f"raw_color_lift_{c}"] for c in "WUBRG"},
            shrunk_color_lift={c: rec[f"shrunk_color_lift_{c}"] for c in "WUBRG"},
        )
    train, val = _split_cards(label_map)
    return set(train), set(val)


# ── external metadata ───────────────────────────────────────────────────


def load_forge_hints(path: Path = FORGE_HINTS) -> pd.DataFrame | None:
    """The sibling scorer study's Forge-side hints, or None if absent.

    Columns: ``name``, ``ai_remove_deck`` (Forge's AI blacklist flag),
    ``ai_remove_deck_kind``, ``draft_rank`` / ``draft_rank_best`` /
    ``draft_rank_n_sets`` (human draft rankings), ``mv``, ``is_creature``.
    Regenerate with ``scripts/scorer_probes/forge_hints.py`` (needs the
    sibling ``../forge`` checkout) if it is missing.
    """
    if not path.exists():
        return None
    return pd.read_csv(path)


def _hint_key(name: str, keys: set[str]) -> str:
    """Name key that hits ``keys``, falling back to the ``"A // B"`` front face."""
    key = _norm_name(name)
    if key in keys or " // " not in name:
        return key
    front = _norm_name(name.split(" // ", 1)[0])
    return front if front in keys else key


def _iter_printing_sets(path: Path):
    """Yield ``(set_code, set_object)`` from AllPrintings.json, set by set.

    ``json.load`` on the whole 630 MB file materialises several GB of
    Python objects; decoding one set at a time with ``raw_decode`` keeps
    only the source string plus one set resident.
    """
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    i = text.index('"data"')
    i = text.index(":", i) + 1
    while text[i].isspace():
        i += 1
    assert text[i] == "{", "unexpected AllPrintings layout"
    i += 1
    while True:
        while text[i].isspace() or text[i] == ",":
            i += 1
        if text[i] == "}":
            return
        code, i = decoder.raw_decode(text, i)
        while text[i].isspace():
            i += 1
        assert text[i] == ":"
        i += 1
        obj, i = decoder.raw_decode(text, i)
        yield code, obj


def load_printing_metadata(
    path: Path = ALLPRINTINGS, cache: Path | None = None,
) -> dict[str, dict]:
    """``name -> {set_code, release_date, year, rarity}`` for the *earliest*
    printing of every card, keyed by both the full name and each face name.

    Cached as a pickle in ``SCRATCH`` — the streaming scan takes a couple
    of minutes.
    """
    cache = cache or (SCRATCH / "printings.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    best: dict[str, dict] = {}
    for code, obj in _iter_printing_sets(path):
        release = obj.get("releaseDate") or "9999-99-99"
        for card in obj.get("cards", ()):
            rarity = card.get("rarity", "")
            keys = {card.get("name", ""), card.get("faceName", "")}
            keys.discard("")
            for raw_key in keys:
                key = _norm_name(raw_key)
                prev = best.get(key)
                if prev is None or release < prev["release_date"]:
                    best[key] = {
                        "set_code": code,
                        "release_date": release,
                        "year": int(release[:4]) if release[:4].isdigit() else None,
                        "rarity": rarity,
                    }
    with open(cache, "wb") as f:
        pickle.dump(best, f)
    return best


# ── the join table ──────────────────────────────────────────────────────

JOIN_TABLE = SCRATCH / "join_table.pkl"


def build_join(
    *, force: bool = False, with_printings: bool = True,
) -> pd.DataFrame:
    """Build (and cache) the one-row-per-label-card join table.

    Columns: ``name``, ``txt_path``, ``npz_path``, ``join_method``,
    ``split`` (``train``/``val``, the encoder's own seed-42 partition),
    the four counters, all eighteen raw/shrunk label cells, ``n_in_deck``,
    ``n_played``, a per-head ``n_<head>`` / ``w_<head>`` pair, the
    ``forge_hints`` columns, earliest-printing ``set_code`` / ``year`` /
    ``rarity``, and the join diagnostics ``dup_group`` (files claimed by
    more than one label row) and ``is_primary`` (False on the lower-``n``
    member of such a pair — probes should filter on it).

    Unresolved label rows are *not* in the table; ``build_join`` writes
    them to ``SCRATCH/unjoined.csv`` instead.
    """
    if JOIN_TABLE.exists() and not force:
        return pd.read_pickle(JOIN_TABLE)

    labels = load_labels()
    exact, aggressive = build_name_index()
    train_names, val_names = reconstruct_split(labels)

    rows: list[dict] = []
    unjoined: list[dict] = []
    for name, rec in labels.items():
        path, method = resolve_card_file(name, exact, aggressive)
        if path is None:
            unjoined.append({"name": name, "reason": "no converted card file",
                             "n_in_deck": rec["wins_when_in_deck"] + rec["losses_when_in_deck"]})
            continue
        npz = path.with_suffix(".npz")
        row = {
            "name": name,
            "txt_path": str(path),
            "npz_path": str(npz) if npz.exists() else None,
            "join_method": method,
            "split": "train" if name in train_names else "val",
        }
        row.update(rec)
        row["n_in_deck"] = rec["wins_when_in_deck"] + rec["losses_when_in_deck"]
        row["n_played"] = rec["wins_when_played"] + rec["losses_when_played"]
        for head in HEADS:
            n = head_effective_n(rec, head)
            row[f"n_{head}"] = n
            row[f"w_{head}"] = head_weight(n)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Join diagnostics: files claimed by more than one label row are the
    # same card spelled two ways across corpus eras; keep the busier row.
    counts = df.groupby("txt_path")["name"].transform("size")
    df["dup_group"] = counts > 1
    df["is_primary"] = True
    for path, group in df[df["dup_group"]].groupby("txt_path"):
        keep = group["n_in_deck"].idxmax()
        df.loc[group.index.difference([keep]), "is_primary"] = False

    hints = load_forge_hints()
    if hints is not None:
        hints = hints.rename(columns={"name": "_hint_name"})
        hints["_key"] = hints["_hint_name"].map(_norm_name)
        hints = hints.drop_duplicates("_key").drop(columns=["_hint_name"])
        # forge_hints.csv keys on Forge's single-face names, so a "A // B"
        # label row only matches through its front face.
        keys = set(hints["_key"])
        df["_key"] = [_hint_key(name, keys) for name in df["name"]]
        df = df.merge(hints, on="_key", how="left").drop(columns=["_key"])
        df["has_forge_hint"] = df["ai_remove_deck"].notna()
    else:
        df["has_forge_hint"] = False

    if with_printings and ALLPRINTINGS.exists():
        meta = load_printing_metadata()
        for col in ("set_code", "year", "rarity"):
            df[f"first_{col}"] = [
                (meta.get(_norm_name(n)) or {}).get(col) for n in df["name"]
            ]
    else:
        for col in ("set_code", "year", "rarity"):
            df[f"first_{col}"] = None

    df = df.sort_values("name").reset_index(drop=True)
    df.to_pickle(JOIN_TABLE)
    pd.DataFrame(unjoined).to_csv(SCRATCH / "unjoined.csv", index=False)
    return df


def read_text(path: Path | str) -> str:
    return ConvertedCardText.from_file(Path(path)).text


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


def label(frame: pd.DataFrame, head: str) -> Label:
    y = num(frame, f"shrunk_{head}")
    w = num(frame, f"w_{head}")
    return Label(y, w, np.isfinite(y) & (w > 0))


# ── embeddings ──────────────────────────────────────────────────────────


def load_embedding_matrix(
    names: Sequence[str], join: pd.DataFrame | None = None,
) -> np.ndarray:
    """``(len(names), 512)`` float32 text vectors from the ``.npz`` cache."""
    join = build_join() if join is None else join
    by_name = dict(zip(join["name"], join["npz_path"]))
    out = np.empty((len(names), TEXT_DIM), dtype=np.float32)
    for i, name in enumerate(names):
        path = by_name.get(name)
        if path is None:
            raise KeyError(f"no cached embedding for card: {name!r}")
        with np.load(path) as f:
            out[i] = f["embedding"][:TEXT_DIM]
    return out


CORPUS_MATRIX = SCRATCH / "corpus_embeddings.npz"


def corpus_embedding_matrix(
    *, force: bool = False, cards_path: Path = CARDS_PATH,
) -> tuple[list[str], np.ndarray]:
    """Every cached ``.npz`` in the corpus → ``(keys, (N, 512))``.

    A key is the extension-less path relative to the cards folder, e.g.
    ``s/serra_angel`` — join it against a row's ``txt_path`` with
    :func:`corpus_key`. This is the reference cloud for
    :func:`manifold_distance`: all 32,606 converted cards, not just the
    ~28k carrying winnability labels.
    """
    if CORPUS_MATRIX.exists() and not force:
        with np.load(CORPUS_MATRIX, allow_pickle=False) as f:
            return list(f["names"]), f["matrix"]
    paths = sorted(cards_path.rglob("*.npz"))
    matrix = np.empty((len(paths), TEXT_DIM), dtype=np.float32)
    names: list[str] = []
    for i, path in enumerate(paths):
        with np.load(path) as f:
            matrix[i] = f["embedding"][:TEXT_DIM]
        names.append(corpus_key(path, cards_path))
    np.savez_compressed(CORPUS_MATRIX, names=np.array(names), matrix=matrix)
    return names, matrix


def corpus_key(path: Path | str, cards_path: Path = CARDS_PATH) -> str:
    """Extension-less corpus-relative key of a card file (``s/serra_angel``)."""
    rel = Path(path).resolve().relative_to(cards_path.resolve())
    return rel.with_suffix("").as_posix()


# ── encoder ─────────────────────────────────────────────────────────────


class EncoderRunner:
    """The gen-4 sealed encoder, loaded once, with batched re-encoding.

    ``encode_texts`` takes *name-stripped* converted card text (a leading
    ``name:`` line is stripped defensively) and returns the 512-dim pooled
    text vector — the first 512 dims of the cached ``.npz``.

    The cache was written by ``encode-cards``, which runs the encoder on
    **CPU, one card at a time, padded to ``max_seq_len``**. That is the
    only configuration that reproduces it bit-exactly: batching or moving
    to CUDA changes the reduction order and shifts the last few bits
    (max |Δ| ≈ 3e-7, ~5e-6 σ of a text dim — irrelevant for probes, fatal
    for an equality check). ``exact=True`` opts into the slow faithful
    path; the default follows ``device``.
    """

    def __init__(
        self, device: str | None = None, checkpoint: Path = ENCODER_CKPT,
        vocab_path: Path = VOCAB_PATH,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.config = SealedEncoderStore().load_encoder(checkpoint)
        self.model.eval().to(self.device)
        self.tokenizer = load_tokenizer(vocab_path)
        self.max_seq_len = self.config.max_seq_len

    def _to(self, device: str) -> None:
        if device != self.device:
            self.model.to(device)
            self.device = device

    @torch.no_grad()
    def encode_texts(
        self, texts: Sequence[str], *, batch_size: int = 64, exact: bool = False,
    ) -> np.ndarray:
        """``(len(texts), 512)`` float32 pooled text vectors."""
        device, batch_size = (("cpu", 1) if exact else (self.device, batch_size))
        self._to(device)
        stripped = [ConvertedCardText(t).without_name_line() for t in texts]
        out = np.empty((len(texts), TEXT_DIM), dtype=np.float32)
        for lo in range(0, len(stripped), batch_size):
            chunk = stripped[lo:lo + batch_size]
            encoded = [self.tokenizer.encode(t, self.max_seq_len) for t in chunk]
            ids = torch.tensor([e[0] for e in encoded], dtype=torch.long, device=device)
            mask = torch.tensor([e[1] for e in encoded], dtype=torch.long, device=device)
            out[lo:lo + len(chunk)] = (
                self.model.encode(ids, mask).cpu().numpy().astype(np.float32)
            )
        return out

    def encode_files(
        self, paths: Sequence[Path], *, batch_size: int = 64, exact: bool = False,
    ) -> np.ndarray:
        return self.encode_texts([read_text(p) for p in paths],
                                 batch_size=batch_size, exact=exact)

    def check_bit_exact(self, paths: Sequence[Path]) -> dict:
        """Re-encode ``paths`` in exact mode and diff against their ``.npz``.

        Returns ``{n, n_exact, max_abs_diff}``; ``n_exact == n`` is the
        contract that lets a counterfactual read-off be attributed to the
        edit rather than to the harness.
        """
        got = self.encode_files(paths, exact=True)
        cached = np.stack([
            np.load(Path(p).with_suffix(".npz"))["embedding"][:TEXT_DIM]
            for p in paths
        ])
        return {
            "n": len(paths),
            "n_exact": int((got == cached).all(axis=1).sum()),
            "max_abs_diff": float(np.abs(got - cached).max()),
        }


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
    alpha, cv_r2 = _choose_alpha(X, y, w, folds=5)
    coef, b = _ridge_solve(X, y, w, [alpha])[alpha]
    return alpha, cv_r2, coef, b


def ridge_fit_eval(X_tr, y_tr, X_te, y_te, w_tr=None):
    """Ridge fitted on the train part; returns ``(r2_te, alpha, coef, b)``."""
    alpha, _, coef, b = ridge_fit(X_tr, y_tr, w_tr)
    return _r2(y_te, X_te @ coef + b), alpha, coef, b


def honest_ridge(
    X: np.ndarray, y: np.ndarray, is_train: np.ndarray, is_val: np.ndarray,
    keep: np.ndarray | None = None,
) -> dict:
    """Unweighted ridge fitted on the encoder's train split, scored on its val split.

    Comparable to the p0 honest/unweighted table; ``val_r2`` is the number
    to report.
    """
    keep = np.ones(len(y), dtype=bool) if keep is None else keep
    good = keep & np.isfinite(y)
    tr = good & is_train
    va = good & is_val
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


# ── tokenization / equivalence classes ──────────────────────────────────


def token_key(text: str, tokenizer=None) -> tuple[int, ...]:
    """Token-id tuple of name-stripped card text — the encoder's real input.

    Two cards with the same key are *indistinguishable to the encoder*:
    same ids, same padding, therefore the same embedding. Grouping on it
    gives the functional-reprint equivalence classes.
    """
    tokenizer = tokenizer or load_tokenizer(VOCAB_PATH)
    return tuple(tokenizer.tokenize_to_ids(ConvertedCardText(text).without_name_line()))


def equivalence_classes(
    join: pd.DataFrame, *, min_size: int = 2, primary_only: bool = True,
) -> list[dict]:
    """Group joined cards by identical name-stripped token sequence.

    Returns one dict per class of at least ``min_size`` members, with the
    member names, their ``shrunk_score_play`` / ``n_in_deck``, and their
    split assignment. Sorted by size descending.
    """
    tokenizer = load_tokenizer(VOCAB_PATH)
    frame = join[join["is_primary"]] if primary_only else join
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for idx, path in zip(frame.index, frame["txt_path"]):
        groups[token_key(read_text(path), tokenizer)].append(idx)
    out = []
    for key, idxs in groups.items():
        if len(idxs) < min_size:
            continue
        sub = join.loc[idxs]
        out.append({
            "n_tokens": len(key),
            "size": len(idxs),
            "names": list(sub["name"]),
            "txt_paths": list(sub["txt_path"]),
            "splits": list(sub["split"]),
            "score_play": [
                None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
                for v in sub["shrunk_score_play"]
            ],
            "n_in_deck": [int(v) for v in sub["n_in_deck"]],
        })
    out.sort(key=lambda c: -c["size"])
    return out


def variance_decomposition(
    classes: Sequence[dict], join: pd.DataFrame,
    column: str = "shrunk_score_play", *, min_n: int = 0,
) -> dict:
    """Within- vs between-class variance for a label, over class members.

    The within-class variance is the part of the label no text-only model
    can ever explain — the members are the same text — so ``1 − within /
    total`` is an upper bound on text-explainable variance, and the
    within term is the irreducible-noise estimate. Reported both against
    the full corpus variance and against the variance of the class
    members only (the fairer comparison: reprints skew common/simple).

    ``min_n`` drops members below an ``n_in_deck`` floor before the
    decomposition. Raising it separates *irreducible* noise from
    *under-observation*: a card seen thirty times has a noisy label for
    reasons a bigger corpus would fix.
    """
    by_name = join.set_index("name")
    values: list[float] = []
    group_ids: list[int] = []
    for gid, cls in enumerate(classes):
        sub = by_name.loc[[n for n in cls["names"] if n in by_name.index]]
        sub = sub[sub["n_in_deck"] >= min_n]
        vals = pd.to_numeric(sub[column], errors="coerce").dropna().to_numpy(float)
        if len(vals) < 2:
            continue
        values.extend(vals)
        group_ids.extend([gid] * len(vals))
    values = np.asarray(values)
    group_ids = np.asarray(group_ids)
    if values.size == 0:
        return {}
    corpus_rows = join[join["n_in_deck"] >= min_n]
    corpus = pd.to_numeric(corpus_rows[column], errors="coerce").dropna().to_numpy(float)
    n_groups = len(np.unique(group_ids))
    ss_within = 0.0
    for gid in np.unique(group_ids):
        v = values[group_ids == gid]
        ss_within += float(((v - v.mean()) ** 2).sum())
    within_var = ss_within / max(1, len(values) - n_groups)
    member_var = float(values.var(ddof=1))
    corpus_var = float(corpus.var(ddof=1))
    return {
        "column": column,
        "min_n": min_n,
        "n_members": int(len(values)),
        "n_classes": int(n_groups),
        "within_class_var": within_var,
        "member_var": member_var,
        "corpus_var": corpus_var,
        "explainable_fraction_vs_members": 1.0 - within_var / member_var,
        "explainable_fraction_vs_corpus": 1.0 - within_var / corpus_var,
        "noise_sd": float(np.sqrt(within_var)),
    }


# ── placebo edits ───────────────────────────────────────────────────────

def subtype_frequencies(
    cards_path: Path = CARDS_PATH, *, cache: Path | None = None,
) -> Counter:
    """Corpus frequency of every creature subtype seen on a ``types:`` line."""
    cache = cache or (SCRATCH / "subtype_freq.pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    freq: Counter = Counter()
    for path in cards_path.rglob("*.txt"):
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("types:"):
                    continue
                words = line[6:].split()
                if "creature" not in words:
                    continue
                freq.update(subtypes(words))
    with open(cache, "wb") as f:
        pickle.dump(freq, f)
    return freq


def _same_frequency_subtype(subtype: str, freq: Counter, exclude: set[str]) -> str | None:
    """Nearest-frequency creature subtype that is not already on the card."""
    target = freq.get(subtype, 0)
    best, best_gap = None, None
    for other, count in freq.items():
        if other == subtype or other in exclude:
            continue
        gap = abs(count - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = other, gap
    return best


def placebo_edits(text: str, freq: Counter | None = None) -> dict[str, str | None]:
    """Meaning-preserving edits of one converted card's text.

    Three families, each ``None`` when the card cannot support it:

    * ``swap_static`` — exchange the first two ``static:`` lines. Keyword
      order on a card is not semantic, so the label is unchanged.
    * ``subtype_swap`` — replace one creature subtype with a corpus-
      frequency-matched other subtype. Tribal payoffs exist, but in a
      sealed pool a lone subtype is nearly inert.
    * ``swap_ability_lines`` — exchange two ability lines of different
      kinds (the strongest test of the bag-of-words null).

    Their probe-prediction shifts are the null distribution any real
    counterfactual edit has to clear.
    """
    freq = subtype_frequencies() if freq is None else freq
    lines = text.splitlines()
    out: dict[str, str | None] = {
        "swap_static": None, "subtype_swap": None, "swap_ability_lines": None,
    }

    statics = [i for i, ln in enumerate(lines) if ln.startswith("static:")]
    if len(statics) >= 2:
        edited = list(lines)
        a, b = statics[0], statics[1]
        edited[a], edited[b] = edited[b], edited[a]
        out["swap_static"] = "\n".join(edited)

    for i, line in enumerate(lines):
        if not line.startswith("types:"):
            continue
        words = line[6:].split()
        if "creature" not in words:
            continue
        subs = subtypes(words)
        if not subs:
            continue
        victim = subs[-1]
        replacement = _same_frequency_subtype(victim, freq, set(subs))
        if replacement is None:
            continue
        edited = list(lines)
        edited[i] = "types: " + " ".join(
            replacement if w == victim else w for w in words
        )
        out["subtype_swap"] = "\n".join(edited)
        break

    abilities = [i for i, ln in enumerate(lines) if ln.startswith(ABILITY_PREFIXES)]
    if len(abilities) >= 2:
        a, b = abilities[0], abilities[-1]
        edited = list(lines)
        edited[a], edited[b] = edited[b], edited[a]
        out["swap_ability_lines"] = "\n".join(edited)
    return out


# ── manifold gate ───────────────────────────────────────────────────────


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def manifold_distance(
    embeddings: np.ndarray,
    reference: np.ndarray | None = None,
    *,
    self_rows: np.ndarray | None = None,
    chunk: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance from each row to its nearest real-card embedding.

    ``reference`` defaults to the full 512-dim corpus cache. ``self_rows``
    gives, per input row, the reference index to exclude (use it when the
    inputs *are* real cards, so the nearest neighbour is not the card
    itself). Returns ``(distance, nearest_index)``.

    An edited card whose distance clears the real-card 95th percentile
    has left the manifold: whatever the probe reads off it is an
    extrapolation, not a counterfactual.
    """
    if reference is None:
        _, reference = corpus_embedding_matrix()
    ref = _normalize(np.asarray(reference, dtype=np.float32))
    query = _normalize(np.asarray(embeddings, dtype=np.float32))
    dist = np.empty(len(query), dtype=np.float32)
    idx = np.empty(len(query), dtype=np.int64)
    for lo in range(0, len(query), chunk):
        sims = query[lo:lo + chunk] @ ref.T
        if self_rows is not None:
            rows = np.arange(sims.shape[0])
            sims[rows, self_rows[lo:lo + chunk]] = -np.inf
        best = sims.argmax(axis=1)
        idx[lo:lo + chunk] = best
        dist[lo:lo + chunk] = 1.0 - sims[np.arange(sims.shape[0]), best]
    return dist, idx


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


def mean_ci(delta: np.ndarray, key: str = "mean_sd", n_boot: int = 4000) -> dict:
    """``{key: mean, ci_lo, ci_hi}`` of a paired per-card difference."""
    lo, hi = bootstrap_ci(delta, n_boot=n_boot, seed=7)
    return {key: float(np.mean(delta)), "ci_lo": lo, "ci_hi": hi}


def contrast_row(delta: np.ndarray, pr_delta: np.ndarray | None = None,
                 off: np.ndarray | None = None, *, key: str = "mean_sd",
                 pr_key: str = "delta_pr", n_boot: int = 4000, **lead) -> dict:
    """One paired arm-minus-arm result row over the base cards.

    ``lead`` names the contrast; ``pr_delta`` and ``off`` (the edited arm's
    off-manifold mask) add the played_rate shift and the off-manifold share.
    """
    row = {**lead, "n": len(delta), **mean_ci(delta, key, n_boot),
           "frac_pos": float((delta > 0).mean())}
    if pr_delta is not None:
        row[pr_key] = float(pr_delta.mean())
    if off is not None:
        row["off_manifold"] = float(np.mean(off))
    return row


def score_arms(bases: Sequence[int], arms: Sequence, edit) -> Scored:
    """Every base card under every arm, read in one batch: ``edit(base, arm)``
    gives a text, and the result is shaped ``(len(bases), len(arms))``."""
    return score_texts([edit(b, a) for b in bases for a in arms]).reshape(
        len(bases), len(arms))


def contrasts(S: Scored, labels: Sequence, *, cols: Sequence[int] | None = None,
              ref: int = 0, field: str = "contrast", **options) -> list[dict]:
    """One :func:`contrast_row` per arm column against the ``ref`` column.

    ``cols`` defaults to the columns after the first, one per label; each row
    names its arm under ``field``, after any extra lead columns in ``options``.
    """
    cols = range(1, len(labels) + 1) if cols is None else cols
    return [contrast_row(S.sp[:, c] - S.sp[:, ref], S.pr[:, c] - S.pr[:, ref],
                         S.off[:, c], **options, **{field: label})
            for c, label in zip(cols, labels)]


def write_table(tab: pd.DataFrame | Sequence[dict], name: str) -> None:
    """Write a result table to ``SCRATCH/name`` and print it."""
    tab = pd.DataFrame(tab)
    tab.to_csv(SCRATCH / name, index=False)
    print(tab.to_string(index=False), flush=True)


def write_summary(summary: dict, name: str, *, indent: int = 2) -> None:
    """Write a summary JSON to ``SCRATCH/name`` and print it."""
    write_json(SCRATCH / name, summary, indent=indent)
    print(json.dumps(plain(summary), indent=indent), flush=True)
