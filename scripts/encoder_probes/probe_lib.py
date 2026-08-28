"""Shared harness for gen-4 sealed card-encoder interpretability probes.

Loads the production sealed encoder once, joins the per-card winnability
labels to the converted card text and its cached ``.npz`` embedding,
reconstructs the encoder's own seed-42 train/val split, and fits ridge
probes from the 512-dim text vector to each of the nine shrunk labels.

Also provides the counterfactual scaffolding the later probes need:
re-encoding of edited card text (bit-exact against the cache in ``exact``
mode), a placebo-edit family that changes text without changing meaning,
and a nearest-neighbour manifold gate over the full embedding cache.

All probe scripts import from this module. Read-only with respect to the
repo and the Y: drive; generated artifacts go to ``output/encoder-probes/``.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
ENCODER_CKPT = REPO / "models/sealed/encoder/full-20260517-014759-attn-6l-8h-8q-0.1mlm-512d.pt"
VOCAB_PATH = REPO / "models/sealed/encoder/vocab.txt"
CARDS_PATH = REPO / "output/cardsfolder-512"
ALLPRINTINGS = REPO / "resources/AllPrintings.json"
FORGE_HINTS = REPO / "output/scorer-probes/forge_hints.csv"
YDATA = Path(r"Y:\Nicolas\mtg\mtg-models-data\sealed\training-data")
WIN_RATES = YDATA / "matches-bo1" / "cards-win-rates.txt"
SCRATCH = REPO / "output" / "encoder-probes"  # generated artifacts (gitignored)
SCRATCH.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "src"))

from price_predictor.domain.card_text import ConvertedCardText  # noqa: E402
from price_predictor.infrastructure.tokenizer_store import load_tokenizer  # noqa: E402
from sealed.domain.card_encoder import CardEncoder  # noqa: E402
from sealed.infrastructure.encoder_store import SealedEncoderStore  # noqa: E402

TEXT_DIM = 512          # pooled text vector; the cached .npz is 512 + 32 features
SHRINKAGE_K = 20.0      # train-encoder --shrinkage-k default, used for the labels

# Head order matches ``train_encoder._ALL_HEAD_NAMES`` / the label file columns.
HEADS: tuple[str, ...] = (
    "score_play", "score_draw", "played_rate", "cast_lift",
    "color_lift_W", "color_lift_U", "color_lift_B", "color_lift_R",
    "color_lift_G",
)
COUNTER_COLUMNS: tuple[str, ...] = (
    "wins_when_played", "wins_when_in_deck",
    "losses_when_played", "losses_when_in_deck",
)


# ── labels ──────────────────────────────────────────────────────────────


def load_labels(path: Path = WIN_RATES) -> dict[str, dict]:
    """``cards-win-rates.txt`` → ``name -> {column: value}``.

    The four counters come back as ``int``; the eighteen raw/shrunk label
    cells as ``float`` or ``None`` (an empty cell means "no signal", not
    "neutral signal" — see the file-format contract in CLAUDE.md). The
    file is UTF-8; reading it as cp1252 crashes on accented card names.
    """
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(";")
        for line in f:
            parts = line.rstrip("\n").split(";")
            if len(parts) != len(header):
                continue
            rec: dict = {}
            for key, value in zip(header[1:], parts[1:]):
                if key in COUNTER_COLUMNS:
                    rec[key] = int(value)
                else:
                    rec[key] = float(value) if value != "" else None
            rows[parts[0]] = rec
    return rows


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

_TYPE_WORDS = frozenset({
    "artifact", "battle", "conspiracy", "creature", "dungeon", "emblem",
    "enchantment", "instant", "kindred", "land", "phenomenon", "plane",
    "planeswalker", "scheme", "sorcery", "tribal", "vanguard",
})
_SUPERTYPE_WORDS = frozenset({
    "basic", "elite", "host", "legendary", "ongoing", "snow", "world",
})


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


# ── per-head observation counts ─────────────────────────────────────────


def _head_n(rec: dict, head: str) -> float | None:
    """Effective observation count behind one head's cell, or None.

    ``played_rate`` and ``cast_lift`` read straight off the four counters.
    The remaining seven heads use denominators (@play / @draw / per-color
    slices) that ``cards-win-rates.txt`` does not carry, but every one of
    them is recoverable from the raw/shrunk pair: both share a numerator,
    so ``shrunk / raw == n / (n + k)`` and ``n = k · shrunk / (raw −
    shrunk)``. That identity degrades when the numerator is near zero
    (the cells are rounded to five decimals), so it is only used when
    ``|raw|`` is comfortably above the rounding floor; otherwise the fall
    back is ``n_in_deck / 2`` for the two @play/@draw heads (each game
    puts exactly one side on the play) and ``n_in_deck`` for a color lift.
    """
    in_deck = rec["wins_when_in_deck"] + rec["losses_when_in_deck"]
    if head == "played_rate":
        return float(in_deck)
    if head == "cast_lift":
        played = rec["wins_when_played"] + rec["losses_when_played"]
        return float(min(played, in_deck - played))

    if head in ("score_play", "score_draw"):
        raw, shrunk = rec[f"raw_{head}"], rec[f"shrunk_{head}"]
        fallback = in_deck / 2.0
    else:
        color = head[-1]
        raw, shrunk = rec[f"raw_color_lift_{color}"], rec[f"shrunk_color_lift_{color}"]
        # color_lift subtracts the card's overall score, so undo that first.
        overall_num = rec["wins_when_played"] - rec["losses_when_played"]
        if in_deck == 0:
            return None
        if raw is not None:
            raw = raw + overall_num / in_deck
        if shrunk is not None:
            shrunk = shrunk + overall_num / (in_deck + SHRINKAGE_K)
        fallback = float(in_deck)
    if raw is None or shrunk is None:
        return None
    if abs(raw) < 5e-4 or raw == shrunk:
        return fallback
    ratio = shrunk / raw
    if not (0.0 < ratio < 1.0):
        return fallback
    return SHRINKAGE_K * ratio / (1.0 - ratio)


def head_weight(n: float | None, k: float = SHRINKAGE_K) -> float:
    """FR-017a per-head sample weight ``n / (n + k)``; 0.0 for a dead cell."""
    if n is None or n <= 0:
        return 0.0
    return float(n / (n + k))


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
    from sealed.application.train_encoder import (  # local: heavy import
        CardCounters, CardLabels, _split_cards,
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
            n = _head_n(rec, head)
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
        self.card_encoder = CardEncoder(
            self.model, self.tokenizer, self.max_seq_len, self.device,
        )

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
        stripped = [
            "\n".join(l for l in t.splitlines() if not l.startswith("name:"))
            for t in texts
        ]
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
        texts = [ConvertedCardText.from_file(Path(p)).text for p in paths]
        return self.encode_texts(texts, batch_size=batch_size, exact=exact)

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


# ── ridge probes ────────────────────────────────────────────────────────

ALPHA_GRID: tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
LOGIT_CLIP = 1e-3


def to_logit(p: np.ndarray, clip: float = LOGIT_CLIP) -> np.ndarray:
    q = np.clip(p, clip, 1.0 - clip)
    return np.log(q / (1.0 - q))


@dataclass
class HeadProbe:
    """One fitted ridge probe: 512 weights, an intercept, and its metrics."""

    head: str
    space: str            # "linear" or "logit" (played_rate only)
    coef: np.ndarray      # (512,)
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
    unpenalised, forms the 512×512 weighted Gram matrix once, and reuses
    its eigendecomposition across the alpha grid.
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
) -> tuple[float, float]:
    """K-fold CV over :data:`ALPHA_GRID`; returns ``(alpha, cv_r2)``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    parts = np.array_split(order, folds)
    scores = {a: 0.0 for a in ALPHA_GRID}
    for part in parts:
        mask = np.ones(len(y), dtype=bool)
        mask[part] = False
        fits = _ridge_solve(X[mask], y[mask], w[mask], ALPHA_GRID)
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
) -> ProbeSet:
    """Ridge probes from the 512-dim text vector to each shrunk label.

    ``mode='fidelity'`` fits on every joined card — the read-off model for
    counterfactual edits, where generalization is not the claim.
    ``mode='honest'`` fits only on the encoder's own train split, so its
    metrics on the val split are an unrecycled generalization number.

    ``weighted`` applies the training objective's ``n/(n+20)`` per-head
    sample weight. ``played_rate`` is fitted twice — linearly and in logit
    space (``played_rate@logit``), since a rate bounded in [0, 1] with mass
    near both ends is not a linear target.

    ``embeddings`` must be row-aligned with ``join``.
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
            alpha, cv_r2 = _choose_alpha(X, y, w, folds=folds)
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


# ── tokenization / equivalence classes ──────────────────────────────────


def token_key(text: str, tokenizer=None) -> tuple[int, ...]:
    """Token-id tuple of name-stripped card text — the encoder's real input.

    Two cards with the same key are *indistinguishable to the encoder*:
    same ids, same padding, therefore the same embedding. Grouping on it
    gives the functional-reprint equivalence classes.
    """
    tokenizer = tokenizer or load_tokenizer(VOCAB_PATH)
    stripped = "\n".join(
        l for l in text.splitlines() if not l.startswith("name:")
    )
    return tuple(tokenizer.tokenize_to_ids(stripped))


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
        groups[token_key(Path(path).read_text(encoding="utf-8", errors="replace"),
                         tokenizer)].append(idx)
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

_ABILITY_PREFIXES = ("static:", "spell[", "activated[", "triggered", "replacement")


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
                freq.update(
                    w for w in words
                    if w not in _TYPE_WORDS and w not in _SUPERTYPE_WORDS
                )
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

    statics = [i for i, l in enumerate(lines) if l.startswith("static:")]
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
        subtypes = [w for w in words
                    if w not in _TYPE_WORDS and w not in _SUPERTYPE_WORDS]
        if not subtypes:
            continue
        victim = subtypes[-1]
        replacement = _same_frequency_subtype(victim, freq, set(subtypes))
        if replacement is None:
            continue
        edited = list(lines)
        edited[i] = "types: " + " ".join(
            replacement if w == victim else w for w in words
        )
        out["subtype_swap"] = "\n".join(edited)
        break

    abilities = [i for i, l in enumerate(lines) if l.startswith(_ABILITY_PREFIXES)]
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
