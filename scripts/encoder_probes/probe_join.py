"""The label-card join: names → converted files, the encoder's split, the join table.

Resolves each ``cards-win-rates.txt`` row to its converted ``.txt`` through
the corpus's own ``name:`` lines, rebuilds the encoder's seed-42 train/val
partition, adds the Forge-side hints and the earliest printing, and caches
the one-row-per-label-card table every probe starts from.
"""

from __future__ import annotations

import json
import pickle
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from probe_paths import ALLPRINTINGS, CARDS_PATH, FORGE_HINTS, SCRATCH, WIN_RATES

from price_predictor.application.ridge_probes import HEADS, head_effective_n, head_weight
from price_predictor.application.ridge_probes import load_labels as _load_labels

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
    """Rebuild ``encoder_split._split_cards`` from the label snapshot.

    The snapshot ``cards-win-rates.txt`` *is* the label map the trainer
    split (it is written immediately before the split, after the
    missing-text drop), and the split only reads the shrunk signed cells,
    so importing the real function on reconstructed ``CardLabels`` gives
    the run's own partition. ``p0_build`` asserts the sizes against the
    training log.
    """
    from sealed.application.encoder_counters import CardCounters
    from sealed.application.encoder_labels import CardLabels
    from sealed.application.encoder_split import _split_cards

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

