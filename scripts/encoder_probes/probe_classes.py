"""Functional-reprint equivalence classes and the label variance they bound."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from probe_encoder import read_text
from probe_paths import VOCAB_PATH

from price_predictor.domain.card_text import ConvertedCardText
from price_predictor.infrastructure.tokenizer_store import load_tokenizer


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

