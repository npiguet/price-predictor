"""Meaning-preserving placebo edits of converted card text, the null any real edit clears."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

from c_common import ABILITY_PREFIXES, subtypes
from probe_paths import CARDS_PATH, SCRATCH


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

