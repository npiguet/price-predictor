"""Choosing the held-out ability texts, and the cards that carry them.

The holdout is keyed on the ability text rather than the card. The model never
reads a card's name — an entity reaches it as computed characteristics plus its
ability lines' embeddings — so the card was always a proxy for the text on it,
and the proxy splits functional reprints: Searing Spear and Lightning Strike
compile to the same script, and a name-keyed holdout would train on one while
holding out the other.

Membership depends on the text's own bytes alone, so adding cards never
reassigns an existing text. That is what lets a depleted collection run freeze
the holdout into the games it collects (FR-130): a selection rule that moved
when Forge gained a set would invalidate the corpus it was collected against.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from zlib import crc32

#: Runs of whitespace differ between scripts that are otherwise the same text,
#: and the difference is formatting rather than mechanism.
_WHITESPACE = re.compile(r"\s+")


def normalize_script_text(text: str) -> str:
    """The form the hash is taken over."""
    return _WHITESPACE.sub(" ", text).strip()


def text_is_held_out(text: str, *, permille: int) -> bool:
    """Whether this text's own bytes put it in the holdout.

    ``crc32`` rather than the built-in ``hash``, which is salted per process for
    strings: a holdout that changed between runs would not match the split the
    corpus was depleted against (FR-088a).
    """
    return crc32(normalize_script_text(text).encode("utf-8")) % 1000 < permille


@dataclass(frozen=True, slots=True)
class HoldoutSelection:
    """The held-out texts and every card carrying one."""

    texts: frozenset[str]
    cards: frozenset[str]


def select_holdout(
    texts_by_card: Mapping[str, Iterable[str]],
    *,
    permille: int,
    max_carriers: int,
) -> HoldoutSelection:
    """Pick the held-out texts, and the cards depletion has to remove."""
    normalized: dict[str, set[str]] = {}
    for card, texts in texts_by_card.items():
        normalized[card] = {normalize_script_text(t) for t in texts if t}

    carriers: Counter[str] = Counter()
    for texts in normalized.values():
        carriers.update(texts)

    held = {
        text
        for text, count in carriers.items()
        if count <= max_carriers and text_is_held_out(text, permille=permille)
    }

    cards = {
        card for card, texts in normalized.items() if texts & held
    }
    return HoldoutSelection(texts=frozenset(held), cards=frozenset(cards))
