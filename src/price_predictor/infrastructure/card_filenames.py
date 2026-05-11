"""Shared helper for converting card names to on-disk filenames."""

from __future__ import annotations

import re
import unicodedata

_MULTI_UNDERSCORE = re.compile(r"_+")


def sanitize_card_name(
    name: str,
    corrections: dict[str, str] | None = None,
) -> str:
    """Convert a card name to its on-disk filename stem.

    NFKD-decomposes accents (a -> a), lowercases, strips punctuation,
    collapses runs of ``_`` into a single ``_`` (so ``"Anchovy & Banana
    Pizza"`` -> ``anchovy_banana_pizza`` rather than the double-underscore
    ``anchovy__banana_pizza``, matching Forge's own filename convention),
    and optionally applies known filename corrections.
    """
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    sanitized = (
        ascii_name.lower()
        .replace(" // ", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace(",", "")
        .replace(":", "")
        .replace("!", "")
        .replace('"', "")
        .replace("&", "")
        .replace("+", "")
        .replace(".", "")
        .replace("-", "_")
        .replace("/", "")
    )
    sanitized = _MULTI_UNDERSCORE.sub("_", sanitized).strip("_")
    if corrections:
        return corrections.get(sanitized, sanitized)
    return sanitized
