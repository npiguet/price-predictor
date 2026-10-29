"""A keyword's name, in the one spelling this package reads.

Forge writes what the card prints (``"first strike"``, ``"ward:2"``,
``"protection from blue"``); the overlay vocabulary, gate 2's table and
``--probe-keywords`` spell them ``first_strike``, ``ward``, ``protection``.
Every reader converts through here, because a copy that forgets the value or
the protection quality matches nothing and says so nowhere. Import-free, so
every reader can share it.
"""

from __future__ import annotations

PROTECTION_FROM = "protection_from_"


def normalize_keyword(raw: str) -> str:
    """Forge's spelling of a keyword, in the one this package reads.

    The value after ``:`` is dropped: it reaches the model through the ability
    line, not through the overlay bit.
    """
    return raw.split(":", 1)[0].strip().lower().replace(" ", "_")


def overlay_keyword(raw: str) -> str:
    """:func:`normalize_keyword`, with every protection folded onto the
    overlay's one ``protection`` slot."""
    name = normalize_keyword(raw)
    return "protection" if name.startswith(PROTECTION_FROM) else name
