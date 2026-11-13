"""A keyword's name, in the one spelling this package reads.

Forge writes what the card prints (``"first strike"``, ``"ward:2"``,
``"protection from blue"``); the overlay vocabulary, gate 2's table and
``--probe-keywords`` spell them ``first_strike``, ``ward``, ``protection``.
Every reader converts through here, because a copy that forgets the value or
a protection's or hexproof's quality matches nothing and says so nowhere. Import-free, so
every reader can share it.
"""

from __future__ import annotations

PROTECTION_FROM = "protection_from_"


def display_name_of(line_text: str) -> str:
    """A keyword line's display name: its text before the first colon (FR-013)."""
    return line_text.split(":", 1)[0].strip()


#: The keywords whose part after Forge's ``:`` is a quality, what the keyword
#: protects from, rather than a value: hexproof from black is no hexproof.
QUALITY_KEYWORDS: frozenset[str] = frozenset({"protection", "hexproof"})


def _spelled(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def normalize_keyword(raw: str) -> str:
    """Forge's spelling of a keyword, in the one this package reads.

    A value after ``:`` is dropped: it reaches the model through the ability
    line, not through the overlay bit. A quality is kept, since it decides
    which keyword it is: Forge's ``Protection:Goblin`` and
    ``Hexproof:Card.Black:black`` read ``protection_from_goblin`` and
    ``hexproof_from_black``, as the printed ``Protection from blue`` reads
    ``protection_from_blue``; the quality is the text Forge describes it by
    (a third part), else its second part.
    """
    name, colon, rest = raw.partition(":")
    name = _spelled(name)
    if colon and name in QUALITY_KEYWORDS:
        parts = rest.split(":")
        quality = _spelled(parts[1] if len(parts) > 1 and parts[1].strip() else parts[0])
        if quality:
            return f"{name}_from_{quality.removeprefix('card.')}"
    return name


def overlay_keyword(raw: str) -> str:
    """:func:`normalize_keyword`, with every protection folded onto the
    overlay's one ``protection`` slot."""
    name = normalize_keyword(raw)
    return "protection" if name.startswith(PROTECTION_FROM) else name
