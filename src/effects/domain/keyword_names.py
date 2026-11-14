"""A keyword's name, in the one spelling this package reads.

Forge writes what the card prints (``"first strike"``, ``"ward:2"``,
``"protection from blue"``); the overlay vocabulary, gate 2's table and
``--probe-keywords`` spell them ``first_strike``, ``ward``, ``protection``.
Every reader converts through here, because a copy that forgets the value or
a protection's or hexproof's quality, or trample's ``Planeswalker``, matches
nothing and says so nowhere. Import-free, so every reader can share it.
"""

from __future__ import annotations

PROTECTION_FROM = "protection_from_"


def display_name_of(line_text: str) -> str:
    """A keyword line's display name: its text before the first colon (FR-013)."""
    return line_text.split(":", 1)[0].strip()


#: The keywords whose part after Forge's ``:`` is a quality, what the keyword
#: protects from, rather than a value: hexproof from black is no hexproof.
QUALITY_KEYWORDS: frozenset[str] = frozenset({"protection", "hexproof"})


#: Forge's spellings whose part after ``:`` makes a keyword of its own rather
#: than a value: ``Trample:Planeswalker`` is trample over planeswalkers
#: (702.19c), the name its printed form reads too.
DETAILED_KEYWORDS: dict[str, str] = {"trample:planeswalker": "trample_over_planeswalkers"}

#: What a keyword also is: a creature with trample over planeswalkers has
#: trample (702.19c), as Forge's ``hasKeyword(Keyword.TRAMPLE)`` answers for it.
IMPLIED_KEYWORDS: dict[str, str] = {"trample_over_planeswalkers": "trample"}


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
    (a third part), else its second part less a selector's type prefix
    (``Creature.Goblin`` reads ``goblin``). A detail that makes a keyword of
    its own (:data:`DETAILED_KEYWORDS`) is kept the same way:
    ``Trample:Planeswalker`` reads ``trample_over_planeswalkers``.
    """
    name, colon, rest = raw.partition(":")
    name = _spelled(name)
    if colon:
        detailed = DETAILED_KEYWORDS.get(f"{name}:{_spelled(rest)}")
        if detailed is not None:
            return detailed
    if colon and name in QUALITY_KEYWORDS:
        parts = rest.split(":")
        if len(parts) > 1 and parts[1].strip():
            quality = _spelled(parts[1])
        else:
            quality = _spelled(parts[0].rsplit(".", 1)[-1])
        if quality:
            return f"{name}_from_{quality}"
    return name


def overlay_keyword(raw: str) -> str:
    """:func:`normalize_keyword`, with every protection folded onto the
    overlay's one ``protection`` slot and a keyword another implies
    (:data:`IMPLIED_KEYWORDS`) onto that one's: trample over planeswalkers is
    trample's slot."""
    name = normalize_keyword(raw)
    if name.startswith(PROTECTION_FROM):
        return "protection"
    return IMPLIED_KEYWORDS.get(name, name)


def with_implied(names: set[str]) -> set[str]:
    """``names`` (normalized) with every keyword one of them implies added
    (:data:`IMPLIED_KEYWORDS`): a reader asking whether a creature has trample
    finds it on one with trample over planeswalkers alone."""
    return names | {IMPLIED_KEYWORDS[name] for name in names if name in IMPLIED_KEYWORDS}


HEXPROOF_FROM = "hexproof_from_"


def changed_keyword(raw: str) -> str:
    """:func:`overlay_keyword` as the keywords-gained and -lost outputs read
    it: a hexproof from a quality also folds onto ``hexproof``, so gaining or
    losing one is a hexproof change, while the input's flag for full hexproof
    stays off for it."""
    name = overlay_keyword(raw)
    return "hexproof" if name.startswith(HEXPROOF_FROM) else name
