"""The closed lists the head's subtype and keyword outputs are drawn over.

Root spec § 11.2 and § 11.3 size ``subtypes_gained``/``subtypes_lost``, the
created-objects groups' subtypes and ``keywords_gained``/``keywords_lost`` by
what the corpus's effects actually grant or remove, a closed list the corpus
build records (FR-052, FR-053), rather than by every subtype and keyword Forge
knows: an output column no record ever sets trains nothing and only widens the
head. This module reads those names off one record; the head reads the lists
checked in as :mod:`effects.domain.output_vocabularies`, which
``scripts/scan_output_vocabularies.py`` writes from a records survey.
"""

from __future__ import annotations

from collections.abc import Iterable

from effects.domain.effect_head_input import CARD_TYPES, OVERLAY_KEYWORDS, SUPERTYPES
from effects.domain.event_schema import EventType
from effects.domain.forge_vocabularies import SUBTYPES
from effects.domain.keyword_names import normalize_keyword, overlay_keyword
from effects.domain.records import ContinuousPayload, EffectRecord, events_of

_SUBTYPE_ORDER: dict[str, int] = {name: index for index, name in enumerate(SUBTYPES)}
_NOT_SUBTYPES = frozenset(CARD_TYPES) | frozenset(SUPERTYPES)


def type_tokens(type_line: str) -> tuple[str, ...]:
    """A created object's type line (``Legendary Creature - Elf Warrior``) as
    lower-case tokens, the dash dropped."""
    return tuple(
        token.lower() for token in type_line.replace("—", "-").split() if token != "-"
    )


def subtypes_of(tokens: Iterable[str]) -> tuple[str, ...]:
    """The tokens that are Forge subtypes, in the order given; a card type,
    a supertype or anything Forge does not list (``all-creature-types``) is
    none."""
    found = []
    for token in tokens:
        name = str(token).strip().lower()
        if name in _SUBTYPE_ORDER and name not in _NOT_SUBTYPES:
            found.append(name)
    return tuple(found)


def subtypes_named(record: EffectRecord) -> set[str]:
    """Every subtype one record's effects grant or remove, or that a token it
    created carries: what ``subtypes_gained``/``subtypes_lost`` and the
    created-objects groups' subtypes can set."""
    names: set[str] = set()
    for event in events_of(record):
        if event.type is EventType.TYPE_CHANGE:
            for key in ("types_added", "types_removed"):
                names.update(subtypes_of(event.params.get(key) or ()))
        elif event.type is EventType.TOKEN_CREATED:
            traits = event.params.get("characteristics") or {}
            names.update(subtypes_of(type_tokens(str(traits.get("types", "")))))
    if isinstance(record.payload, ContinuousPayload):
        for contribution in record.payload.contributions:
            change = contribution.type_changes
            names.update(subtypes_of((*change.gained, *change.lost)))
    return names


def keywords_named(record: EffectRecord, known: frozenset[str]) -> set[str]:
    """Every keyword one record's effects grant or remove, or that a token it
    created carries, in the head's spelling (:func:`overlay_keyword`).

    ``known`` holds Forge's keywords in :func:`normalize_keyword`'s spelling;
    a granted static's text, which Forge lists beside the keywords, is none.
    """
    raw: list[str] = []
    for event in events_of(record):
        if event.type is EventType.KEYWORD_CHANGE:
            raw.extend(str(keyword) for keyword in event.params.get("keywords") or ())
        elif event.type is EventType.TOKEN_CREATED:
            traits = event.params.get("characteristics") or {}
            keywords = traits.get("keywords") or ()
            raw.extend([keywords] if isinstance(keywords, str) else map(str, keywords))
    if isinstance(record.payload, ContinuousPayload):
        for contribution in record.payload.contributions:
            change = contribution.keyword_changes
            raw.extend((*change.gained, *change.lost))
    names = set()
    for keyword in raw:
        name = overlay_keyword(keyword)
        if name in OVERLAY_KEYWORDS or normalize_keyword(keyword) in known:
            names.add(name)
    return names


def ordered_subtypes(names: Iterable[str]) -> tuple[str, ...]:
    """Subtypes in Forge's own order (``TypeLists.txt``)."""
    return tuple(sorted(set(names) & _SUBTYPE_ORDER.keys(), key=_SUBTYPE_ORDER.__getitem__))


def ordered_keywords(names: Iterable[str]) -> tuple[str, ...]:
    """The overlay keywords first, in their order, which gate 2 and the
    probes read; then every other keyword alphabetically."""
    rest = sorted(set(names) - set(OVERLAY_KEYWORDS))
    return (*OVERLAY_KEYWORDS, *rest)
