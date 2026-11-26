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

from effects.domain.effect_head_layout import OVERLAY_KEYWORDS
from effects.domain.event_schema import Event, EventType
from effects.domain.forge_vocabularies import SUBTYPES
from effects.domain.input_vocabularies import CARD_TYPES, SUPERTYPES
from effects.domain.keyword_names import changed_keyword, normalize_keyword
from effects.domain.records import ContinuousPayload, Contribution, EffectRecord, events_of

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
        names.update(_event_subtypes(event))
    for contribution in _contributions(record):
        change = contribution.type_changes
        names.update(subtypes_of((*change.gained, *change.lost)))
    return names


def keywords_named(record: EffectRecord, known: frozenset[str]) -> set[str]:
    """Every keyword one record's effects grant or remove, or that a token it
    created carries, in the head's spelling (:func:`changed_keyword`).

    ``known`` holds Forge's keywords in :func:`normalize_keyword`'s spelling;
    a granted static's text, which Forge lists beside the keywords, is none.
    """
    raw: list[str] = []
    for event in events_of(record):
        raw.extend(_event_keywords(event))
    for contribution in _contributions(record):
        change = contribution.keyword_changes
        raw.extend((*change.gained, *change.lost))
    names = set()
    for keyword in raw:
        name = changed_keyword(keyword)
        if name in OVERLAY_KEYWORDS or normalize_keyword(keyword) in known:
            names.add(name)
    return names


def _contributions(record: EffectRecord) -> tuple[Contribution, ...]:
    """A continuous record's per-entity contributions; none for any other."""
    if isinstance(record.payload, ContinuousPayload):
        return record.payload.contributions
    return ()


def _token_traits(event: Event) -> dict:
    """The characteristics a created token carries."""
    return event.params.get("characteristics") or {}


def _event_subtypes(event: Event) -> list[str]:
    """The subtypes one event grants, removes or gives a created token."""
    if event.type is EventType.TYPE_CHANGE:
        return [
            name for key in ("types_added", "types_removed")
            for name in subtypes_of(event.params.get(key) or ())
        ]
    if event.type is EventType.TOKEN_CREATED:
        return list(subtypes_of(type_tokens(str(_token_traits(event).get("types", "")))))
    return []


def _event_keywords(event: Event) -> list[str]:
    """The raw keywords one event grants, removes or gives a created token."""
    if event.type is EventType.KEYWORD_CHANGE:
        return [str(keyword) for keyword in event.params.get("keywords") or ()]
    if event.type is EventType.TOKEN_CREATED:
        keywords = _token_traits(event).get("keywords") or ()
        return [keywords] if isinstance(keywords, str) else list(map(str, keywords))
    return []


def ordered_subtypes(names: Iterable[str]) -> tuple[str, ...]:
    """Subtypes in Forge's own order (``TypeLists.txt``)."""
    return tuple(sorted(set(names) & _SUBTYPE_ORDER.keys(), key=_SUBTYPE_ORDER.__getitem__))


def ordered_keywords(names: Iterable[str]) -> tuple[str, ...]:
    """The overlay keywords first, in their order, which gate 2 and the
    probes read; then every other keyword alphabetically."""
    rest = sorted(set(names) - set(OVERLAY_KEYWORDS))
    return (*OVERLAY_KEYWORDS, *rest)
