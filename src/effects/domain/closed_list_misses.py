"""The subtypes and keywords a record changes that the head's closed output lists lack.

``effect_targets`` writes a subtype or keyword into its column and nothing for a
name the lists (:mod:`effects.domain.output_vocabularies`) do not hold, which a
reader takes for "none gained". A name outside the lists is counted here, so
``RecordTargets.unsupervised`` and the corpus manifest say how many were dropped.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from effects.domain.closed_lists import subtypes_of, type_tokens
from effects.domain.continuous_lines import line_changes
from effects.domain.event_schema import EventType
from effects.domain.input_vocabularies import vocabulary_slot
from effects.domain.keyword_names import changed_keyword
from effects.domain.output_vocabularies import KEYWORDS_CHANGED, SUBTYPES_CHANGED
from effects.domain.records import ContinuousPayload, EffectRecord, Event, events_of

SUBTYPE_MISSED = "subtype_outside_closed_list"
KEYWORD_MISSED = "keyword_outside_closed_list"


def _token_keywords(event: Event) -> list[str]:
    keywords = (event.params.get("characteristics") or {}).get("keywords") or ()
    return [keywords] if isinstance(keywords, str) else [str(k) for k in keywords]


def _type_change_names(event: Event) -> tuple[list[str], list[str]]:
    params = event.params
    changed = (*(params.get("types_added") or ()), *(params.get("types_removed") or ()))
    return list(subtypes_of(changed)), []


def _keyword_change_names(event: Event) -> tuple[list[str], list[str]]:
    if event.params.get("hidden"):
        return [], []
    return [], [str(keyword) for keyword in event.params.get("keywords") or ()]


def _token_names(event: Event) -> tuple[list[str], list[str]]:
    types = (event.params.get("characteristics") or {}).get("types", "")
    return list(subtypes_of(type_tokens(str(types)))), _token_keywords(event)


_EVENT_NAMES = {
    EventType.TYPE_CHANGE: _type_change_names,
    EventType.KEYWORD_CHANGE: _keyword_change_names,
    EventType.TOKEN_CREATED: _token_names,
}


def _event_names(event: Event) -> tuple[list[str], list[str]]:
    """``(subtypes, keywords)`` one event changes, in the spelling it wrote."""
    read = _EVENT_NAMES.get(event.type)
    return read(event) if read else ([], [])


def _continuous_names(record: EffectRecord) -> tuple[list[str], list[str]]:
    """The subtypes and keywords a continuous record's lines change, read
    against its board as the targets read them."""
    subtypes: list[str] = []
    keywords: list[str] = []
    entities = {entity.id: entity for entity in record.state.entities}
    for contribution in record.payload.contributions:
        for line in line_changes(contribution, entities.get(contribution.entity)):
            names = (*line.gained, *line.lost)
            if line.vocabulary is SUBTYPES_CHANGED:
                subtypes.extend(names)
            elif line.vocabulary is KEYWORDS_CHANGED:
                keywords.extend(names)
    return subtypes, keywords


def _names(record: EffectRecord) -> tuple[list[str], list[str]]:
    subtypes: list[str] = []
    keywords: list[str] = []
    for event in events_of(record):
        found_subtypes, found_keywords = _event_names(event)
        subtypes += found_subtypes
        keywords += found_keywords
    if isinstance(record.payload, ContinuousPayload):
        found_subtypes, found_keywords = _continuous_names(record)
        subtypes += found_subtypes
        keywords += found_keywords
    return subtypes, keywords


def _missing(names: Iterable[str], vocabulary: tuple[str, ...]) -> int:
    return sum(1 for name in names if vocabulary_slot(name, vocabulary) is None)


def closed_list_misses(record: EffectRecord) -> Counter[str]:
    """How many subtype and keyword changes of ``record`` name a value the
    closed output lists lack, by :data:`SUBTYPE_MISSED` and :data:`KEYWORD_MISSED`;
    a count of zero is absent."""
    subtypes, keywords = _names(record)
    counts = Counter()
    counts[SUBTYPE_MISSED] = _missing(subtypes, SUBTYPES_CHANGED)
    counts[KEYWORD_MISSED] = _missing(map(changed_keyword, keywords), KEYWORDS_CHANGED)
    return +counts
