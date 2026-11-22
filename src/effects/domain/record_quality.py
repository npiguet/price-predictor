"""Which records a curated corpus refuses on sight (FR-148): four rules.

A resolution with no acting ability, or whose acting keys all map to no
rendered line, is an outcome with no cause, and a trigger, continuous or
decision record whose acting keys all name no sidecar would train with no
acting text; a record with more events than
any single ability resolves is a whole game landed on one record; and a key
inside a sidecar's declared range but in neither of its lists is a
reconversion (or, a handful of times, a charm cast through Flashback, its mode
keyed as the keyword trait plus an ``option`` no line carries).

An unattributed event is watched, not refused: the bracket collector records
only events inside the ability's own resolution, so the outcome is that
ability's even when no clause claims it, and every combat-damage event is
``unresolved`` by design.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from effects.domain.event_schema import AttributionKind
from effects.domain.provenance import KeyResolution, ProvenanceKey
from effects.domain.records import EffectRecord, RecordKind, acting_keys, events_of


class QualityDefect(StrEnum):
    """Why a record is refused, as the manifest's ``quality_dropped`` spells it."""

    NO_ABILITY = "no-ability"
    EVENT_FLOOD = "event-flood"
    NO_ACTING_TEXT = "no-acting-text"
    UNJOINABLE_KEY = "unjoinable-key"


#: The join answers that mean no sidecar says what the key is: the script was
#: never converted, its tree has no root in this run, or the key is a keyword's
#: bare key naming none of several lines. ``DROPPED`` (a design) and
#: ``RUNTIME_ONLY`` (text the join cannot reach) are not gaps.
NO_SIDECAR = frozenset({
    KeyResolution.UNCONVERTED, KeyResolution.UNCONFIGURED_TREE, KeyResolution.AMBIGUOUS,
})

#: More events than this on one record is a game, not an ability; board wipes
#: over two full boards sit under forty.
MAX_EVENTS_PER_RECORD = 64


def quality_defect(
    record: EffectRecord, *, max_events: int = MAX_EVENTS_PER_RECORD,
) -> QualityDefect | None:
    """The first reason this record is refused, or None when it is clean.

    A non-positive ``max_events`` turns the event-flood rule off, as
    ``CapHeap`` reads a non-positive cap; as a literal ceiling, zero would
    refuse nearly every record.
    """
    if record.kind is RecordKind.RESOLUTION and not record.ability:
        return QualityDefect.NO_ABILITY
    events = events_of(record)
    if max_events > 0 and len(events) > max_events:
        return QualityDefect.EVENT_FLOOD
    return None


def unjoinable_key_defect(
    record: EffectRecord, resolve: Callable[[ProvenanceKey], KeyResolution],
) -> QualityDefect | None:
    """``UNJOINABLE_KEY`` when an acting key is one its sidecar cannot place.

    Every kind with acting keys, not only resolutions: the surface builder
    and trainer join them all and would raise on the same key later.
    ``resolve`` raises ``KeyError`` on such a key.
    """
    for key in acting_keys(record):
        try:
            resolve(key)
        except KeyError:
            return QualityDefect.UNJOINABLE_KEY
    return None


def acting_text_defect(
    record: EffectRecord, resolve: Callable[[ProvenanceKey], KeyResolution],
) -> QualityDefect | None:
    """``NO_ACTING_TEXT`` when every acting key maps to no rendered line.

    A ``RUNTIME_ONLY`` key keeps the record: it is a real ability whose text
    the join does not reach. So does a permanent spell's implicit cast
    (:func:`is_permanent_spell`): the record of a permanent resolving is kept,
    keyed to the permanent's card by that key (FR-059), though no line
    carries text for it. Any other kind with acting keys (a trigger, a
    continuous effect, a decision) is refused only where every key names no
    sidecar at all (:data:`NO_SIDECAR`): such a record would train with no
    acting text, and a key the join leaves textless on purpose is no gap.
    Apart from ``quality_defect`` because it needs the sidecars, which
    ``validate-corpus`` does not have; run ``unjoinable_key_defect`` first,
    since ``resolve`` raises on such a key.
    """
    if record.kind is RecordKind.RESOLUTION:
        outcomes = {key: resolve(key) for key in record.ability or ()}
        return None if not outcomes or _keeps_resolution(outcomes) else QualityDefect.NO_ACTING_TEXT
    outcomes = {key: resolve(key) for key in acting_keys(record)}
    if outcomes and set(outcomes.values()) <= NO_SIDECAR:
        return QualityDefect.NO_ACTING_TEXT
    return None


def _keeps_resolution(outcomes: dict[ProvenanceKey, KeyResolution]) -> bool:
    """A line or a runtime-only trait among the keys, or a permanent spell."""
    return bool(set(outcomes.values()) & {KeyResolution.LINE, KeyResolution.RUNTIME_ONLY}) or any(
        is_permanent_spell(key, outcome) for key, outcome in outcomes.items())


def is_permanent_spell(key: ProvenanceKey, resolution: KeyResolution) -> bool:
    """Whether ``key`` is a permanent's implicit cast spell: the first spell
    trait of a face, which the converter drops because the card's own lines
    carry everything it does. A creature or other permanent spell resolves
    through it alone."""
    return (
        resolution is KeyResolution.DROPPED and key.trait_kind == "spell"
        and key.index_within_kind == 0 and key.option is None
    )


def record_refusal(
    record: EffectRecord, resolve: Callable[[ProvenanceKey], KeyResolution],
    *, max_events: int = MAX_EVENTS_PER_RECORD,
) -> QualityDefect | None:
    """The one refusal rule: every pass that counts kept records applies it,
    so the survey, the write pass and the held-out coverage round agree."""
    return (
        quality_defect(record, max_events=max_events)
        or unjoinable_key_defect(record, resolve)
        or acting_text_defect(record, resolve)
    )


def has_unattributed_events(record: EffectRecord) -> bool:
    """Whether any of this record's events names no producing clause.

    A watched number over every kind, combat included, so it reads the
    collector's attribution rather than which kinds a rule reaches.
    """
    return any(
        event.attribution is AttributionKind.UNRESOLVED for event in events_of(record)
    )
