"""Detector family ``resolution_outcome``: how a spell or ability left the stack.

A stack object leaves the stack ``resolved``, ``fizzled`` (every target
became illegal, 608.2b), ``partially_fizzled``, ``declined`` or ``countered``
(701.6a). A rule about leaving the stack is on its ``fires`` side when the
outcome is the one it names, and on its near miss when the outcome is one of
``near``.

Where the outcome is written depends on the generation. Gen-1 and gen-2 stamp
it on the cost half. Gen-3 (root spec § 8.14, FR-042b) writes it on the
effect half, and a countered object writes no effect half at all: its fate is
the countering clause's effect half, which holds the countered object's
``spell_countered`` event. So on gen-3 a ``countered`` observation is a half
holding ``spell_countered`` (whose subjects are the countered objects), read
as such only for a rule naming ``countered``, and every other outcome is the
effect half's own.

Parameters:

- ``outcome`` (required): the outcome on the ``fires`` side;
- ``near``: the outcomes of the near miss, ``["resolved"]`` by default;
- ``keyword``: the record acts through this keyword's line (ward's
  trigger, the half that counters or resolves);
- ``source_keyword``: the stack object's card prints this keyword (a gift
  spell): the countered object's card on a countering half;
- ``targeted``: true to count only stack objects that had a target, so a
  fizzle rule's near miss is a targeted spell that resolved. The targets are
  read as cast (``Refs.as_cast``: ``targets_at_cast``, else ``targets``); a
  fizzled half of a record without ``targets_at_cast``, whose ``targets`` lost
  the illegal ones, counts as targeted by definition.

The condition entities are the stack object (``refs.source``, or the
countered object) and its targets, whose legality a fizzle reads; the outcome
entity is the stack object, which is what leaves the stack resolved, fizzled
or countered. The outcome is read at the record, through the verdict head's
``resolution_outcome``, on the effect half that carries it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    card_has_keyword,
    is_keyword_line,
)


@dataclass(frozen=True, slots=True)
class _Observed:
    outcome: ResolutionOutcome
    condition: tuple[str, ...]
    outcome_entities: tuple[str, ...]
    targeted: bool


def _observed(record: EffectRecord, countered_asked: bool) -> _Observed | None:
    """The stack object's outcome a record shows, or None where it shows none."""
    if record.kind is not RecordKind.RESOLUTION:
        return None
    refs = record.state.refs
    source = (refs.source,) if refs.source else ()
    if record.moment is Moment.ACTIVATION:
        outcome = getattr(record.payload, "outcome", None)
        if outcome is None:
            return None
        return _Observed(outcome, tuple(dict.fromkeys((*source, *refs.as_cast))), source,
                         bool(refs.as_cast))
    payload = record.payload
    if not isinstance(payload, ResolutionPayload):
        return None
    if countered_asked:
        countered = tuple(dict.fromkeys(
            subject for event in payload.events if event.type is EventType.SPELL_COUNTERED
            for subject in event.subjects))
        if countered:
            return _Observed(ResolutionOutcome.COUNTERED, countered, countered, True)
    if payload.outcome is None:
        return None
    # The targets as cast (V5 #25); a fizzled half with no targets_at_cast
    # (an older record) had a target by definition.
    targeted = bool(refs.as_cast) or (
        refs.targets_at_cast is None and payload.outcome is ResolutionOutcome.FIZZLED)
    return _Observed(payload.outcome, tuple(dict.fromkeys((*source, *refs.as_cast))), source,
                     targeted)


def _prints(record: EffectRecord, lines: SidecarLookup | None, observed: _Observed,
            keyword: str) -> bool:
    """Whether the stack object's card prints ``keyword``: the countered
    object's on a countering half, the acting line's card otherwise."""
    if observed.outcome is ResolutionOutcome.COUNTERED and observed.outcome_entities != (
            (record.state.refs.source,) if record.state.refs.source else ()):
        return any(entity is not None and card_has_keyword(entity.line_keys, lines, keyword)
                   for entity in map(record.state.entity, observed.outcome_entities))
    return card_has_keyword(record.ability or (), lines, keyword)


@dataclass(frozen=True, slots=True)
class ResolutionOutcomeFamily:
    id: str = "resolution_outcome"
    params: frozenset[str] = frozenset({"outcome", "near", "targeted", "source_keyword",
                                        "keyword"})
    general: frozenset[str] = frozenset({"outcome", "near"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def validate(self, params: Mapping[str, Any]) -> None:
        ResolutionOutcome(params["outcome"])
        for near in params.get("near", ["resolved"]):
            ResolutionOutcome(near)
        dv.check(self.id, params, {"targeted": dv.boolean, "source_keyword": dv.keyword,
                                   "keyword": dv.keyword})

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        fires = ResolutionOutcome(params["outcome"])
        near = {ResolutionOutcome(o) for o in params.get("near", ["resolved"])}
        observed = _observed(record, ResolutionOutcome.COUNTERED in near | {fires})
        if observed is None or (params.get("targeted") and not observed.targeted):
            return None
        keyword = params.get("source_keyword")
        if keyword is not None and not _prints(record, lines, observed, keyword):
            return None
        if "keyword" in params and not is_keyword_line(acting_line(record, lines),
                                                       params["keyword"]):
            return None
        if observed.outcome is fires:
            return FamilyMatch(Polarity.FIRES, True, observed.condition,
                               observed.outcome_entities)
        if observed.outcome in near:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, observed.condition,
                               observed.outcome_entities)
        return None


FAMILY = register(ResolutionOutcomeFamily())
