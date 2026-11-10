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
- ``targeted``: true to count only stack objects that had a target, so a
  fizzle rule's near miss is a targeted spell that resolved. On a gen-3 effect
  half the targets read at resolution have lost the illegal ones, so a
  fizzled half counts as targeted by definition.

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
        return _Observed(outcome, tuple(dict.fromkeys((*source, *refs.targets))), source,
                         bool(refs.targets))
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
    targeted = bool(refs.targets) or payload.outcome is ResolutionOutcome.FIZZLED
    return _Observed(payload.outcome, tuple(dict.fromkeys((*source, *refs.targets))), source,
                     targeted)


@dataclass(frozen=True, slots=True)
class ResolutionOutcomeFamily:
    id: str = "resolution_outcome"
    params: frozenset[str] = frozenset({"outcome", "near", "targeted"})
    general: frozenset[str] = frozenset({"outcome", "near"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        fires = ResolutionOutcome(params["outcome"])
        near = {ResolutionOutcome(o) for o in params.get("near", ["resolved"])}
        observed = _observed(record, ResolutionOutcome.COUNTERED in near | {fires})
        if observed is None or (params.get("targeted") and not observed.targeted):
            return None
        if observed.outcome is fires:
            return FamilyMatch(Polarity.FIRES, True, observed.condition,
                               observed.outcome_entities)
        if observed.outcome in near:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, observed.condition,
                               observed.outcome_entities)
        return None


FAMILY = register(ResolutionOutcomeFamily())
