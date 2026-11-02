"""Detector family ``resolution_outcome``: how a spell or ability left the stack.

The cost half of every resolution pair records the stack object's
``outcome``: ``resolved``, ``fizzled`` (every target became illegal,
608.2b), ``partially_fizzled``, ``declined`` or ``countered`` (701.6a). A rule
about leaving the stack is on its ``fires`` side when the outcome is the one
it names, and on its near miss when the outcome is one of ``near``.

Parameters:

- ``outcome`` (required): the outcome on the ``fires`` side;
- ``near``: the outcomes of the near miss, ``["resolved"]`` by default;
- ``targeted``: true to count only stack objects that had a target, so a
  fizzle rule's near miss is a targeted spell that resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionOutcome
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register


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
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.ACTIVATION:
            return None
        if params.get("targeted") and not record.state.refs.targets:
            return None
        outcome = record.payload.outcome
        source = (record.state.refs.source,) if record.state.refs.source else ()
        if outcome is fires:
            return FamilyMatch(Polarity.FIRES, True, source)
        if outcome in near:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, source)
        return None


FAMILY = register(ResolutionOutcomeFamily())
