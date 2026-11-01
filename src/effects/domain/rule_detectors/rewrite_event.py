"""Detector family ``rewrite_event``: a replacement effect meeting an event.

A ``rewrite`` record is written whenever a replacement or prevention effect
is checked against an event: ``incoming`` is the event as proposed, and
``result`` what the handler did with it. A replacement rule is on its
``fires`` side when the event of its kind was replaced as the rule says, and
on its near miss when such an event met the effect and was not.

Parameters:

- ``event`` (required): the incoming event's type (``damage_dealt``,
  ``zone_change``, ``card_drawn`` …);
- ``result``: the handler result on the ``fires`` side, ``replaced`` by
  default (``prevented``, ``updated`` …);
- ``to_zone``: for a ``zone_change``, the destination the incoming event names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, RecordKind, RewriteResult
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register


@dataclass(frozen=True, slots=True)
class RewriteEvent:
    id: str = "rewrite_event"
    params: frozenset[str] = frozenset({"event", "result", "to_zone"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        wanted = RewriteResult(params.get("result", RewriteResult.REPLACED))
        if record.kind is not RecordKind.REWRITE:
            return None
        incoming = record.payload.incoming
        if incoming.type is not event:
            return None
        if "to_zone" in params and (incoming.params or {}).get("to_zone") != params["to_zone"]:
            return None
        if record.payload.result is None:
            return None  # collected before the result was recorded: no side to read
        fires = record.payload.result is wanted
        return FamilyMatch(Polarity.FIRES if fires else Polarity.DOES_NOT_FIRE, fires,
                           tuple(incoming.subjects))


FAMILY = register(RewriteEvent())
