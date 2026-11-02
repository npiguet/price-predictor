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
  default (``prevented``, ``updated`` …), or a list of them (a skipped step
  is ``replaced`` by a one-shot effect and ``skipped`` by a lasting one);
- ``mode``: the replacement's Forge mode, which the incoming event carries as
  its ``mode`` parameter (``Moved``, ``DamageDone``, ``BeginPhase``); the one
  name for an event the vocabulary has no type of its own for, such as a
  phase beginning (a ``state_flag_change``);
- any parameter the incoming event's schema lists (``event_schema.EVENT_PARAMS``,
  as ``outcome_event`` reads them): ``from_zone``/``to_zone`` of a
  ``zone_change``, ``combat`` of a ``damage_dealt`` …, text compared without
  case;
- ``keyword``: the replacing line is the line this keyword produced
  (``etbCounter``, ``Madness``, ``Tribute``, ``Unleash``), compared in
  ``keyword_names``' spelling;
- ``prevention``: true to count only prevention effects. Forge reports a
  prevention as ``prevented`` only for a ``Prevent$ True`` replacement or
  protection; a shield (``PreventDamage``, ``PreventionEffect$ True``)
  reports ``replaced`` when it absorbs the whole event and ``updated`` when
  it absorbs part, as a doubling or a redirection also does. So a prevention
  effect is recognised by its line, whose chain prevents (``Prevent…`` in a
  key or an API, ``Fog``, a protection keyword), or by a ``prevented``
  result, and its ``fires`` side is any of the three results that prevented
  something; ``not_replaced`` (damage that can't be prevented) is its near
  miss. A ``result`` given beside it still names the ``fires`` side.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EventType
from effects.domain.provenance import SidecarLine, SidecarLookup
from effects.domain.records import EffectRecord, RecordKind, RewriteResult
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import acting_line, is_keyword_line
from effects.domain.rule_detectors.outcome_event import (
    EVENT_FILTERS,
    event_filters,
    meets_filters,
    same_value,
)

_OWN = frozenset({"event", "result", "mode", "keyword", "prevention"})

#: A chain that prevents: a ``PreventDamage``/``Prevent$``/``PreventionEffect$``
#: key or API, a ``Fog``, or protection printed or granted.
PREVENTS = re.compile(r"Prevent|\bFog\b|^Protection|Keyword\$ Protection")

#: The results by which a prevention effect prevented some of the event.
PREVENTING = frozenset({RewriteResult.PREVENTED, RewriteResult.REPLACED, RewriteResult.UPDATED})


def is_prevention_line(line: SidecarLine | None) -> bool:
    """Whether the replacing line's chain is a prevention effect's."""
    return line is not None and bool(line.script_text) and bool(
        PREVENTS.search(line.script_text))


def _results(value: Any) -> frozenset[RewriteResult]:
    values = [value] if isinstance(value, str | RewriteResult) else list(value)
    return frozenset(RewriteResult(v) for v in values)


@dataclass(frozen=True, slots=True)
class RewriteEvent:
    id: str = "rewrite_event"
    params: frozenset[str] = _OWN | EVENT_FILTERS
    general: frozenset[str] = frozenset({"event", "result", "prevention", "keyword"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        filters = event_filters(self.id, event, params, _OWN)
        prevention = bool(params.get("prevention"))
        wanted = (_results(params["result"]) if "result" in params
                  else PREVENTING if prevention else frozenset({RewriteResult.REPLACED}))
        if record.kind is not RecordKind.REWRITE:
            return None
        incoming = record.payload.incoming
        if incoming.type is not event:
            return None
        found = incoming.params or {}
        if "mode" in params and not same_value(found.get("mode"), params["mode"]):
            return None
        if not meets_filters(incoming, filters):
            return None
        result = record.payload.result
        if result is None:
            return None  # collected before the result was recorded: no side to read
        if "keyword" in params or prevention:
            line = acting_line(record, lines)
            if "keyword" in params and not is_keyword_line(line, params["keyword"]):
                return None
            if prevention and not (result is RewriteResult.PREVENTED
                                   or is_prevention_line(line)):
                return None
        fires = result in wanted
        return FamilyMatch(Polarity.FIRES if fires else Polarity.DOES_NOT_FIRE, fires,
                           tuple(incoming.subjects))


FAMILY = register(RewriteEvent())
