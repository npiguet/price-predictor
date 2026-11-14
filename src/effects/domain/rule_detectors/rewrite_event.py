"""Detector family ``rewrite_event``: a replacement effect meeting an event.

A ``rewrite`` record is written whenever a replacement or prevention effect
is checked against an event: ``incoming`` is the event as proposed, and
``result`` what the handler did with it. A replacement rule is on its
``fires`` side when the event of its kind was replaced as the rule says, and
on its near miss when such an event met the effect and was not. A gen-3
record whose ``incoming_unknown`` says Forge could not copy the event as it
arrived shows neither side: its ``incoming`` is the event as it left, which
may differ from what came in in type, subjects and amounts. A set-aside
ability's run (``buffered_events``) is no event of this family.

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
- ``amount_change``: ``doubled``, ``increased`` or ``decreased``, the
  outgoing event's ``amount`` against the incoming one's (a doubler, a
  damage replacement adding to it, a partial prevention); a rewrite with no
  readable outgoing amount speaks for nothing;
- ``keyword``: the replacing line is the line this keyword produced
  (``etbCounter``, ``Madness``, ``Tribute``, ``Unleash``), compared in
  ``keyword_names``' spelling;
- ``paid_as_cost``: true to count only an event a cost's action made
  (118.11): it names a ``cause`` and the stack is empty
  (``global.stack_size`` 0), so no spell or ability was resolving, and the
  cause's move was the payment Forge makes before it puts an activated
  ability on the stack (or a mana ability's). A cost paid while the stack
  holds something cannot be told from a resolution's effect there and is on
  neither side;
- ``prevention``: true to count only prevention effects. Forge reports a
  prevention as ``prevented`` only for a ``Prevent$ True`` replacement or
  protection; a shield (``PreventDamage``, ``PreventionEffect$ True``)
  reports ``replaced`` when it absorbs the whole event and ``updated`` when
  it absorbs part, as a doubling or a redirection also does. So a prevention
  effect is recognised by its line, whose chain prevents (``Prevent…`` in a
  key or an API, ``Fog``, a protection keyword), or by a ``prevented``
  result, and its ``fires`` side is any of the three results that prevented
  something; ``not_replaced`` (damage that can't be prevented) is its near
  miss. A ``result`` given beside it still names the ``fires`` side;
- ``outcome_event``'s subject qualifiers (``subject_is_copy``,
  ``subject_merged``, ``subject_face_state``, ``subject_designation``,
  ``subject_is_token`` …): the incoming event's subject is such an object in
  the snapshot (a merged or melded permanent moving, a copy entering), which
  picks the events the rule speaks for on both sides; an event whose subjects
  cannot be read is on neither.

The condition entities are the incoming event's subjects and the replacing
effect's source; the outcome entities are the subjects of the event as
proposed and as it went out, where the replacement's result falls (the
creature the damage was prevented on, the player whose draw was replaced).
A rewrite that ran another ability carries no per-entity target, its
outcome arriving as that ability's resolution record.
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
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import acting_line, is_keyword_line, source_ids
from effects.domain.rule_detectors.outcome_event import (
    EVENT_FILTER_CHECKS,
    EVENT_FILTERS,
    QUALIFIER_CHECKS,
    SUBJECT_QUALIFIER_NEEDS,
    SUBJECT_QUALIFIERS,
    event_filters,
    meets_filters,
    same_value,
    subject_verdict,
)

#: How the outgoing event's ``amount`` compares with the incoming one's.
AMOUNT_CHANGES = frozenset({"doubled", "increased", "decreased"})
_OWN = frozenset({"event", "result", "mode", "keyword", "prevention",
                  "amount_change", "paid_as_cost"}) | frozenset(
    SUBJECT_QUALIFIERS)

#: A chain that prevents: a ``PreventDamage``/``Prevent$``/``PreventionEffect$``
#: key or API, a ``Fog``, or protection printed or granted.
PREVENTS = re.compile(r"Prevent|\bFog\b|^Protection|Keyword\$ Protection")

#: The results by which a prevention effect prevented some of the event.
PREVENTING = frozenset({RewriteResult.PREVENTED, RewriteResult.REPLACED, RewriteResult.UPDATED})


def is_prevention_line(line: SidecarLine | None) -> bool:
    """Whether the replacing line's chain is a prevention effect's."""
    return line is not None and bool(line.script_text) and bool(
        PREVENTS.search(line.script_text))


def _amount_changed(incoming, outgoing, change: str) -> bool:
    """Whether the outgoing event's ``amount`` compares with the incoming one's
    as ``change`` says; False where either is unreadable."""
    before = (incoming.params or {}).get("amount")
    after = None if outgoing is None else (outgoing.params or {}).get("amount")
    if not isinstance(before, int) or not isinstance(after, int):
        return False
    return {"doubled": after == 2 * before and before > 0, "increased": after > before,
            "decreased": after < before}[change]


def paid_as_cost(record: EffectRecord, incoming) -> bool:
    """Whether the incoming event is a cost's action: it names a cause and
    nothing was on the stack, so nothing was resolving (see the module
    docstring's ``paid_as_cost``)."""
    return bool((incoming.params or {}).get("cause")) and record.state.global_.stack_size == 0


def _results(value: Any) -> frozenset[RewriteResult]:
    values = [value] if isinstance(value, str | RewriteResult) else list(value)
    return frozenset(RewriteResult(v) for v in values)


@dataclass(frozen=True, slots=True)
class RewriteEvent:
    id: str = "rewrite_event"
    params: frozenset[str] = _OWN | EVENT_FILTERS
    general: frozenset[str] = frozenset({"event", "result", "prevention", "keyword",
                                         "paid_as_cost"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset(field for name, field in SUBJECT_QUALIFIER_NEEDS.items()
                         if name in params)

    def events(self, params: Mapping[str, Any]) -> frozenset[EventType]:
        return frozenset({EventType(params["event"])})

    def validate(self, params: Mapping[str, Any]) -> None:
        event_filters(self.id, EventType(params["event"]), params, _OWN)
        dv.check(self.id, params, {**EVENT_FILTER_CHECKS, **QUALIFIER_CHECKS,
                                   "keyword": dv.keyword, "prevention": dv.boolean,
                                   "paid_as_cost": dv.boolean,
                                   "mode": dv.mode_of_rewrite,
                                   "amount_change": dv.one_of(*AMOUNT_CHANGES)})
        if params.get("paid_as_cost") is False:
            raise ValueError(f"{self.id}: paid_as_cost is declared true or left out; "
                             "an event no cost made is no population the record reads")
        if "result" in params:
            _results(params["result"])

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        filters = event_filters(self.id, event, params, _OWN)
        prevention = bool(params.get("prevention"))
        wanted = (_results(params["result"]) if "result" in params
                  else PREVENTING if prevention else frozenset({RewriteResult.REPLACED}))
        if record.kind is not RecordKind.REWRITE:
            return None
        incoming = record.payload.arrived
        if incoming is None:
            return None  # what came in is unknown: its event, subjects and amounts
        # A set-aside ability's run (``buffered_events``) is a place of its
        # own (``emittable_events.BUFFERED``), never this family's event.
        if incoming.type is not event:
            return None
        found = incoming.params or {}
        if "mode" in params and not same_value(found.get("mode"), params["mode"]):
            return None
        if not meets_filters(incoming, filters):
            return None
        if not subject_verdict(incoming.subjects, record.state, params):
            return None  # not the object the rule is about, or unreadable
        if params.get("paid_as_cost") and not paid_as_cost(record, incoming):
            return None
        result = record.payload.result
        if result is None:
            return None  # collected before the result was recorded: no side to read
        if "amount_change" in params and not _amount_changed(
                incoming, record.payload.outgoing, params["amount_change"]):
            return None
        if "keyword" in params or prevention:
            line = acting_line(record, lines)
            if "keyword" in params and not is_keyword_line(line, params["keyword"]):
                return None
            if prevention and not (result is RewriteResult.PREVENTED
                                   or is_prevention_line(line)):
                return None
        fires = result in wanted
        outgoing = record.payload.outgoing
        outcome = tuple(dict.fromkeys((
            *incoming.subjects, *(outgoing.subjects if outgoing is not None else ()))))
        condition = tuple(dict.fromkeys((*incoming.subjects, *source_ids(record))))
        return FamilyMatch(Polarity.FIRES if fires else Polarity.DOES_NOT_FIRE, fires,
                           condition, outcome)


FAMILY = register(RewriteEvent())
