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
- ``caused_with_empty_stack`` (118.11): true to count only an incoming event
  that names a ``cause`` while the stack is empty (``global.stack_size`` 0).
  That is all it reads: no record says that a cost is being paid (a rewrite's state
  names the replacement's source, never the item paying a cost, and the
  event's ``cause`` is the causing ability's card whatever the ability is
  doing). It therefore catches a cost paid with the stack empty (Forge pays
  an activated ability's costs, and a mana ability's, before the ability
  is on the stack), and with it every other event a cause made off the
  stack: a special action (a land played, ``hand`` to ``battlefield``), a
  mana ability's own effects, an ability Forge resolves without the stack
  (``ComputerUtil.playNoStack``, a ``Static$ True`` trigger's effect). It
  misses every cost paid while the stack holds something, which is on
  neither side. A declaration keeps to a cost's action through its other
  parameters (118.11-1's ``zone_change`` from the battlefield to a
  graveyard leaves out a land played);
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
from effects.domain.rule_detectors import BindsParams, EventlessFamily, FamilyMatch, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import acting_line, is_keyword_line, source_ids
from effects.domain.rule_detectors.outcome_events import (
    EVENT_FILTER_CHECKS,
    EVENT_FILTERS,
    event_filters,
    meets_filters,
    same_value,
)
from effects.domain.rule_detectors.outcome_subjects import (
    QUALIFIER_CHECKS,
    SUBJECT_QUALIFIER_NEEDS,
    SUBJECT_QUALIFIERS,
    subject_verdict,
)

#: How the outgoing event's ``amount`` compares with the incoming one's.
AMOUNT_CHANGES = frozenset({"doubled", "increased", "decreased"})
_OWN = frozenset({"event", "result", "mode", "keyword", "prevention",
                  "amount_change", "caused_with_empty_stack"}) | frozenset(
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


def caused_with_empty_stack(record: EffectRecord, incoming) -> bool:
    """Whether the incoming event names a cause while the stack is empty: a
    cost paid with nothing on the stack, or any other event a cause made off
    the stack (the module docstring's ``caused_with_empty_stack`` says which)."""
    return bool((incoming.params or {}).get("cause")) and record.state.global_.stack_size == 0


def _results(value: Any) -> frozenset[RewriteResult]:
    values = [value] if isinstance(value, str | RewriteResult) else list(value)
    return frozenset(RewriteResult(v) for v in values)


@dataclass(frozen=True, slots=True)
class RewriteEvent(BindsParams, EventlessFamily):
    id: str = "rewrite_event"
    kinds: frozenset[str] = frozenset({"rewrite"})
    params: frozenset[str] = _OWN | EVENT_FILTERS
    general: frozenset[str] = frozenset({"event", "result", "prevention", "keyword",
                                         "caused_with_empty_stack"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset(field for name, field in SUBJECT_QUALIFIER_NEEDS.items()
                         if name in params)

    def events(self, params: Mapping[str, Any]) -> frozenset[EventType]:
        return frozenset({EventType(params["event"])})

    def validate(self, params: Mapping[str, Any]) -> None:
        event_filters(self.id, EventType(params["event"]), params, _OWN)
        dv.check(self.id, params, {**EVENT_FILTER_CHECKS, **QUALIFIER_CHECKS,
                                   "keyword": dv.keyword, "prevention": dv.boolean,
                                   "caused_with_empty_stack": dv.boolean,
                                   "mode": dv.mode_of_rewrite,
                                   "amount_change": dv.one_of(*AMOUNT_CHANGES)})
        if params.get("caused_with_empty_stack") is False:
            raise ValueError(f"{self.id}: caused_with_empty_stack is declared true or left out; "
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
        if params.get("caused_with_empty_stack") and not caused_with_empty_stack(record, incoming):
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
