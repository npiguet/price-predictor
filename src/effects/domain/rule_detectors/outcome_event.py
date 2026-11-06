"""Detector family ``outcome_event``: a resolution producing an event of one kind.

Many rules are stated by what a resolution does: a permanent spell enters the
battlefield (608.3a), an instant goes to the graveyard (608.2n), a token or a
delayed trigger is created, a planeswalker enters with loyalty counters. A
record shows such a rule on its ``fires`` side when its effect half holds the
event. A cost half holds no events, only a summary of what was paid, so a cost
is no outcome this family reads. A ``combat`` record's damage step and a
``turn_action`` record's action list their events too, and a declaration of
either kind reads them the same way; neither has an acting line, so the
parameters that read one (``api``, ``keyword``, ``script_param``,
``source_token_kind``) match nothing there.
The near miss is a resolution of the same kind of ability that did not produce
it, so it is counted only when ``api`` or ``keyword`` names that kind: without
one, every other resolution in the corpus would count as a near miss.

Parameters:

- ``event`` (required): the event type (``zone_change``, ``token_created`` …);
- any parameter the event's schema lists (``event_schema.EVENT_PARAMS``):
  ``from_zone``/``to_zone`` of a ``zone_change``, ``counter_type`` of a
  ``counter_change``, ``flag`` of a ``state_flag_change``, ``out`` of a
  ``phased`` …, each an equality the event must meet, text compared without
  case and a counter type in ``normalize_counter_type``'s spelling (records
  write ``+1/+1`` and ``P1P1`` both);
- ``token_kind``: a ``token_created`` event's token has this subtype
  (``Treasure``, ``Food``: a kind rule 111.10 defines);
- ``min_count``: how many such events the record must hold, 1 by default;
- ``api``: the acting line's script API type (``ChangeZone``, ``Token`` …),
  which restricts both sides to resolutions of that API;
- ``api_in_chain``: true to let ``api`` name any segment of the acting line's
  chain, a sub-ability as well as the root (Aura Graft's ``Attach`` below its
  ``GainControl``); the root alone otherwise, which most declarations want,
  since a sub-ability may not run (an optional "you may attach", a branch a
  condition skips) and its near miss would then be no near miss;
- ``keyword``: the acting line is the line this keyword produced (``Equip``,
  ``Cycling``, ``Outlast``: an ability a keyword generates acts through the
  keyword's own line, whose API is no API of its own), compared in
  ``keyword_names``' spelling; like ``api`` it restricts both sides;
- ``source_keyword``: the acting line's card prints this keyword on any of its
  lines, read off its sidecar as ``verdict_bit`` reads it: an enters-with-
  counters keyword (amplify, modular, sunburst, bloodthirst, graft) works in
  the permanent spell's resolution, whose acting line is the implicit spell,
  no keyword's; it restricts both sides too;
- ``script_param``: the acting line's root segment holds this parameter, a
  name (``Monstrosity``, for ``Monstrosity$ True``) or a name -> value map
  (``{"Mode": "TurnFaceUp"}``, ``"*"`` for any value), which tells apart the
  keyword actions Forge writes as a parameter of a common API (monstrosity,
  bolster, adapt and support are all ``PutCounter``);
- ``absent``: true where the rule is that the event does *not* happen (a token
  whose characteristics the battlefield forbids, a duration already over): the
  half without the event is the ``fires`` side and the one holding it the near
  miss; it needs ``api`` or ``keyword``, or every other resolution would fire;
- ``source_token_kind``: the acting ability's source is a token of this kind
  (a Food's or a Clue's own ability).

Qualifiers. A rule stated about *which* such event happened is on its
``fires`` side when one of the events meets every qualifier given, and on its
near miss when the record holds such events and none meets them (a creature
leaving the battlefield that was not in combat):

- ``subject_in_combat``: true where the event's subject is attacking or
  blocking in the snapshot, false where it is not;
- ``subject_is_token``: true where the event's subject is a token in the
  snapshot, false where it is a card;
- ``controller_is_owner``: the subject's controller is (true) or is not
  (false) its owner, read off the snapshot's gen-3 ``owner``;
- ``subject_is_source``: the subject is (true) or is not (false) the acting
  ability's source, ``refs.source`` (a Saga sacrificing itself, a permanent
  sacrificed for its own unpaid upkeep);
- ``with_event``, ``without_event``: the same record does, or does not, hold
  an event of this type as well (the sacrifice beside the counters it pays for,
  the Equipment entering with no ``attached`` event): a type, or a map of
  ``event`` and that event's filters (``{"event": "zone_change", "from_zone":
  "battlefield", "to_zone": "graveyard"}``, a sacrifice as a gen-2 record
  writes one).

A qualifier cannot be combined with ``absent``.

Prior art: ``event_schema.EVENT_PARAMS``, the per-type parameter table every
reader of events already trusts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EVENT_PARAMS, Event, EventType
from effects.domain.provenance import SidecarLine, SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    card_has_keyword,
    is_keyword_line,
    root_params_of,
    script_params_match,
    source_is_token_of_kind,
)
from effects.domain.script_text import api_of, segments
from effects.domain.script_text import params as params_of
from effects.domain.state_snapshot import EntityState, StateSnapshot, normalize_counter_type

__all__ = ["EVENT_FILTERS", "FAMILY", "acting_api", "event_filters", "source_is_token_of_kind",
           "token_kind_of"]

#: Event parameters a declaration may not filter on: they name a card or an
#: object of one game (``cause``, ``card_name``, ``controller``), never a
#: condition of a rule, or hold a structure no equality reads.
_UNFILTERABLE = frozenset({
    "cause", "source", "card_name", "token_script_id", "copy_source", "characteristics",
    "controller", "owner", "attached_to", "detached_from", "targets", "blockers", "blocked",
    "defender", "called", "options", "piles", "mana_by_color"})
EVENT_FILTERS = frozenset(
    name for names in EVENT_PARAMS.values() for name in names) - _UNFILTERABLE


def _in_combat(entity: EntityState, state: StateSnapshot) -> bool | None:
    combat = entity.combat
    return combat is not None and bool(combat.attacking or combat.blocking)


def _is_token(entity: EntityState, state: StateSnapshot) -> bool | None:
    return entity.token_script_id is not None


def _controller_is_owner(entity: EntityState, state: StateSnapshot) -> bool | None:
    # None on a gen-1 or gen-2 snapshot, which records no owner.
    return None if entity.owner is None else entity.owner == entity.controller


def _is_source(entity: EntityState, state: StateSnapshot) -> bool | None:
    return None if state.refs.source is None else entity.id == state.refs.source


#: Qualifiers read off the event's subject in the snapshot; None is unreadable.
_SUBJECT_QUALIFIERS = {
    "subject_in_combat": _in_combat,
    "subject_is_token": _is_token,
    "controller_is_owner": _controller_is_owner,
    "subject_is_source": _is_source,
}
_RECORD_QUALIFIERS = frozenset({"with_event", "without_event"})
_QUALIFIERS = frozenset(_SUBJECT_QUALIFIERS) | _RECORD_QUALIFIERS
_OWN = frozenset({"event", "min_count", "api", "api_in_chain", "keyword", "source_keyword",
                  "script_param", "absent", "token_kind", "source_token_kind"}) | _QUALIFIERS
#: The record kinds whose payload lists the events a rule's outcome is.
_EVENT_KINDS = frozenset({RecordKind.RESOLUTION, RecordKind.COMBAT, RecordKind.TURN_ACTION})


def acting_api(record: EffectRecord, lines: SidecarLookup | None) -> str | None:
    """The acting line's script API type, or None where no line is readable."""
    line = acting_line(record, lines)
    return line.script_api_type if line is not None else None


def _acts_through(line: SidecarLine | None, api: str, in_chain: bool) -> bool:
    """Whether ``line``'s root API is ``api``, or under ``in_chain`` any
    segment's of its chain."""
    if line is None:
        return False
    if line.script_api_type == api:
        return True
    return in_chain and bool(line.script_text) and any(
        api_of(params_of(segment)) == api for segment in segments(line.script_text))


def same_value(found: Any, wanted: Any, name: str | None = None) -> bool:
    """An event parameter's equality, text compared without case; a counter
    type (``name`` ``counter_type``) in ``normalize_counter_type``'s spelling."""
    if isinstance(found, str) and isinstance(wanted, str):
        if name == "counter_type":
            return normalize_counter_type(found) == normalize_counter_type(wanted)
        return found.lower() == wanted.lower()
    return found == wanted


def meets_filters(event: Event, filters: Mapping[str, Any]) -> bool:
    """Whether ``event`` meets every one of a declaration's equalities."""
    found = event.params or {}
    return all(same_value(found.get(name), value, name) for name, value in filters.items())


def _event_spec(spec: Any) -> tuple[EventType, dict[str, Any]]:
    """A ``with_event``/``without_event`` value: a type, or ``event`` and its filters."""
    if isinstance(spec, Mapping):
        event = EventType(spec["event"])
        return event, event_filters("outcome_event", event, spec, frozenset({"event"}))
    return EventType(spec), {}


def _holds(listed: tuple[Event, ...], spec: Any) -> bool:
    event, filters = _event_spec(spec)
    return any(e.type is event and meets_filters(e, filters) for e in listed)


def event_filters(family: str, event: EventType, params: Mapping[str, Any],
                  own: frozenset[str]) -> dict[str, Any]:
    """The declaration's event-parameter equalities: every parameter not in
    ``own``, each one the event's schema lists and an equality can read."""
    filters = {name: value for name, value in params.items() if name not in own}
    unknown = set(filters) - set(EVENT_PARAMS.get(event, ())) | set(filters) & _UNFILTERABLE
    if unknown:
        raise ValueError(f"{family}: {event.value} has no parameter {sorted(unknown)}")
    return filters


def _subtypes(type_line: str) -> frozenset[str]:
    """The subtypes of a ``Creature - Dragon Horror`` type line, lower-cased."""
    _, dash, rest = type_line.partition(" - ")
    return frozenset(word.lower() for word in rest.split()) if dash else frozenset()


def token_kind_of(event: Event) -> frozenset[str]:
    """A ``token_created`` event's token subtypes, lower-cased."""
    characteristics = (event.params or {}).get("characteristics") or {}
    return _subtypes(str(characteristics.get("types") or ""))


def _subject_verdict(event: Event, state: StateSnapshot,
                     params: Mapping[str, Any]) -> bool | None:
    """Whether ``event`` meets every subject qualifier; None where unreadable."""
    wanted = {name: bool(params[name]) for name in _SUBJECT_QUALIFIERS if name in params}
    if not wanted:
        return True
    entities = [e for e in (state.entity(s) for s in event.subjects) if e is not None]
    verdicts = []
    for entity in entities:
        found = [_SUBJECT_QUALIFIERS[name](entity, state) for name in wanted]
        if None in found:
            continue
        verdicts.append(all(f == v for f, v in zip(found, wanted.values(), strict=True)))
    return any(verdicts) if verdicts else None


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    id: str = "outcome_event"
    params: frozenset[str] = _OWN | EVENT_FILTERS
    general: frozenset[str] = frozenset(
        {"event", "api", "api_in_chain", "keyword", "source_keyword", "absent"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset({"owner"}) if "controller_is_owner" in params else frozenset()

    def _filters(self, event: EventType, params: Mapping[str, Any]) -> dict[str, Any]:
        filters = event_filters(self.id, event, params, _OWN)
        if "token_kind" in params and event is not EventType.TOKEN_CREATED:
            raise ValueError("outcome_event: token_kind reads a token_created event")
        for name in _RECORD_QUALIFIERS & set(params):
            _event_spec(params[name])  # an unknown type or filter raises
        return filters

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        filters = self._filters(event, params)
        api, keyword = params.get("api"), params.get("keyword")
        source_keyword = params.get("source_keyword")
        scoped = api is not None or keyword is not None or source_keyword is not None
        absent = bool(params.get("absent"))
        qualified = bool(_QUALIFIERS & set(params))
        if absent and not scoped:
            raise ValueError("outcome_event: absent needs api, keyword or source_keyword, "
                             "or every resolution fires")
        if absent and qualified:
            raise ValueError("outcome_event: a qualifier picks among events that happened, "
                             "which absent says did not")
        if record.kind not in _EVENT_KINDS or (
                record.kind is RecordKind.RESOLUTION and record.moment is not Moment.RESOLUTION):
            return None
        if scoped or "script_param" in params:
            line = acting_line(record, lines)
            if api is not None and not _acts_through(line, api, bool(params.get("api_in_chain"))):
                return None
            if keyword is not None and not is_keyword_line(line, keyword):
                return None
            if "script_param" in params and not script_params_match(
                    root_params_of(line), params["script_param"]):
                return None
        if source_keyword is not None and not card_has_keyword(
                record.ability or (), lines, source_keyword):
            return None
        kind = params.get("source_token_kind")
        if kind is not None and not source_is_token_of_kind(record.state, kind, record.ability):
            return None
        token_kind = params.get("token_kind")
        listed = record.payload.listed_events()
        wanted = [
            e for e in listed
            if e.type is event
            and meets_filters(e, filters)
            and (token_kind is None or token_kind.lower() in token_kind_of(e))]
        if qualified:
            return self._qualified(record, params, wanted, listed, scoped)
        happened = len(wanted) >= int(params.get("min_count", 1))
        subjects = _subjects(wanted)
        if absent:
            # The rule's outcome is the event's absence, so that is what is observed.
            if happened:
                return FamilyMatch(Polarity.DOES_NOT_FIRE, False, subjects)
            return FamilyMatch(Polarity.FIRES, True, ())
        if happened:
            return FamilyMatch(Polarity.FIRES, True, subjects)
        if not scoped:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, ())

    @staticmethod
    def _qualified(record: EffectRecord, params: Mapping[str, Any], wanted: list[Event],
                   listed: tuple[Event, ...], scoped: bool) -> FamilyMatch | None:
        """The rule's side among the events that happened, by its qualifiers."""
        record_ok = all((
            "with_event" not in params or _holds(listed, params["with_event"]),
            "without_event" not in params or not _holds(listed, params["without_event"]),
        ))
        read = [(e, _subject_verdict(e, record.state, params)) for e in wanted]
        read = [(e, verdict) for e, verdict in read if verdict is not None]
        meeting = [e for e, verdict in read if verdict and record_ok]
        if len(meeting) >= int(params.get("min_count", 1)):
            return FamilyMatch(Polarity.FIRES, True, _subjects(meeting))
        if read:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False,
                               _subjects(e for e, _ in read if e not in meeting))
        if scoped:
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, ())
        return None


def _subjects(events) -> tuple[str, ...]:
    return tuple(dict.fromkeys(s for e in events for s in e.subjects))


FAMILY = register(OutcomeEvent())
