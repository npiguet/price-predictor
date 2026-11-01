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
  case;
- ``token_kind``: a ``token_created`` event's token has this subtype
  (``Treasure``, ``Food``: a kind rule 111.10 defines);
- ``min_count``: how many such events the record must hold, 1 by default;
- ``api``: the acting line's script API type (``ChangeZone``, ``Token`` …),
  which restricts both sides to resolutions of that API;
- ``keyword``: the acting line is the line this keyword produced (``Equip``,
  ``Cycling``, ``Outlast``: an ability a keyword generates acts through the
  keyword's own line, whose API is no API of its own), compared in
  ``keyword_names``' spelling; like ``api`` it restricts both sides;
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
- ``with_event``, ``without_event``: the same record does, or does not, hold
  an event of this type as well (the sacrifice beside the counters it pays for,
  the Equipment entering with no ``attached`` event).

A qualifier cannot be combined with ``absent``.

Prior art: ``event_schema.EVENT_PARAMS``, the per-type parameter table every
reader of events already trusts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.event_schema import EVENT_PARAMS, Event, EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    is_keyword_line,
    root_params_of,
    script_params_match,
    source_is_token_of_kind,
)
from effects.domain.state_snapshot import EntityState, StateSnapshot

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


def _in_combat(entity: EntityState) -> bool | None:
    combat = entity.combat
    return combat is not None and bool(combat.attacking or combat.blocking)


def _is_token(entity: EntityState) -> bool | None:
    return entity.token_script_id is not None


def _controller_is_owner(entity: EntityState) -> bool | None:
    owner = getattr(entity, "owner", None)  # gen-3 snapshots only
    return None if owner is None else owner == entity.controller


#: Qualifiers read off the event's subject in the snapshot; None is unreadable.
_SUBJECT_QUALIFIERS = {
    "subject_in_combat": _in_combat,
    "subject_is_token": _is_token,
    "controller_is_owner": _controller_is_owner,
}
_RECORD_QUALIFIERS = frozenset({"with_event", "without_event"})
_QUALIFIERS = frozenset(_SUBJECT_QUALIFIERS) | _RECORD_QUALIFIERS
_OWN = frozenset({"event", "min_count", "api", "keyword", "script_param", "absent",
                  "token_kind", "source_token_kind"}) | _QUALIFIERS
#: The record kinds whose payload lists the events a rule's outcome is.
_EVENT_KINDS = frozenset({RecordKind.RESOLUTION, RecordKind.COMBAT, RecordKind.TURN_ACTION})


def acting_api(record: EffectRecord, lines: SidecarLookup | None) -> str | None:
    """The acting line's script API type, or None where no line is readable."""
    line = acting_line(record, lines)
    return line.script_api_type if line is not None else None


def same_value(found: Any, wanted: Any) -> bool:
    """An event parameter's equality, text compared without case."""
    if isinstance(found, str) and isinstance(wanted, str):
        return found.lower() == wanted.lower()
    return found == wanted


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
        found = [_SUBJECT_QUALIFIERS[name](entity) for name in wanted]
        if None in found:
            continue
        verdicts.append(all(f == v for f, v in zip(found, wanted.values(), strict=True)))
    return any(verdicts) if verdicts else None


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    id: str = "outcome_event"
    params: frozenset[str] = _OWN | EVENT_FILTERS

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset({"owner"}) if "controller_is_owner" in params else frozenset()

    def _filters(self, event: EventType, params: Mapping[str, Any]) -> dict[str, Any]:
        filters = event_filters(self.id, event, params, _OWN)
        if "token_kind" in params and event is not EventType.TOKEN_CREATED:
            raise ValueError("outcome_event: token_kind reads a token_created event")
        for name in _RECORD_QUALIFIERS & set(params):
            EventType(params[name])  # an unknown type raises
        return filters

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        filters = self._filters(event, params)
        api, keyword = params.get("api"), params.get("keyword")
        scoped = api is not None or keyword is not None
        absent = bool(params.get("absent"))
        qualified = bool(_QUALIFIERS & set(params))
        if absent and not scoped:
            raise ValueError("outcome_event: absent needs api or keyword, or every "
                             "resolution fires")
        if absent and qualified:
            raise ValueError("outcome_event: a qualifier picks among events that happened, "
                             "which absent says did not")
        if record.kind not in _EVENT_KINDS or (
                record.kind is RecordKind.RESOLUTION and record.moment is not Moment.RESOLUTION):
            return None
        if scoped or "script_param" in params:
            line = acting_line(record, lines)
            if api is not None and (line is None or line.script_api_type != api):
                return None
            if keyword is not None and not is_keyword_line(line, keyword):
                return None
            if "script_param" in params and not script_params_match(
                    root_params_of(line), params["script_param"]):
                return None
        kind = params.get("source_token_kind")
        if kind is not None and not source_is_token_of_kind(record.state, kind, record.ability):
            return None
        token_kind = params.get("token_kind")
        listed = record.payload.listed_events()
        wanted = [
            e for e in listed
            if e.type is event
            and all(same_value((e.params or {}).get(name), value)
                    for name, value in filters.items())
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
        types = {e.type for e in listed}
        record_ok = all((
            "with_event" not in params or EventType(params["with_event"]) in types,
            "without_event" not in params or EventType(params["without_event"]) not in types,
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
