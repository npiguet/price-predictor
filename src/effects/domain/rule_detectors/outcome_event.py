"""Detector family ``outcome_event``: a resolution producing an event of one kind.

Many rules are stated by what a resolution does: a permanent spell enters the
battlefield (608.3a), an instant goes to the graveyard (608.2n), a token or a
delayed trigger is created, a planeswalker enters with loyalty counters. A
record shows such a rule on its ``fires`` side when its effect half holds the
event. A cost half holds no events, only a summary of what was paid, so a cost
is no outcome this family reads.
The near miss is a resolution of the same kind of ability that did not produce
it, so it is counted only when ``api`` names that kind: without it, every other
resolution in the corpus would count as a near miss.

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
- ``absent``: true where the rule is that the event does *not* happen (a token
  whose characteristics the battlefield forbids, a duration already over): the
  half without the event is the ``fires`` side and the one holding it the near
  miss; it needs ``api``, or every other resolution would fire;
- ``source_token_kind``: the acting ability's source is a token of this kind
  (a Food's or a Clue's own ability).

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
from effects.domain.state_snapshot import StateSnapshot

#: Event parameters a declaration may not filter on: they name a card or an
#: object of one game (``cause``, ``card_name``, ``controller``), never a
#: condition of a rule, or hold a structure no equality reads.
_UNFILTERABLE = frozenset({
    "cause", "source", "card_name", "token_script_id", "copy_source", "characteristics",
    "controller", "owner", "attached_to", "detached_from", "targets", "blockers", "blocked",
    "defender", "called", "options", "piles", "mana_by_color"})
_EVENT_FILTERS = frozenset(
    name for names in EVENT_PARAMS.values() for name in names) - _UNFILTERABLE
_OWN = frozenset({"event", "min_count", "api", "absent", "token_kind", "source_token_kind"})


def acting_api(record: EffectRecord, lines: SidecarLookup | None) -> str | None:
    """The acting line's script API type, or None where no line is readable."""
    if lines is None or not record.ability:
        return None
    line = lines.first_line(record.ability)
    return line.script_api_type if line is not None else None


def _same(found: Any, wanted: Any) -> bool:
    if isinstance(found, str) and isinstance(wanted, str):
        return found.lower() == wanted.lower()
    return found == wanted


def _subtypes(type_line: str) -> frozenset[str]:
    """The subtypes of a ``Creature - Dragon Horror`` type line, lower-cased."""
    _, dash, rest = type_line.partition(" - ")
    return frozenset(word.lower() for word in rest.split()) if dash else frozenset()


def token_kind_of(event: Event) -> frozenset[str]:
    """A ``token_created`` event's token subtypes, lower-cased."""
    characteristics = (event.params or {}).get("characteristics") or {}
    return _subtypes(str(characteristics.get("types") or ""))


def source_is_token_of_kind(state: StateSnapshot, kind: str) -> bool:
    """Whether the record's source entity is a token with subtype ``kind``."""
    source = state.refs.source
    entity = next((e for e in state.entities if e.id == source), None)
    return (entity is not None and entity.token_script_id is not None
            and kind.lower() in {s.lower() for s in entity.subtypes})


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    id: str = "outcome_event"
    params: frozenset[str] = _OWN | _EVENT_FILTERS

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        return frozenset()

    def _filters(self, event: EventType, params: Mapping[str, Any]) -> dict[str, Any]:
        filters = {name: value for name, value in params.items() if name not in _OWN}
        unknown = set(filters) - set(EVENT_PARAMS.get(event, ())) | set(filters) & _UNFILTERABLE
        if unknown:
            raise ValueError(f"outcome_event: {event.value} has no parameter {sorted(unknown)}")
        if "token_kind" in params and event is not EventType.TOKEN_CREATED:
            raise ValueError("outcome_event: token_kind reads a token_created event")
        return filters

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        filters = self._filters(event, params)
        api = params.get("api")
        absent = bool(params.get("absent"))
        if absent and api is None:
            raise ValueError("outcome_event: absent needs api, or every resolution fires")
        if record.kind is not RecordKind.RESOLUTION or record.moment is not Moment.RESOLUTION:
            return None
        if api is not None and acting_api(record, lines) != api:
            return None
        kind = params.get("source_token_kind")
        if kind is not None and not source_is_token_of_kind(record.state, kind):
            return None
        token_kind = params.get("token_kind")
        wanted = [
            e for e in record.payload.listed_events()
            if e.type is event
            and all(_same((e.params or {}).get(name), value) for name, value in filters.items())
            and (token_kind is None or token_kind.lower() in token_kind_of(e))]
        happened = len(wanted) >= int(params.get("min_count", 1))
        subjects = tuple(dict.fromkeys(s for e in wanted for s in e.subjects))
        if absent:
            # The rule's outcome is the event's absence, so that is what is observed.
            if happened:
                return FamilyMatch(Polarity.DOES_NOT_FIRE, False, subjects)
            return FamilyMatch(Polarity.FIRES, True, ())
        if happened:
            return FamilyMatch(Polarity.FIRES, True, subjects)
        if api is None:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, ())


FAMILY = register(OutcomeEvent())
