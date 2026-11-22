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
  which restricts both sides to resolutions of that API; or a list of them,
  any of which counts (``["Pump", "PumpAll"]``: one mechanic two APIs carry);
- ``api_in_chain``: true to let ``api`` name any segment of the acting line's
  chain, a sub-ability as well as the root (Aura Graft's ``Attach`` below its
  ``GainControl``); the root alone otherwise, which most declarations want,
  since a sub-ability may not run (an optional "you may attach", a branch a
  condition skips) and its near miss would then be no near miss;
- ``keyword``: the acting line is the line this keyword produced (``Equip``,
  ``Cycling``, ``Outlast``: an ability a keyword generates acts through the
  keyword's own line, whose API is no API of its own), compared in
  ``keyword_names``' spelling; like ``api`` it restricts both sides, and
  given beside ``api`` a line either names counts (an Attach resolution or
  an equip, both of which attach);
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
- ``chain_param``: as ``script_param``, on any segment of the acting line's
  chain, the root's included (a ``Duration$`` or ``WontPhaseInNormal$`` on
  the ``Phases`` sub-ability, an ``ExcessDamage$`` on a ``DealDamage`` below
  another root API);
- ``absent``: true where the rule is that the event does *not* happen (a token
  whose characteristics the battlefield forbids, a duration already over): the
  half without the event is the ``fires`` side and the one holding it the near
  miss; it needs ``api`` or ``keyword``, or every other resolution would fire;
- ``source_token_kind``: the acting ability's source is a token of this kind
  (a Food's or a Clue's own ability).

Qualifiers. A rule stated about *which* such event happened is on its
``fires`` side when one of the events meets every qualifier given, and on its
near miss when the record holds such events and none meets them (a creature
leaving the battlefield that was not in combat). As for the unqualified
near miss, that is counted only when ``api``, ``keyword`` or
``source_keyword`` names the kind of ability (``source_dies``' narrower near
miss aside): without one, every resolution moving a creature would be a near
miss of every rule about which creature it moved.

- ``subject_in_combat``: true where the event's subject is attacking or
  blocking in the snapshot, false where it is not; a player subject (an
  unblocked attacker's combat damage to the player it attacks, 510.1b) reads
  false when an attacker attacks it and is unreadable otherwise, and no other
  qualifier reads a player;
- ``subject_is_token``: true where the event's subject is a token in the
  snapshot, false where it is a card;
- ``controller_is_owner``: the subject's controller is (true) or is not
  (false) its owner, read off the snapshot's gen-3 ``owner``;
- ``subject_is_source``: the subject is (true) or is not (false) the acting
  ability's source, ``refs.source`` (a Saga sacrificing itself, a permanent
  sacrificed for its own unpaid upkeep);
- ``subject_is_copy``: the subject is (true) or is not (false) a copy of
  another object: it names a ``copy_source``, or a gen-3 ``copy_of`` link
  leaves it;
- ``subject_merged``: the subject is (true) or is not (false) a merged
  permanent, one a gen-3 ``merged_with`` link reaches (a mutated permanent);
- ``subject_face_state``: the subject's gen-3 face state is this one
  (``melded``, ``face_down``, ``flipped``, ``back``, ``front``);
- ``subject_designation``: the subject holds this gen-3 designation
  (``unearthed``, ``suspected``, ``monstrous`` …);
- ``subject_type``: the subject has this card type in the snapshot (an
  attached permanent that is a creature);
- ``with_event``, ``without_event``: the same record does, or does not, hold
  an event of this type as well (the sacrifice beside the counters it pays for,
  the Equipment entering with no ``attached`` event): a type, or a map of
  ``event`` and that event's filters (``{"event": "zone_change", "from_zone":
  "battlefield", "to_zone": "graveyard"}``, a sacrifice as a record writes
  one);
- ``source_dies``: the event's ``source`` (a ``damage_dealt`` or
  ``damage_prevented`` event names one, a non-combat one by its ``cause``,
  which for damage is the same object) does (true) or does not (false) move
  from the battlefield to a graveyard in the same record: a creature dying in
  a combat damage step that still dealt its damage there (510.2). Declared
  true, its near miss is narrower than the other qualifiers': a first
  striker's victim, which dies before it can deal its damage. That is a
  creature with power above 0 that such an event deals damage to in the
  first-strike damage step (the snapshot's ``combat_substep``) and that dies
  in the record having dealt none of them. Every other record is neither
  side: one in which no such creature dies (an ordinary unblocked attack), a
  0-power creature dying (it had no damage to deal), and a creature dying in
  the regular step having dealt none there (a first striker that dealt its
  damage in the first-strike step's record).

A qualifier cannot be combined with ``absent``.

Record filters, which narrow both sides to records whose acting source or
targets are of one kind (D4): ``source_type`` (the source card, as the
snapshot shows it, has this card type: a resolving artifact spell, an
instant), ``acting_face`` (``front`` or ``alternative``: an acting key on
its card's first face or another, an Adventure or a back face) and
``target_face_state`` (a target of the acting ability is in this gen-3 face
state). Like ``api`` they read the acting line's record, so an ``absent``
declaration still needs ``api``, ``keyword`` or ``source_keyword``.

A record kind lists only some event types (``emittable_events``): a
declaration asking its kind for one it never lists is set aside by the index
as untaggable. On a degraded worker's record a missing event only the engine
hooks build may have happened, so neither an ``absent`` declaration's
``fires`` side nor a scoped near miss is read there; a forced resolution's
fork forwards the hook events, so its missing event reads as real.

Entities. The condition entities are the acting ability's source, its
targets and its controller (``actor_player``): what the rule's condition
reads. The outcome entities are where the matching events' outcomes are
placed (``effect_targets.placed_subjects``: an event's subjects, or for a
spell copy, a permanent copy, a retarget or a venture, which Forge names no
subject for, the object it happened to), and for a card a ``zone_change``
moves out of a hand or a library, its owner too, whose cards discarded,
drawn or milled and library events count the move. A tag whose outcome is an
absence — an ``absent`` declaration's ``fires`` side, a scoped near miss
holding no such event — names as its outcome entities where the outcome
would have fallen: the acting clause's targets as the effect half's refs
block holds them, else its controller (``actor_player``), so a rule test
reads a per-entity output on both polarities. A qualified near miss whose
missing events place nothing names the same.

Prior art: ``event_schema.EVENT_PARAMS``, the per-type parameter table every
reader of events already trusts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from effects.domain.effect_targets import placed_subjects
from effects.domain.event_schema import EVENT_PARAMS, Event, EventType
from effects.domain.provenance import SidecarLine, SidecarLookup
from effects.domain.records import EffectRecord, Moment, RecordKind, ResolutionOutcome
from effects.domain.rule_catalogue import Polarity
from effects.domain.rule_detectors import FamilyMatch, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import (
    acting_line,
    apis_of,
    card_has_keyword,
    is_keyword_line,
    root_params_of,
    script_params_match,
    source_entities,
    source_ids,
    source_is_token_of_kind,
)
from effects.domain.rule_detectors.emittable_events import absence_readable
from effects.domain.rule_detectors.entity_facts import (
    has_card_type,
    has_designation,
    has_face_state,
    is_copy,
    is_merged,
)
from effects.domain.script_text import api_of, segments
from effects.domain.script_text import params as params_of
from effects.domain.state_snapshot import (
    CombatSubstep,
    EntityState,
    StateSnapshot,
    normalize_counter_type,
)
from effects.domain.wire_closed_lists import DAMAGE_TYPES, GAME_LOSS_REASONS

__all__ = ["EVENT_FILTERS", "FAMILY", "SUBJECT_QUALIFIERS", "SUBJECT_QUALIFIER_NEEDS",
           "acting_api", "declared_events", "event_filters", "source_is_token_of_kind",
           "subject_verdict",
           "token_kind_of"]

#: Event parameters a declaration may not filter on: they name a card or an
#: object of one game (``cause``, ``card_name``, ``controller``), never a
#: condition of a rule, or hold a structure no equality reads.
_UNFILTERABLE = frozenset({
    "cause", "source", "card_name", "token_script_id", "copy_source", "characteristics",
    "controller", "owner", "attached_to", "detached_from", "targets", "blockers", "blocked",
    "defender", "called", "options", "piles", "mana_by_color", "name", "mana_cost"})
EVENT_FILTERS = frozenset(
    name for names in EVENT_PARAMS.values() for name in names) - _UNFILTERABLE


#: The checks an event filter's value passes (``declared_values``).
EVENT_FILTER_CHECKS = {
    "from_zone": dv.zone, "to_zone": dv.zone, "counter_type": dv.counter_type,
    "to_state": dv.face_state, "damage_type": dv.one_of(*DAMAGE_TYPES),
    "reason": dv.one_of(*GAME_LOSS_REASONS),
}
#: The checks the subject qualifiers' values pass.
QUALIFIER_CHECKS = {
    "subject_in_combat": dv.boolean, "subject_is_token": dv.boolean,
    "controller_is_owner": dv.boolean, "subject_is_source": dv.boolean,
    "subject_is_copy": dv.boolean, "subject_merged": dv.boolean,
    "subject_face_state": dv.face_state, "subject_designation": dv.designation,
    "subject_type": dv.card_type,
}
_CHECKS = {
    **EVENT_FILTER_CHECKS, **QUALIFIER_CHECKS,
    "api": dv.each(dv.api), "keyword": dv.keyword, "source_keyword": dv.keyword,
    "source_token_kind": dv.token_kind, "token_kind": dv.token_kind, "min_count": dv.integer,
    "absent": dv.boolean, "api_in_chain": dv.boolean, "script_param": dv.script_param,
    "chain_param": dv.script_param, "source_type": dv.card_type,
    "target_face_state": dv.face_state, "source_dies": dv.boolean,
}


def _flag(read):
    """A yes-or-no qualifier: the fact ``read`` returns, compared with the
    declared true or false."""
    def verdict(entity: EntityState, state: StateSnapshot, wanted: Any) -> bool | None:
        found = read(entity, state)
        return None if found is None else found == bool(wanted)
    return verdict


def _in_combat(entity: EntityState, state: StateSnapshot) -> bool | None:
    combat = entity.combat
    return combat is not None and bool(combat.attacking or combat.blocking)


def _is_token(entity: EntityState, state: StateSnapshot) -> bool | None:
    return entity.token_script_id is not None


def _controller_is_owner(entity: EntityState, state: StateSnapshot) -> bool | None:
    # None where the snapshot records no owner.
    return None if entity.owner is None else entity.owner == entity.controller


def _is_source(entity: EntityState, state: StateSnapshot) -> bool | None:
    return None if state.refs.source is None else entity.id == state.refs.source


#: Qualifiers read off an event's subject in the snapshot, each the subject's
#: verdict for the declared value; None is unreadable.
SUBJECT_QUALIFIERS = {
    "subject_in_combat": _flag(_in_combat),
    "subject_is_token": _flag(_is_token),
    "controller_is_owner": _flag(_controller_is_owner),
    "subject_is_source": _flag(_is_source),
    "subject_is_copy": _flag(is_copy),
    "subject_merged": _flag(is_merged),
    "subject_face_state": lambda entity, state, wanted: has_face_state(entity, wanted),
    "subject_designation": lambda entity, state, wanted: has_designation(entity, wanted),
    "subject_type": lambda entity, state, wanted: has_card_type(entity, wanted),
}
#: The gen-3 field each subject qualifier reads, where it reads one.
SUBJECT_QUALIFIER_NEEDS = {
    "controller_is_owner": "owner",
    "subject_merged": "links",
    "subject_face_state": "face_state",
    "subject_designation": "designations",
}
_RECORD_QUALIFIERS = frozenset({"with_event", "without_event"})
#: Qualifiers read off the event itself against the record's other events.
_EVENT_QUALIFIERS = frozenset({"source_dies"})
#: The parameters that read the acting line or its source.
_LINE_PARAMS = frozenset({"api", "keyword", "source_keyword", "script_param", "chain_param",
                          "source_token_kind", "source_type", "acting_face",
                          "target_face_state"})
_QUALIFIERS = frozenset(SUBJECT_QUALIFIERS) | _RECORD_QUALIFIERS | _EVENT_QUALIFIERS
#: Record filters read off the acting source and its targets (D4).
_SOURCE_FILTERS = frozenset({"source_type", "acting_face", "target_face_state",
                             "half_outcome"})
_OWN = frozenset({"event", "min_count", "api", "api_in_chain", "keyword", "source_keyword",
                  "script_param", "chain_param", "absent", "token_kind",
                  "source_token_kind"}) | _QUALIFIERS | _SOURCE_FILTERS
#: The record kinds whose payload lists the events a rule's outcome is.
_EVENT_KINDS = frozenset({RecordKind.RESOLUTION, RecordKind.COMBAT, RecordKind.TURN_ACTION})


_GAME_END_EVENTS = frozenset({EventType.PLAYER_WON, EventType.PLAYER_LOST})


def _reads_record(record: EffectRecord, event: EventType) -> bool:
    """Whether a declaration of ``event`` reads ``record``: an effect half, a
    combat or a turn action; the game-end record only for a game-end event,
    since the events the collector rescued into it are no resolution's."""
    if record.kind not in _EVENT_KINDS:
        return False
    if record.kind is RecordKind.RESOLUTION and record.moment is not Moment.RESOLUTION:
        return False
    return event in _GAME_END_EVENTS or not record.is_game_end_record


def acting_api(record: EffectRecord, lines: SidecarLookup | None) -> str | None:
    """The acting line's script API type, or None where no line is readable."""
    line = acting_line(record, lines)
    return line.script_api_type if line is not None else None


def _acts_through(line: SidecarLine | None, api: str | list[str], in_chain: bool) -> bool:
    """Whether ``line``'s root API is ``api`` (or one of a list), or under
    ``in_chain`` any segment's of its chain."""
    if line is None:
        return False
    wanted = apis_of(api)
    if line.script_api_type in wanted:
        return True
    return in_chain and bool(line.script_text) and any(
        api_of(params_of(segment)) in wanted for segment in segments(line.script_text))


def _chain_holds(line: SidecarLine | None, wanted: Any) -> bool:
    """Whether a segment of ``line``'s chain holds the parameter(s) ``wanted``."""
    if line is None or not line.script_text:
        return False
    return any(script_params_match(params_of(segment), wanted)
               for segment in segments(line.script_text))


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


def declared_events(params: Mapping[str, Any]) -> frozenset[EventType]:
    """Every event type a declaration reads: its ``event`` and the types its
    ``with_event``/``without_event`` name."""
    found = {EventType(params["event"])}
    for name in _RECORD_QUALIFIERS & set(params):
        found.add(_event_spec(params[name])[0])
    return frozenset(found)


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


def subject_verdict(subjects: tuple[str, ...], state: StateSnapshot,
                    params: Mapping[str, Any]) -> bool | None:
    """Whether one of ``subjects`` meets every subject qualifier of
    ``params``; None where none of them can be read."""
    wanted = {name: params[name] for name in SUBJECT_QUALIFIERS if name in params}
    if not wanted:
        return True
    verdicts = []
    for subject in subjects:
        found = _subject_reads(subject, state, wanted)
        if found is not None and None not in found:
            verdicts.append(all(found))
    return any(verdicts) if verdicts else None


def _subject_reads(subject: str, state: StateSnapshot,
                   wanted: Mapping[str, Any]) -> list[bool | None] | None:
    """One subject's verdict per qualifier in ``wanted``; None where the
    subject is no snapshot object. A player is read for ``subject_in_combat``
    alone (:func:`_player_in_combat`); every other qualifier is unreadable."""
    entity = state.entity(subject)
    if entity is not None:
        return [SUBJECT_QUALIFIERS[name](entity, state, value) for name, value in wanted.items()]
    if not any(player.id == subject for player in state.players):
        return None
    return [_player_in_combat(subject, state, value) if name == "subject_in_combat" else None
            for name, value in wanted.items()]


def _player_in_combat(player: str, state: StateSnapshot, wanted: Any) -> bool | None:
    """``subject_in_combat`` of a player: false for the defending player, whom
    an attacker attacks (510.1b: an unblocked attacker's damage to it), since a
    player is no creature in combat; unreadable for a player no attacker attacks."""
    attacked = any(e.combat is not None and e.combat.attacking == player
                   for e in state.entities)
    return (not bool(wanted)) if attacked else None


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    id: str = "outcome_event"
    kinds: frozenset[str] = frozenset({"combat", "resolution", "turn_action"})
    params: frozenset[str] = _OWN | EVENT_FILTERS
    general: frozenset[str] = frozenset(
        {"event", "api", "api_in_chain", "keyword", "source_keyword", "absent"})

    def needs(self, params: Mapping[str, Any]) -> frozenset[str]:
        found = {field for name, field in SUBJECT_QUALIFIER_NEEDS.items() if name in params}
        if "target_face_state" in params:
            found.add("face_state")
        return frozenset(found)

    def events(self, params: Mapping[str, Any]) -> frozenset[EventType]:
        return declared_events(params)

    def reads_acting_line(self, params: Mapping[str, Any]) -> bool:
        """Whether the declaration only speaks for records acting through a line."""
        return bool(_LINE_PARAMS & set(params))

    def _filters(self, event: EventType, params: Mapping[str, Any]) -> dict[str, Any]:
        filters = event_filters(self.id, event, params, _OWN)
        if "token_kind" in params and event is not EventType.TOKEN_CREATED:
            raise ValueError("outcome_event: token_kind reads a token_created event")
        for name in _RECORD_QUALIFIERS & set(params):
            _event_spec(params[name])  # an unknown type or filter raises
        if "source_dies" in params and "source" not in EVENT_PARAMS.get(event, ()):
            raise ValueError(f"outcome_event: source_dies reads an event naming a source, "
                             f"which {event.value} does not")
        return filters

    def validate(self, params: Mapping[str, Any]) -> None:
        self._filters(EventType(params["event"]), params)
        dv.check(self.id, params, _CHECKS)
        scoped = any(params.get(name) is not None
                     for name in ("api", "keyword", "source_keyword"))
        absent = bool(params.get("absent"))
        if absent and not scoped:
            raise ValueError("outcome_event: absent needs api, keyword or source_keyword, "
                             "or every resolution fires")
        if absent and _QUALIFIERS & set(params):
            raise ValueError("outcome_event: a qualifier picks among events that happened, "
                             "which absent says did not")
        if params.get("acting_face", "front") not in ("front", "alternative"):
            raise ValueError("outcome_event: acting_face is front or alternative")
        if "half_outcome" in params:
            ResolutionOutcome(params["half_outcome"])

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        event = EventType(params["event"])
        filters = self._filters(event, params)
        api, keyword = params.get("api"), params.get("keyword")
        source_keyword = params.get("source_keyword")
        scoped = api is not None or keyword is not None or source_keyword is not None
        absent = bool(params.get("absent"))
        qualified = bool(_QUALIFIERS & set(params))
        if not _reads_record(record, event):
            return None
        if scoped or "script_param" in params or "chain_param" in params:
            line = acting_line(record, lines)
            by_api = api is not None and _acts_through(line, api, bool(params.get("api_in_chain")))
            by_keyword = keyword is not None and is_keyword_line(line, keyword)
            if (api is not None or keyword is not None) and not (by_api or by_keyword):
                return None
            if "script_param" in params and not script_params_match(
                    root_params_of(line), params["script_param"]):
                return None
            if "chain_param" in params and not _chain_holds(line, params["chain_param"]):
                return None
        if source_keyword is not None and not card_has_keyword(
                record.ability or (), lines, source_keyword):
            return None
        kind = params.get("source_token_kind")
        if kind is not None and not source_is_token_of_kind(record.state, kind, record.ability):
            return None
        if not _source_fits(record, params):
            return None
        token_kind = params.get("token_kind")
        listed = record.payload.listed_events()
        wanted = [
            e for e in listed
            if e.type is event
            and meets_filters(e, filters)
            and (token_kind is None or token_kind.lower() in token_kind_of(e))]
        # A hook-derived event a hookless record lacks may have happened: no
        # side that reads its absence (V5 #15).
        absence = all(absence_readable(record, e) for e in declared_events(params))
        if qualified:
            return self._qualified(record, params, wanted, listed, scoped, absence)
        happened = len(wanted) >= int(params.get("min_count", 1))
        condition = _condition(record)
        outcome = _outcome(record, wanted)
        if absent:
            # The rule's outcome is the event's absence, so that is what is observed.
            if happened:
                return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition, outcome)
            return FamilyMatch(Polarity.FIRES, True, condition,
                               _would_fall_on(record)) if absence else None
        if happened:
            return FamilyMatch(Polarity.FIRES, True, condition, outcome)
        if not scoped or not absence:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition, _would_fall_on(record))

    @staticmethod
    def _qualified(record: EffectRecord, params: Mapping[str, Any], wanted: list[Event],
                   listed: tuple[Event, ...], scoped: bool,
                   absence: bool = True) -> FamilyMatch | None:
        """The rule's side among the events that happened, by its qualifiers."""
        record_ok = all((
            "with_event" not in params or _holds(listed, params["with_event"]),
            "without_event" not in params or not _holds(listed, params["without_event"]),
        ))
        read = [(e, _both(subject_verdict(e.subjects, record.state, params),
                          _source_verdict(e, listed, params))) for e in wanted]
        read = [(e, verdict) for e, verdict in read if verdict is not None]
        meeting = [e for e, verdict in read if verdict and record_ok]
        condition = _condition(record)
        if len(meeting) >= int(params.get("min_count", 1)):
            return FamilyMatch(Polarity.FIRES, True, condition, _outcome(record, meeting))
        if not absence:
            return None
        if params.get("source_dies") is True:
            # The contrast 510.2 draws: a first striker's victim, dying before
            # it could deal its damage, not every step whose damage dealers
            # all survive.
            victims = _victims(record, wanted, listed)
            return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition,
                               victims) if victims else None
        if not scoped:
            return None
        return FamilyMatch(Polarity.DOES_NOT_FIRE, False, condition,
                           _near_outcome(record, [e for e, _ in read if e not in meeting]))


def _both(first: bool | None, second: bool | None) -> bool | None:
    """Two verdicts together; None where either cannot be read."""
    return None if first is None or second is None else first and second


def _source_of(event: Event) -> str | None:
    """A damage event's ``source``; a damage event outside combat writes no
    ``source``, only the ``cause``, the same object for damage."""
    found = event.params or {}
    return found.get("source") or found.get("cause") or None


def _dies(entity_id: str, listed: tuple[Event, ...]) -> bool:
    """Whether the record moves ``entity_id`` from the battlefield to a graveyard."""
    return any(e.type is EventType.ZONE_CHANGE and entity_id in e.subjects
               and same_value((e.params or {}).get("from_zone"), "battlefield")
               and same_value((e.params or {}).get("to_zone"), "graveyard")
               for e in listed)


def _source_verdict(event: Event, listed: tuple[Event, ...],
                    params: Mapping[str, Any]) -> bool | None:
    """Whether ``event``'s ``source`` dies in the record as ``source_dies``
    says (True when it is not declared); None where the event names none."""
    if "source_dies" not in params:
        return True
    source = _source_of(event)
    if source is None:
        return None
    return _dies(source, listed) == bool(params["source_dies"])


def _victims(record: EffectRecord, wanted: list[Event],
             listed: tuple[Event, ...]) -> tuple[str, ...]:
    """A first striker's victims: in the first-strike damage step, the
    creatures with power above 0 that ``wanted``'s events deal damage to and
    that die in the record having dealt none of those events themselves.
    None in any other step, or where the snapshot names no step."""
    if record.state.global_.combat_substep != CombatSubstep.FIRST_STRIKE:
        return ()
    sources = {_source_of(e) for e in wanted}
    found = []
    for event in wanted:
        for subject in event.subjects:
            entity = record.state.entity(subject)
            if (subject not in sources and entity is not None
                    and has_card_type(entity, "creature") and entity.power > 0
                    and _dies(subject, listed)):
                found.append(subject)
    return tuple(dict.fromkeys(found))


def _source_fits(record: EffectRecord, params: Mapping[str, Any]) -> bool:
    """The record filters read off the acting source and its targets:
    ``source_type`` (the source card, as the snapshot shows it, has this card
    type), ``acting_face`` (``front`` for an acting key on its card's first
    face, ``alternative`` for another: an Adventure, a back face, a split
    half), ``target_face_state`` (a target of the acting ability is in
    this gen-3 face state) and ``half_outcome`` (the effect half's own outcome:
    a fizzled or declined spell's half holds its card's move off the stack).
    A record none can be read on fits none."""
    if "source_type" in params:
        wanted = params["source_type"].lower()
        if not any(wanted in {t.lower() for t in e.types}
                   for e in source_entities(record.state, record.ability or ())):
            return False
    if "acting_face" in params:
        keys = record.ability or ()
        if not keys or (keys[0].face != 0) != (params["acting_face"] == "alternative"):
            return False
    if "half_outcome" in params and getattr(record.payload, "outcome", None) != (
            ResolutionOutcome(params["half_outcome"])):
        return False
    if "target_face_state" in params:
        targets = [record.state.entity(t) for t in record.state.refs.as_cast]
        if not any(t is not None and has_face_state(t, params["target_face_state"])
                   for t in targets):
            return False
    return True


def _condition(record: EffectRecord) -> tuple[str, ...]:
    """The acting ability's source, its targets and its controller."""
    actor = (record.actor_player,) if record.actor_player else ()
    return tuple(dict.fromkeys((*source_ids(record), *record.state.refs.as_cast, *actor)))


def _would_fall_on(record: EffectRecord) -> tuple[str, ...]:
    """Where an outcome that did not happen would have fallen: the acting
    clause's targets as the effect half's refs block holds them (those left
    after the illegal-target check, so a target gone since names no slot),
    else its controller."""
    targets = record.state.refs.targets
    if targets:
        return tuple(dict.fromkeys(targets))
    return (record.actor_player,) if record.actor_player else ()


def _near_outcome(record: EffectRecord, events) -> tuple[str, ...]:
    """A qualified near miss's outcome entities: where the events that
    missed the qualifiers fell, else where the outcome would have fallen."""
    return _outcome(record, events) or _would_fall_on(record)


_OWNED_ZONES = frozenset({"hand", "library"})


def _outcome(record: EffectRecord, events) -> tuple[str, ...]:
    """Where ``events``' outcomes are placed, and the owner of a card a
    ``zone_change`` moves out of a hand or a library."""
    found: list[str] = []
    for event in events:
        placed = placed_subjects(record, event)
        found += placed
        if (event.type is EventType.ZONE_CHANGE
                and str(event.params.get("from_zone", "")).lower() in _OWNED_ZONES):
            for subject in placed:
                entity = record.state.entity(subject)
                if entity is not None and (entity.owner or entity.controller):
                    found.append(entity.owner or entity.controller)
    return tuple(dict.fromkeys(found))


FAMILY = register(OutcomeEvent())
