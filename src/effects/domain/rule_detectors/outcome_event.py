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
- ``candidate`` (``cast`` or ``activated``, as ``acting_line.ability_kind``
  reads the acting key): the acting ability is its card's cast or an
  activated ability. ``source_keyword`` is the card's, so a declaration about
  the cast of a card with the keyword names it, or an Equip activation, a
  granted static or a trigger of the card would be read for the cast; the
  damage rules of wither and infect (any line of the card deals damage) name
  none;
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

from effects.domain.event_schema import EVENT_PARAMS, EventType
from effects.domain.provenance import SidecarLookup
from effects.domain.records import EffectRecord, ResolutionOutcome
from effects.domain.rule_detectors import FamilyMatch, Matcher, register
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.acting_line import acting_line, source_is_token_of_kind
from effects.domain.rule_detectors.outcome_events import (
    EVENT_FILTER_CHECKS,
    EVENT_FILTERS,
    RECORD_QUALIFIERS,
    EventSpec,
    declared_events,
    event_filters,
    token_kind_of,
)
from effects.domain.rule_detectors.outcome_matcher import (
    LINE_PARAMS,
    OWN_PARAMS,
    QUALIFIERS,
    OutcomeMatcher,
    is_scoped,
)
from effects.domain.rule_detectors.outcome_subjects import (
    QUALIFIER_CHECKS,
    SUBJECT_QUALIFIER_NEEDS,
    SUBJECT_QUALIFIERS,
    subject_verdict,
)

__all__ = ["EVENT_FILTERS", "FAMILY", "SUBJECT_QUALIFIERS", "SUBJECT_QUALIFIER_NEEDS",
           "acting_api", "declared_events", "event_filters", "source_is_token_of_kind",
           "subject_verdict",
           "token_kind_of"]

_CHECKS = {
    **EVENT_FILTER_CHECKS, **QUALIFIER_CHECKS,
    "api": dv.each(dv.api), "keyword": dv.keyword, "source_keyword": dv.keyword,
    "candidate": dv.one_of("cast", "activated"),
    "source_token_kind": dv.token_kind, "token_kind": dv.token_kind, "min_count": dv.integer,
    "absent": dv.boolean, "api_in_chain": dv.boolean, "script_param": dv.script_param,
    "chain_param": dv.script_param, "source_type": dv.card_type,
    "target_face_state": dv.face_state, "source_dies": dv.boolean,
}


def acting_api(record: EffectRecord, lines: SidecarLookup | None) -> str | None:
    """The acting line's script API type, or None where no line is readable."""
    line = acting_line(record, lines)
    return line.script_api_type if line is not None else None


def _check_filters(params: Mapping[str, Any]) -> None:
    """The event filters, ``token_kind``, the record qualifiers' events and
    ``source_dies`` against the declared event's schema."""
    event = EventType(params["event"])
    event_filters("outcome_event", event, params, OWN_PARAMS)
    if "token_kind" in params and event is not EventType.TOKEN_CREATED:
        raise ValueError("outcome_event: token_kind reads a token_created event")
    for name in RECORD_QUALIFIERS & set(params):
        EventSpec.of(params[name])  # an unknown type or filter raises
    if "source_dies" in params and "source" not in EVENT_PARAMS.get(event, ()):
        raise ValueError(f"outcome_event: source_dies reads an event naming a source, "
                         f"which {event.value} does not")


def _check_absent(params: Mapping[str, Any]) -> None:
    if not params.get("absent"):
        return
    if not is_scoped(params):
        raise ValueError("outcome_event: absent needs api, keyword or source_keyword, "
                         "or every resolution fires")
    if QUALIFIERS & set(params):
        raise ValueError("outcome_event: a qualifier picks among events that happened, "
                         "which absent says did not")


def _check_record_filters(params: Mapping[str, Any]) -> None:
    if params.get("acting_face", "front") not in ("front", "alternative"):
        raise ValueError("outcome_event: acting_face is front or alternative")
    if "half_outcome" in params:
        ResolutionOutcome(params["half_outcome"])


@dataclass(frozen=True, slots=True)
class OutcomeEvent:
    id: str = "outcome_event"
    kinds: frozenset[str] = frozenset({"combat", "resolution", "turn_action"})
    params: frozenset[str] = OWN_PARAMS | EVENT_FILTERS
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
        return bool(LINE_PARAMS & set(params))

    def validate(self, params: Mapping[str, Any]) -> None:
        _check_filters(params)
        dv.check(self.id, params, _CHECKS)
        _check_absent(params)
        _check_record_filters(params)

    def compile(self, params: Mapping[str, Any]) -> Matcher:
        return OutcomeMatcher.of(params)

    def match(self, record: EffectRecord, lines: SidecarLookup | None,
              params: Mapping[str, Any]) -> FamilyMatch | None:
        return self.compile(params)(record, lines)


FAMILY = register(OutcomeEvent())
