"""The subject qualifiers of ``outcome_event``, read off an event's subject in the snapshot.

A qualifier names which events a rule speaks for by their subject: one in
combat, a token, a copy, one whose controller is its owner. Each reads the
subject's verdict for the declared value, None where it cannot be read. A
player is read for ``subject_in_combat`` alone (the defending player of an
unblocked attacker, 510.1b); every other qualifier is unreadable on one.
``rewrite_event`` and ``trigger_mode`` read the same qualifiers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from effects.domain.event_schema import Event, damage_source
from effects.domain.records import CombatPayload, EffectRecord
from effects.domain.rule_detectors import declared_values as dv
from effects.domain.rule_detectors.entity_facts import (
    has_card_type,
    has_designation,
    has_face_state,
    is_copy,
    is_merged,
)
from effects.domain.state_snapshot import EntityState, StateSnapshot

#: A qualifier's verdict on one subject for its declared value; None is unreadable.
Qualifier = Callable[[EntityState, StateSnapshot, Any], "bool | None"]


def _flag(read: Callable[[EntityState, StateSnapshot], bool | None]) -> Qualifier:
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


def _face_state(entity: EntityState, state: StateSnapshot, wanted: Any) -> bool | None:
    return has_face_state(entity, wanted)


def _designation(entity: EntityState, state: StateSnapshot, wanted: Any) -> bool | None:
    return has_designation(entity, wanted)


def _card_type(entity: EntityState, state: StateSnapshot, wanted: Any) -> bool | None:
    return has_card_type(entity, wanted)


#: Qualifiers read off an event's subject in the snapshot, each the subject's
#: verdict for the declared value; None is unreadable.
SUBJECT_QUALIFIERS: dict[str, Qualifier] = {
    "subject_in_combat": _flag(_in_combat),
    "subject_is_token": _flag(_is_token),
    "controller_is_owner": _flag(_controller_is_owner),
    "subject_is_source": _flag(_is_source),
    "subject_is_copy": _flag(is_copy),
    "subject_merged": _flag(is_merged),
    "subject_face_state": _face_state,
    "subject_designation": _designation,
    "subject_type": _card_type,
}
#: The gen-3 field each subject qualifier reads, where it reads one.
SUBJECT_QUALIFIER_NEEDS = {
    "controller_is_owner": "owner",
    "subject_merged": "links",
    "subject_face_state": "face_state",
    "subject_designation": "designations",
}
#: The checks the subject qualifiers' values pass (``declared_values``).
QUALIFIER_CHECKS = {
    "subject_in_combat": dv.boolean, "subject_is_token": dv.boolean,
    "controller_is_owner": dv.boolean, "subject_is_source": dv.boolean,
    "subject_is_copy": dv.boolean, "subject_merged": dv.boolean,
    "subject_face_state": dv.face_state, "subject_designation": dv.designation,
    "subject_type": dv.card_type,
}


def subject_qualifiers(params: Mapping[str, Any]) -> dict[str, Any]:
    """The subject qualifiers ``params`` declares, name -> declared value, in
    :data:`SUBJECT_QUALIFIERS`' order."""
    return {name: params[name] for name in SUBJECT_QUALIFIERS if name in params}


def subjects_meet(subjects: tuple[str, ...], state: StateSnapshot,
                  wanted: Mapping[str, Any]) -> bool | None:
    """Whether one of ``subjects`` meets every qualifier of ``wanted``
    (:func:`subject_qualifiers`); None where none of them can be read."""
    if not wanted:
        return True
    verdicts = [all(found) for found in (_subject_reads(s, state, wanted) for s in subjects)
                if _readable(found)]
    return any(verdicts) if verdicts else None


def subject_verdict(subjects: tuple[str, ...], state: StateSnapshot,
                    params: Mapping[str, Any]) -> bool | None:
    """Whether one of ``subjects`` meets every subject qualifier of
    ``params``; None where none of them can be read."""
    return subjects_meet(subjects, state, subject_qualifiers(params))


def _readable(found: list[bool | None] | None) -> bool:
    return found is not None and None not in found


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
    return [_player_qualifier(name, subject, state, value) for name, value in wanted.items()]


def _player_qualifier(name: str, player: str, state: StateSnapshot,
                      wanted: Any) -> bool | None:
    if name != "subject_in_combat":
        return None
    return _player_in_combat(player, state, wanted)


def _player_in_combat(player: str, state: StateSnapshot, wanted: Any) -> bool | None:
    """``subject_in_combat`` of a player: false for the defending player, whom
    an attacker attacks (510.1b: an unblocked attacker's damage to it), since a
    player is no creature in combat; unreadable for a player no attacker attacks."""
    attacked = any(e.combat is not None and e.combat.attacking == player
                   for e in state.entities)
    return (not bool(wanted)) if attacked else None


def player_damage_unblocked(record: EffectRecord, event: Event,
                            wanted: Mapping[str, Any]) -> bool | None:
    """Whether a combat event's damage to a player is an unblocked attacker's
    (510.1b), the one reading of ``subject_in_combat`` a player has: a blocked
    attacker's (a trampler's excess, 702.19b) is not. True where the qualifier
    is not declared, the record is no combat or the event names no player;
    None where the event names no source."""
    payload = record.payload
    if "subject_in_combat" not in wanted or not isinstance(payload, CombatPayload):
        return True
    if not any(_is_player(subject, record.state) for subject in event.subjects):
        return True
    source = damage_source(event)
    if source is None:
        return None
    return source in payload.attackers and not payload.blocks.get(source)


def _is_player(subject: str, state: StateSnapshot) -> bool:
    return state.entity(subject) is None and any(p.id == subject for p in state.players)
