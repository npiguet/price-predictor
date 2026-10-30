"""``key_sites``: every printed-line key a record names, emblems aside."""

from __future__ import annotations

from dataclasses import replace

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    Candidate,
    ForbiddenEntity,
    PlayabilityBlockersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    RewritePayload,
    key_sites,
)
from effects.domain.state_snapshot import GrantedTemporary
from tests.unit.effects.builders import make_entity


def _key(name: str, kind: str = "static") -> ProvenanceKey:
    return ProvenanceKey(f"cardsfolder/{name[0]}/{name}.txt", 0, kind, 0)


PRINTED, ATTACHED, TEMPORARY = _key("printed"), _key("attached"), _key("temporary")
EMBLEM = _key("emblem")


def _board(snapshot):
    entity = make_entity(
        "E1", "Bear", printed=(PRINTED,), granted_attached=(ATTACHED,),
        granted_temporary=GrantedTemporary(abilities=(TEMPORARY,)),
    )
    return replace(
        snapshot, entities=(entity,),
        global_=replace(snapshot.global_, emblems=(EMBLEM,)),
    )


def test_the_acting_line_comes_first_then_each_entitys_keys(
    make_record, snapshot, ability_key,
):
    record = make_record(state=_board(snapshot))
    assert list(key_sites(record)) == [ability_key, PRINTED, ATTACHED, TEMPORARY]


def test_a_rewrite_names_the_ability_that_ran_instead(make_record, snapshot):
    ran = _key("replacer", "replacement")
    record = make_record(
        kind=RecordKind.REWRITE, state=_board(snapshot), ability=(_key("shield"),),
        payload=RewritePayload(incoming=Event(type=EventType.DAMAGE_DEALT), replaced_by=(ran,)),
    )
    assert list(key_sites(record))[-1] == ran


def test_a_decision_names_its_candidates_and_their_statics(make_record, snapshot):
    candidate, static = _key("candidate", "spell"), _key("tax")
    record = make_record(
        kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.DECISION, state=snapshot,
        payload=PlayabilityDecisionPayload(candidates=(Candidate(
            ability=(candidate,), can_play=False, affordable=False,
            has_legal_target=True, responsible_static=(static,),
        ),)),
    )
    assert list(key_sites(record)) == [candidate, static]


def test_a_legality_record_names_its_forbidding_statics(make_record, snapshot):
    static = _key("propaganda")
    record = make_record(
        kind=RecordKind.PLAYABILITY, subkind=PlayabilitySubkind.BLOCKERS, state=snapshot,
        payload=PlayabilityBlockersPayload(
            anchor_attacker="E9",
            forbidden=(ForbiddenEntity(entity="E1", responsible_static=(static,)),),
        ),
    )
    assert list(key_sites(record)) == [static]


def test_emblems_are_never_named(make_record, snapshot):
    assert EMBLEM not in set(key_sites(make_record(state=_board(snapshot))))
