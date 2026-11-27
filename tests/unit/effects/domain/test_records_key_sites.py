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
from tests.unit.effects.record_builders import lines_of, make_entity


def _key(name: str, kind: str = "static") -> ProvenanceKey:
    return ProvenanceKey(f"cardsfolder/{name[0]}/{name}.txt", 0, kind, 0)


PRINTED, ATTACHED, TEMPORARY = _key("printed"), _key("attached"), _key("temporary")
EMBLEM = _key("emblem")


def _board(snapshot):
    entity = make_entity(
        "E1", "Bear", lines=lines_of((PRINTED,), (ATTACHED,)),
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


def test_a_gen3_entity_names_every_line_s_key_in_force_or_not(make_record, snapshot, ability_key):
    """A face not up and a granted line join their sidecars too."""
    from effects.domain.state_snapshot import CardLine

    own, other_face, granted = _key("printed", "trigger"), _key("back"), _key("anthem")
    entity = make_entity("E1", "Bear", lines=(
        CardLine(keys=(own,), kind="trigger", granted=False),
        CardLine(keys=(other_face,), kind="static", granted=False, active=False),
        CardLine(keys=(granted,), kind="keyword", granted=True, fragment="Flying"),
        CardLine(keys=(), kind="keyword", granted=True, fragment="Haste"),
    ), granted_temporary=GrantedTemporary(abilities=(TEMPORARY,)))
    record = make_record(state=replace(snapshot, entities=(entity,)))
    assert list(key_sites(record)) == [ability_key, own, other_face, granted, TEMPORARY]


def test_a_golden_record_names_its_floating_effects_and_link_ends():
    """The collector's own boards: every floating effect's source line and
    every line a link ends at is among the record's key sites, so the sidecar
    join and the holdout split both see them."""
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import java_golden_records

    floating = ends = 0
    for record in map(record_from_dict, java_golden_records()):
        # Without its entities, whose lines name most of the same keys.
        sites = set(key_sites(replace(record, state=replace(record.state, entities=()))))
        for made in record.state.global_.floating_effects:
            floating += len(made.source_lines)
            assert set(made.source_lines) <= sites, record.record_id
        for link in record.state.links or ():
            ends += len(link.line_keys())
            assert set(link.line_keys()) <= sites, record.record_id
    assert floating and ends


def test_a_delayed_trigger_names_its_source_line(make_record, snapshot):
    """No golden holds a pending delayed trigger yet."""
    from effects.domain.state_snapshot import CommandZoneObject

    source = _key("astral_slide", "trigger")
    board = replace(snapshot, global_=replace(snapshot.global_, delayed_triggers=(
        CommandZoneObject(id="D7", source_lines=(source,)),)))
    assert source in set(key_sites(make_record(state=board)))
