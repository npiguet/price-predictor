"""``rule_family`` over real gen-1 records and their sidecars (T071, FR-047–048).

Every record here is cut from a real shard (``scripts/make_fixture_records.py``);
the few tests that need a shape the sample lacks — a keyword on a combat
participant, a static named by a candidate — edit a copy of a real record
rather than build one by hand.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from effects.domain.provenance import ProvenanceKey, SidecarLine
from effects.domain.records import Candidate, ForbiddenEntity, RecordKind
from effects.domain.rule_families import (
    CANNOT_PLAY,
    NO_LEGAL_TARGET,
    NO_MODE,
    NONE,
    UNAFFORDABLE,
    candidate_family,
    legality_modes,
    root_param,
    rule_family,
)
from effects.domain.state_snapshot import GrantedTemporary
from effects.infrastructure.record_io import read_shard
from effects.infrastructure.sidecar_io import SidecarCache

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "effects"


@pytest.fixture(scope="module")
def records():
    return list(read_shard(_FIXTURES / "gen1-records.jsonl.gz"))


@pytest.fixture(scope="module")
def sidecars():
    root = _FIXTURES / "gen1-sidecars"
    return SidecarCache({
        "cardsfolder": root / "cardsfolder", "tokenscripts": root / "tokenscripts",
    })


def _first(records, kind, discriminator=None, test=lambda r: True):
    for record in records:
        found = (record.moment or record.subkind)
        if record.kind is kind and (
            discriminator is None or (found is not None and found.value == discriminator)
        ) and test(record):
            return record
    raise AssertionError(f"fixture holds no {kind} {discriminator}")


def test_a_resolution_takes_its_acting_lines_api_type(records, sidecars):
    record = _first(records, RecordKind.RESOLUTION, "resolution",
                    lambda r: r.ability and r.ability[0].script_file.endswith("tcri_building.txt"))
    assert rule_family(record, sidecars) == "GainLife"


def test_both_halves_of_a_resolution_share_its_family(records, sidecars):
    halves = [r for r in records if r.link_id and r.ability
              and r.ability[0].script_file.endswith("crustacean_commando.txt")]
    assert {r.moment.value for r in halves} == {"activation", "resolution"}
    assert {rule_family(r, sidecars) for r in halves} == {"Token"}


def test_a_charm_resolution_is_the_charm_family(records, sidecars):
    record = _first(records, RecordKind.RESOLUTION, "resolution",
                    lambda r: r.ability and "you_come_to_a_river" in r.ability[0].script_file)
    assert rule_family(record, sidecars) == "Charm"


def test_a_trigger_takes_its_root_segments_mode(records, sidecars):
    record = _first(records, RecordKind.TRIGGER)
    assert rule_family(record, sidecars) == "ChangesZone"


def test_a_trigger_with_no_mode_is_no_mode(records):
    record = _first(records, RecordKind.TRIGGER)
    line = SidecarLine(line_index=0, line_kind="triggered", provenance=record.ability,
                       script_api_type="Draw", script_text="Execute$ SV1 | TriggerZones$ Hand")
    assert rule_family(record, _Lines({record.ability[0]: line})) == NO_MODE


def test_a_continuous_record_takes_its_static_mode(records, sidecars):
    record = _first(records, RecordKind.CONTINUOUS,
                    test=lambda r: "bramblewood_paragon" in r.ability[0].script_file)
    assert rule_family(record, sidecars) == "Continuous"


def test_a_multi_mode_static_keeps_its_mode_list_whole(records):
    record = _first(records, RecordKind.CONTINUOUS)
    line = SidecarLine(line_index=0, line_kind="static", provenance=record.ability,
                       script_api_type="CantBlockBy,CantAttack",
                       script_text="Mode$ CantBlockBy,CantAttack | ValidCard$ Creature")
    assert rule_family(record, _Lines({record.ability[0]: line})) == "CantBlockBy,CantAttack"


def test_a_rewrite_takes_its_event(records, sidecars):
    record = _first(records, RecordKind.REWRITE,
                    test=lambda r: "bramblewood_paragon" in r.ability[0].script_file)
    assert rule_family(record, sidecars) == "ETBReplacement"


def test_a_combat_without_damage_step_keywords_is_none(records, sidecars):
    record = _first(records, RecordKind.COMBAT, test=lambda r: not r.fork)
    assert rule_family(record, sidecars) == NONE


def test_a_combat_family_is_the_sorted_keyword_set(records, sidecars):
    record = _first(records, RecordKind.COMBAT, test=lambda r: not r.fork)
    entities = tuple(
        dataclasses.replace(entity, granted_temporary=GrantedTemporary(
            keywords=("trample", "first_strike"),
        )) if entity.combat is not None else entity
        for entity in record.state.entities
    )
    edited = dataclasses.replace(record, state=dataclasses.replace(record.state, entities=entities))
    assert rule_family(edited, sidecars) == "first_strike,trample"


def test_a_keyword_acting_line_is_its_keywords_family(records):
    """Spec Story 4 scenario 7 / FR-048: whatever the kind."""
    record = _first(records, RecordKind.TRIGGER)
    line = SidecarLine(line_index=0, line_kind="static", provenance=record.ability,
                       script_api_type="Keyword", script_text="Ward:2")
    assert rule_family(record, _Lines({record.ability[0]: line})) == "Ward"


def test_a_decision_with_an_unaffordable_candidate_is_unaffordable(records, sidecars):
    record = _first(records, RecordKind.PLAYABILITY, "decision")
    assert not record.payload.candidates[0].affordable
    assert rule_family(record, sidecars) == UNAFFORDABLE


def test_decision_candidates_by_verdict_and_by_static(records):
    """Spec Story 4 scenario 11: a static's mode, else the first failing bit."""
    static = ProvenanceKey("cardsfolder/x/curse.txt", 0, "static", 0)
    lines = _Lines({static: SidecarLine(
        line_index=0, line_kind="static", provenance=(static,),
        script_api_type="CantBeCast", script_text="Mode$ CantBeCast | ValidCard$ Card",
    )})
    unaffordable = Candidate(ability=(), can_play=True, affordable=False, has_legal_target=True)
    forbidden = Candidate(ability=(), can_play=False, affordable=True, has_legal_target=True,
                          responsible_static=(static,))
    assert candidate_family(unaffordable, lines) == UNAFFORDABLE
    assert candidate_family(forbidden, lines) == "CantBeCast"
    assert candidate_family(dataclasses.replace(forbidden, responsible_static=()), lines) == (
        CANNOT_PLAY
    )
    assert candidate_family(Candidate(
        ability=(), can_play=True, affordable=True, has_legal_target=False,
    ), lines) == NO_LEGAL_TARGET
    assert candidate_family(Candidate(
        ability=(), can_play=True, affordable=True, has_legal_target=True,
    ), lines) == NONE


def test_a_blockers_record_restricted_by_flying_is_flying(records, sidecars):
    record = _first(records, RecordKind.PLAYABILITY, "blockers",
                    test=lambda r: any(f.responsible_static for f in r.payload.forbidden))
    assert rule_family(record, sidecars) == "Flying"


def _restricted(records, *statics):
    """A real blockers record whose forbidden creatures name ``statics``' keys."""
    keys = [ProvenanceKey(f"cardsfolder/x/s{i}.txt", 0, "static", 0)
            for i in range(len(statics))]
    lines = _Lines({
        key: SidecarLine(line_index=0, line_kind="static", provenance=(key,),
                         script_api_type=mode, script_text=f"Mode$ {mode}")
        for key, mode in zip(keys, statics)
    })
    record = _first(records, RecordKind.PLAYABILITY, "blockers")
    payload = dataclasses.replace(record.payload, forbidden=tuple(
        ForbiddenEntity(entity=f"E{i}", responsible_static=(key,))
        for i, key in enumerate(keys)
    ))
    return dataclasses.replace(record, payload=payload), lines


def test_a_legality_record_is_in_its_rarest_modes_family(records):
    record, lines = _restricted(records, "CantBlockBy", "CantBlock")
    counts = {"CantBlockBy": 3, "CantBlock": 900}
    assert rule_family(record, lines, legality_mode_counts=counts) == "CantBlockBy"


def test_a_multi_mode_static_counts_each_of_its_modes(records):
    """Pacifism declares ``Mode$ CantAttack,CantBlock`` in one field; it
    restricts two rules, not one rule named after both."""
    record, lines = _restricted(records, "CantAttack,CantBlock", "CantAttack")
    assert legality_modes(record, lines) == frozenset({"CantAttack", "CantBlock"})
    counts = {"CantAttack": 40, "CantBlock": 7}
    assert rule_family(record, lines, legality_mode_counts=counts) == "CantBlock"


def test_without_mode_counts_the_family_is_still_one_mode(records):
    """A dataset built before the counts existed still gets a single-rule
    family, the first mode by name, rather than the combination."""
    record, lines = _restricted(records, "CantBlockBy", "CantBlock")
    assert rule_family(record, lines) == "CantBlock"


def test_a_legality_record_with_no_static_is_none(records, sidecars):
    record = _first(records, RecordKind.PLAYABILITY, "attackers")
    assert rule_family(record, sidecars) == NONE


def test_every_fixture_record_has_a_family(records, sidecars):
    assert all(rule_family(record, sidecars) for record in records)


def test_root_param_reads_the_root_segment_only():
    text = "Mode$ ChangesZone | Execute$ SV1 [SEG] SV1: DB$ Draw | Mode$ Other"
    assert root_param(text, "Mode") == "ChangesZone"
    assert root_param(text, "Event") is None
    assert root_param(None, "Mode") is None


class _Lines:
    """A minimal sidecar cache: ``line_for`` over a fixed table."""

    def __init__(self, lines):
        self._lines = lines

    def line_for(self, key):
        return self._lines.get(key)
