"""``LossyForks`` counts forks by ``fork_kind``: an intervention and a protection
fork share their flags, so only the kind tells them apart.

Both records are the fixture's real interventional effect half, the second with
its ``fork_kind`` edited to ``protection``; both lose something in the copy.
"""

from __future__ import annotations

from effects.application.corpus_validation.envelope_checks import LossyForks
from effects.application.corpus_validation.findings import Thresholds
from effects.domain.records import EffectRecord, RecordOrigin
from effects.infrastructure.record_codec import record_from_dict
from tests.unit.effects.builders import gen3_fixture_records

_NAME = "fork records whose copy left something out"


def _forks() -> tuple[EffectRecord, EffectRecord]:
    raw = next(r for r in gen3_fixture_records() if r.get("fork_kind") == "intervention"
               and r["kind"] == "resolution")
    edited = {**raw, "copy_dropped": {"command": 1}}
    protection = {**edited, "fork_kind": "protection", "record_id": raw["record_id"] + "~p"}
    return record_from_dict(edited), record_from_dict(protection)


def _measured(*records: EffectRecord) -> str:
    lossy = LossyForks()
    for record in records:
        lossy.add(record)
    return next(f for f in lossy.findings(Thresholds()) if f.name == _NAME).measured


def test_an_intervention_and_a_protection_fork_of_one_turn_are_two_groups():
    intervention, protection = _forks()
    assert intervention.origin is RecordOrigin.FORCED
    assert protection.origin is RecordOrigin.PROTECTION_FORK
    assert (intervention.game_id, intervention.state.global_.turn) == (
        protection.game_id, protection.state.global_.turn)
    assert LossyForks.group_of(intervention) != LossyForks.group_of(protection)
    assert "2/2 forks" in _measured(intervention, protection)


def test_the_records_of_one_fork_are_one_group():
    intervention, _ = _forks()
    twin = record_from_dict({**_dict_of(intervention), "record_id": "twin"})
    assert "1/1 forks" in _measured(intervention, twin)


def _dict_of(record: EffectRecord) -> dict:
    return next(r for r in gen3_fixture_records() if r["record_id"] == record.record_id) | {
        "copy_dropped": {"command": 1}}
