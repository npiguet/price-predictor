"""Shared record fixtures for the schema suites."""

from __future__ import annotations

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot
from tests.unit.effects import record_builders


@pytest.fixture
def snapshot() -> StateSnapshot:
    return record_builders.make_snapshot(
        global_=GlobalState(
            turn=4, phase="main1", active="P0", priority="P0", stack_size=1,
        ),
        tiers=frozenset(),
    )


@pytest.fixture
def ability_key() -> ProvenanceKey:
    return ProvenanceKey(
        script_file="cardsfolder/l/lightning_bolt.txt",
        face=0,
        trait_kind="spell",
        index_within_kind=0,
    )


@pytest.fixture
def make_record(snapshot, ability_key):
    """:func:`record_builders.make_record` on the ``snapshot`` board, with collected
    ids and timestamp; a resolution acts through ``ability_key`` and its
    effect half deals damage."""

    def build(kind: RecordKind = RecordKind.RESOLUTION, **overrides) -> EffectRecord:
        defaults: dict = {
            "record_id": "run.3.11",
            "timestamp": "2026-09-06T15:31:12.152018Z",
            "game_id": "run.3.2",
            "state": snapshot,
        }
        if kind is RecordKind.RESOLUTION:
            defaults["ability"] = (ability_key,)
            if overrides.get("moment", Moment.RESOLUTION) is Moment.RESOLUTION:
                defaults["payload"] = ResolutionPayload(
                    events=(Event(type=EventType.DAMAGE_DEALT),),
                )
        defaults.update(overrides)
        return record_builders.make_record(kind, **defaults)

    return build


@pytest.fixture(scope="module")
def gen3_records():
    from effects.infrastructure.record_codec import record_from_dict
    from tests.unit.effects.builders import gen3_fixture_records

    return {record.record_id: record
            for record in map(record_from_dict, gen3_fixture_records())}
