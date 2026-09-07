"""Shared record fixtures for the schema suites."""

from __future__ import annotations

import pytest

from effects.domain.event_schema import Event, EventType
from effects.domain.provenance import ProvenanceKey
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    Costs,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.state_snapshot import GlobalState, StateSnapshot


@pytest.fixture
def snapshot() -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=4, phase="main1", active="P0", priority="P0", stack_size=1,
        ),
        players=(),
        entities=(),
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
    """Build a valid record of any kind, overriding whatever the test needs."""

    def build(**overrides) -> EffectRecord:
        kind = overrides.pop("kind", RecordKind.RESOLUTION)
        defaults: dict = {
            "record_id": "run.3.11",
            "run_id": "run",
            "timestamp": "2026-09-06T15:31:12.152018Z",
            "game_id": "run.3.2",
            "kind": kind,
            "actor_player": "P0",
            "state": snapshot,
        }
        if kind is RecordKind.RESOLUTION:
            moment = overrides.pop("moment", Moment.RESOLUTION)
            defaults["moment"] = moment
            defaults["ability"] = (ability_key,)
            defaults["payload"] = (
                ResolutionPayload(events=(Event(type=EventType.DAMAGE_DEALT),))
                if moment is Moment.RESOLUTION
                else ActivationPayload(
                    costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
                )
            )
        elif kind is RecordKind.COMBAT:
            defaults["payload"] = CombatPayload()
        else:
            raise ValueError(f"make_record has no default payload for {kind}")
        defaults.update(overrides)
        return EffectRecord(**defaults)

    return build
