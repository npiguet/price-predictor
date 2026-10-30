"""Fork records and the flags that tell the two kinds apart (T124).

An interventional resolution and a damage-step probe are both forks, and the
probe's ``interventional = False`` is what separates them **by flags alone** — a
reader never has to look at the payload to know which it has.
"""

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
from effects.domain.state_snapshot import (
    GlobalState,
    StateSnapshot,
)
from tests.unit.effects.builders import make_record, make_snapshot

_KEY = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)

def _snapshot() -> StateSnapshot:
    return make_snapshot(
        global_=GlobalState(
            turn=4, phase="main1", active="P0", priority="P0", stack_size=1,
        ),
    )


def _record(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "run.0.1", "game_id": "run.0.1",
        "state": _snapshot(), "ability": (_KEY,),
    }
    defaults.update(overrides)
    return make_record(**defaults)


def _combat(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "run.0.2", "game_id": "run.0.1", "state": _snapshot(),
    }
    defaults.update(overrides)
    return make_record(RecordKind.COMBAT, **defaults)


class TestInterventionalResolutions:
    def test_it_carries_no_link_id(self):
        """The effect half is written alone: there was no real activation."""
        with pytest.raises(ValueError, match="no activation partner"):
            _record(interventional=True, fork=True, link_id="pair-1")

    def test_it_is_a_resolution_and_nothing_else(self):
        with pytest.raises(ValueError, match="forced resolution"):
            _combat(interventional=True, fork=True)

    def test_it_still_names_its_acting_line(self):
        """The ability is the whole point: it is what the fork forced."""
        assert _record(interventional=True, fork=True).ability == (_KEY,)

    def test_it_carries_the_effect_half_only(self):
        record = _record(
            interventional=True, fork=True,
            payload=ResolutionPayload(events=(
                Event(type=EventType.DAMAGE_DEALT, subjects=("E1",),
                      params={"amount": 3}),
            )),
        )
        assert record.moment is Moment.RESOLUTION
        assert len(record.payload.events) == 1

class TestDamageStepProbes:
    def test_a_probe_is_a_fork_that_intervened_in_nothing(self):
        record = _combat(fork=True, interventional=False)
        assert record.is_probe

    def test_the_flags_alone_distinguish_it_from_an_intervention(self):
        probe = _combat(fork=True)
        intervention = _record(interventional=True, fork=True)
        assert (probe.fork, probe.interventional) == (True, False)
        assert (intervention.fork, intervention.interventional) == (True, True)
        assert probe.is_probe
        assert not intervention.is_probe

    def test_mirror_of_requires_the_fork_flag(self):
        with pytest.raises(ValueError, match="mirror_of"):
            _combat(mirror_of="run.0.9")

    def test_it_is_an_ordinary_combat_record_otherwise(self):
        record = _combat(fork=True, mirror_of="run.0.9")
        assert record.kind is RecordKind.COMBAT
        assert record.ability is None

    def test_an_ordinary_combat_record_is_not_a_probe(self):
        assert not _combat().is_probe


class TestTheDiffIsNeverATrainingTarget:
    def test_a_probe_record_carries_no_diff_field(self):
        """FR-042: the real-versus-fork difference is computed at evaluation
        time. A record that carried it would invite training on it, which would
        teach the model to reproduce its own errors."""
        from dataclasses import fields

        names = {f.name for f in fields(EffectRecord)}
        assert not {"diff", "real_vs_fork", "delta"} & names

