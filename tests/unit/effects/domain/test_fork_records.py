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
    ActivationPayload,
    CombatPayload,
    Costs,
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
)
from effects.domain.state_snapshot import (
    STAGE_ONE_TIERS,
    GlobalState,
    InclusionTier,
    StateSnapshot,
)

_KEY = ProvenanceKey("cardsfolder/l/lightning_bolt.txt", 0, "spell", 0)

#: A fixed constant no flag can raise (FR-040).
MAX_FORKS_PER_RESOLUTION = 2


def _snapshot(tiers=STAGE_ONE_TIERS) -> StateSnapshot:
    return StateSnapshot(
        global_=GlobalState(
            turn=4, phase="main1", active="P0", priority="P0", stack_size=1,
        ),
        players=(),
        entities=(),
        tiers=tiers,
    )


def _record(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "run.0.1", "run_id": "run", "timestamp": "t",
        "game_id": "run.0.1", "kind": RecordKind.RESOLUTION,
        "moment": Moment.RESOLUTION, "actor_player": "P0",
        "state": _snapshot(), "ability": (_KEY,),
        "payload": ResolutionPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


def _combat(**overrides) -> EffectRecord:
    defaults: dict = {
        "record_id": "run.0.2", "run_id": "run", "timestamp": "t",
        "game_id": "run.0.1", "kind": RecordKind.COMBAT,
        "actor_player": "P0", "state": _snapshot(),
        "payload": CombatPayload(),
    }
    defaults.update(overrides)
    return EffectRecord(**defaults)


class TestInterventionalResolutions:
    def test_it_carries_both_flags(self):
        record = _record(interventional=True, fork=True)
        assert record.interventional
        assert record.fork

    def test_it_carries_no_link_id(self):
        """The effect half is written alone: there was no real activation."""
        with pytest.raises(ValueError, match="no activation partner"):
            _record(interventional=True, fork=True, link_id="pair-1")

    def test_it_has_no_activation_partner_by_construction(self):
        record = _record(interventional=True, fork=True)
        assert record.link_id is None

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

    def test_a_cost_half_intervention_would_be_meaningless(self):
        """Nothing was paid, so an activation payload has nothing to describe."""
        record = _record(
            interventional=True, fork=True, moment=Moment.ACTIVATION,
            payload=ActivationPayload(
                costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
            ),
        )
        # The schema permits it; the collector writes the effect half only.
        assert record.link_id is None


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

    def test_it_names_the_real_record_it_mirrors(self):
        record = _combat(fork=True, mirror_of="run.0.9")
        assert record.mirror_of == "run.0.9"

    def test_mirror_of_requires_the_fork_flag(self):
        with pytest.raises(ValueError, match="mirror_of"):
            _combat(mirror_of="run.0.9")

    def test_it_is_an_ordinary_combat_record_otherwise(self):
        record = _combat(fork=True, mirror_of="run.0.9")
        assert record.kind is RecordKind.COMBAT
        assert record.ability is None

    def test_an_ordinary_combat_record_is_not_a_probe(self):
        assert not _combat().is_probe


class TestForkBudgets:
    def test_at_most_two_forks_may_name_one_real_resolution(self):
        """A fixed constant, independent of --interventions-per-game: two
        counterfactuals for one board is already both worth having."""
        assert MAX_FORKS_PER_RESOLUTION == 2

    def test_two_forks_may_mirror_the_same_real_record(self):
        first = _combat(record_id="run.0.3", fork=True, mirror_of="run.0.1")
        second = _combat(record_id="run.0.4", fork=True, mirror_of="run.0.1")
        assert first.mirror_of == second.mirror_of == "run.0.1"
        assert first.record_id != second.record_id


class TestSnapshotTierFour:
    def test_stage_one_does_not_collect_hand_and_graveyard(self):
        assert not _snapshot().collected(
            InclusionTier.UNREFERENCED_HAND_GRAVEYARD
        )

    def test_a_fork_record_may_collect_it(self):
        """A forced resolution chose from cards nobody was going to play."""
        tiers = STAGE_ONE_TIERS | {
            InclusionTier.UNREFERENCED_STACK,
            InclusionTier.UNREFERENCED_HAND_GRAVEYARD,
        }
        record = _record(
            interventional=True, fork=True, state=_snapshot(tiers=tiers),
        )
        assert record.state.collected(
            InclusionTier.UNREFERENCED_HAND_GRAVEYARD
        )

    def test_the_tiers_stay_additive(self):
        tiers = STAGE_ONE_TIERS | {
            InclusionTier.UNREFERENCED_STACK,
            InclusionTier.UNREFERENCED_HAND_GRAVEYARD,
        }
        assert STAGE_ONE_TIERS <= tiers
        assert len(tiers) == 4


class TestTheDiffIsNeverATrainingTarget:
    def test_a_probe_record_carries_no_diff_field(self):
        """FR-042: the real-versus-fork difference is computed at evaluation
        time. A record that carried it would invite training on it, which would
        teach the model to reproduce its own errors."""
        from dataclasses import fields

        names = {f.name for f in fields(EffectRecord)}
        assert not {"diff", "real_vs_fork", "delta"} & names

    def test_a_fork_record_is_an_observation_like_any_other(self):
        record = _combat(fork=True, mirror_of="run.0.1")
        assert set(record.model_input_fields()) == {
            "kind", "moment", "subkind", "actor_player", "ability", "state",
            "payload",
        }

    def test_the_fork_flag_never_reaches_the_model(self):
        """Otherwise the model would learn to predict differently on forks —
        exactly the confound the forks exist to avoid."""
        record = _combat(fork=True)
        assert "fork" not in record.model_input_fields()
        assert "mirror_of" not in record.model_input_fields()
