"""Which records the trainer validates on, and where they come from.

The number this capture produces selects the shipping checkpoint and decides
when ``--patience`` stops the run, so three properties of it are load-bearing
and each has a test below.

**The sample follows the training mixture.** ``build-corpus`` mixes the training
stratum and leaves both validation strata in the proportions collection
produced. Those differ by more than a factor of five on the largest class, so a
validation loss read off the natural proportions is measuring a different
objective than the one training descends.

**The shards are drawn across the stratum.** A stratum's shard list is in path
order, so its opening shards all come from whichever collection run sorts first.

**Reading stops early only when the split is inherited.** A split derived from
the shards has to see every one of them, because a shard it skipped is a game
the checkpoint would fail to enumerate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from effects.application.train_effect_model import (
    CorpusSplit,
    HeldOutCards,
    TrainEffectModelConfig,
    sampling_class,
)
from effects.domain.records import (
    ActivationPayload,
    CombatPayload,
    ContinuousPayload,
    Costs,
    EffectRecord,
    Moment,
    PlayabilityAttackersPayload,
    PlayabilityDecisionPayload,
    PlayabilitySubkind,
    RecordKind,
    ResolutionOutcome,
    ResolutionPayload,
    RewritePayload,
    TriggerPayload,
)
from effects.domain.event_schema import Event, EventType
from effects.domain.state_snapshot import GlobalState, StateSnapshot

COUNTER = iter(range(1_000_000))

#: The smallest payload each kind accepts. Only the record's *class* matters
#: here, and that is read off ``kind``/``moment``/``subkind``, never the payload.
_PAYLOADS = {
    (RecordKind.RESOLUTION, Moment.RESOLUTION): ResolutionPayload,
    (RecordKind.RESOLUTION, Moment.ACTIVATION): lambda: ActivationPayload(
        costs=Costs(), outcome=ResolutionOutcome.RESOLVED,
    ),
    (RecordKind.REWRITE, None): lambda: RewritePayload(incoming=Event(type=EventType.LIFE_CHANGE, params={'delta': 1})),
    (RecordKind.CONTINUOUS, None): ContinuousPayload,
    (RecordKind.COMBAT, None): CombatPayload,
    (RecordKind.TRIGGER, None): lambda: TriggerPayload(
        event=Event(type=EventType.LIFE_CHANGE, params={'delta': 1}), fired=True,
    ),
    (RecordKind.PLAYABILITY, PlayabilitySubkind.DECISION):
        PlayabilityDecisionPayload,
    (RecordKind.PLAYABILITY, PlayabilitySubkind.ATTACKERS):
        PlayabilityAttackersPayload,
}


def _record(kind: RecordKind, *, game_id: str, subkind=None, moment=None):
    index = next(COUNTER)
    # Each kind takes its own payload type; the envelope enforces the pairing.
    payload = _PAYLOADS[(kind, moment or subkind)]()
    return EffectRecord(
        record_id=f"r{index}", run_id="run", timestamp="t", game_id=game_id,
        kind=kind, moment=moment, subkind=subkind, actor_player="P0",
        ability=None,
        state=StateSnapshot(
            global_=GlobalState(
                turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
            ),
            players=(), entities=(),
        ),
        payload=payload,
    )


def _natural_shard(game_id: str) -> list[EffectRecord]:
    """One shard in the proportions collection actually produces.

    Dominated by ``playability-decision`` the way the curated validation strata
    are: about six records in ten, against the tenth of a batch the training
    mixture gives it.
    """
    records = [
        _record(
            RecordKind.PLAYABILITY, game_id=game_id,
            subkind=PlayabilitySubkind.DECISION,
        )
        for _ in range(60)
    ]
    records += [
        _record(RecordKind.RESOLUTION, game_id=game_id, moment=Moment.RESOLUTION)
        for _ in range(14)
    ]
    records += [
        _record(RecordKind.RESOLUTION, game_id=game_id, moment=Moment.ACTIVATION)
        for _ in range(10)
    ]
    records += [_record(RecordKind.TRIGGER, game_id=game_id) for _ in range(6)]
    records += [_record(RecordKind.CONTINUOUS, game_id=game_id) for _ in range(5)]
    records += [_record(RecordKind.COMBAT, game_id=game_id) for _ in range(4)]
    records += [
        _record(
            RecordKind.PLAYABILITY, game_id=game_id,
            subkind=PlayabilitySubkind.ATTACKERS,
        )
        for _ in range(4)
    ]
    records += [_record(RecordKind.REWRITE, game_id=game_id) for _ in range(3)]
    return records


@pytest.fixture
def loop_with_shards(monkeypatch):
    """A loop over 60 fake shards, half in each stratum, plus a read counter."""
    from effects.application import training_loop as module

    shards = [Path(f"validation/s{i:03d}.jsonl.gz") for i in range(60)]
    # Shard i carries game "gi", and games alternate between the two strata so
    # a draw from anywhere in the list can fill both.
    card = frozenset(f"g{i}" for i in range(60) if i % 2 == 0)
    game = frozenset(f"g{i}" for i in range(60) if i % 2 == 1)
    read: list[Path] = []
    # A 2,048-record sample would need 143 rewrite records per stratum and
    # these 30 shards a stratum hold 90, so the sample could never fill and
    # every shard would be read — the very behaviour two of these tests are
    # about. Scaled down so the fixture can satisfy the mixture it is testing.
    monkeypatch.setattr(module, "VALIDATION_RECORDS_PER_STRATUM", 200)

    def fake_load(path: Path):
        read.append(path)
        return _natural_shard(f"g{int(path.stem.split('.')[0][1:])}")

    monkeypatch.setattr(module, "load_shard", fake_load)

    def build(*, inherited: bool):
        return module.TrainingLoop(
            TrainEffectModelConfig(),
            held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
            inherited=CorpusSplit(
                held_out_cards=(),
                card_disjoint_games=card,
                game_disjoint_games=game,
            ) if inherited else None,
            validation_shards=shards,
            training_shards=[],
        )

    return build, read, shards


class TestTheSampleFollowsTheTrainingMixture:
    """The share of each class in the captured sample."""

    def test_the_dominant_natural_class_is_cut_back_to_its_training_share(
        self, loop_with_shards,
    ):
        build, _read, _shards = loop_with_shards
        loop = build(inherited=True)
        loop._capture_validation()

        held = loop.card_disjoint
        share = sum(
            1 for r in held if sampling_class(r) == "playability-decision"
        ) / len(held)
        # Natural proportion is about 0.6; the training mixture asks for 0.10.
        assert share == pytest.approx(0.10, abs=0.03), (
            f"playability-decision is {share:.0%} of the sample; the training "
            "mixture gives it 10%, so the validation loss is weighted toward a "
            "class training barely optimizes"
        )

    def test_the_classes_training_favours_are_not_starved(
        self, loop_with_shards,
    ):
        build, _read, _shards = loop_with_shards
        loop = build(inherited=True)
        loop._capture_validation()

        held = loop.card_disjoint
        share = sum(
            1 for r in held if sampling_class(r) == "resolution-effect"
        ) / len(held)
        # Natural proportion is about 0.14; the training mixture asks for 0.30.
        assert share == pytest.approx(0.30, abs=0.03)

    def test_both_strata_are_sampled(self, loop_with_shards):
        build, _read, _shards = loop_with_shards
        loop = build(inherited=True)
        loop._capture_validation()

        assert loop.card_disjoint and loop.game_disjoint

    def test_consecutive_records_are_not_one_class_at_a_time(
        self, loop_with_shards,
    ):
        """Batches are cut from this list, and a batch scores only the fields
        its classes supervise — so a class-ordered sample would score each
        batch against a smaller objective than a training batch sees."""
        build, _read, _shards = loop_with_shards
        loop = build(inherited=True)
        loop._capture_validation()

        head = loop.card_disjoint[:32]
        assert len({sampling_class(r) for r in head}) > 1


class TestHowManyShardsAreRead:
    """Reading the whole stratum to keep two thousand records of it."""

    def test_an_inherited_split_stops_once_the_sample_is_full(
        self, loop_with_shards,
    ):
        build, read, shards = loop_with_shards
        loop = build(inherited=True)
        loop._capture_validation()

        assert len(read) < len(shards), (
            "every shard was read though the manifest already carried the "
            "split; the sample needed only a handful of them"
        )

    def test_a_derived_split_reads_every_shard(self, loop_with_shards):
        """It has to: a shard it skips is a game the checkpoint cannot name."""
        build, read, shards = loop_with_shards
        loop = build(inherited=False)
        loop._capture_validation()

        assert len(read) == len(shards)

    def test_the_shards_read_are_not_the_front_of_the_list(
        self, loop_with_shards,
    ):
        build, read, shards = loop_with_shards
        loop = build(inherited=True)
        loop._capture_validation()

        assert set(read) != set(shards[:len(read)]), (
            "the sample came off the front of the shard list, which on a "
            "curated corpus is one collection run's shards"
        )
