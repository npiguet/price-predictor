"""Reading the corpus one shard at a time (T155).

The corpus reaches 14M records and would need hundreds of gigabytes to hold, so
the trainer never holds more than one shard. Four properties make that safe, and
each has a test below.

**A game never spans two shards**, so which stratum a game belongs to is
decidable from the shard alone. The collector gives every shard its own game-id
namespace, which is what lets the split be accumulated rather than derived.

**Validation comes from reserved shards the trainer never trains on.** Freezing
the validation records is what keeps ``--patience`` meaningful: an early-stopping
rule that compared a new sample of records each epoch would measure which
records got drawn rather than whether the model improved.

**A game naming a held-out card is excluded from training wherever it is found**,
not only in the reserved shards. Training on it would make the card-disjoint
number — the one that picks the shipping checkpoint — score partly on cards the
encoder had already seen.

**The split is enumerated as it accumulates.** The checkpoint records game ids
rather than the rule that produced them, because the corpus is append-only and
re-deriving the split against a grown corpus gives a different answer.
"""

from __future__ import annotations

from pathlib import Path

from effects.application.train_effect_model import (
    CorpusSplit,
    HeldOutCards,
    SplitAccumulator,
    epoch_shards,
    reserved_shard_indices,
    shard_games,
    steps_per_shard,
)
from effects.domain.records import (
    EffectRecord,
    Moment,
    RecordKind,
    ResolutionPayload,
)
from effects.domain.state_snapshot import EntityState, GlobalState, StateSnapshot


def _entity(name: str) -> EntityState:
    return EntityState(
        id=f"E-{name}", name=name, zone="battlefield", controller="P0",
    )


def _record(game_id: str, entities=()) -> EffectRecord:
    return EffectRecord(
        record_id=f"{game_id}.1", run_id="run", timestamp="t", game_id=game_id,
        kind=RecordKind.RESOLUTION, moment=Moment.RESOLUTION, actor_player="P0",
        ability=None,
        state=StateSnapshot(
            global_=GlobalState(
                turn=1, phase="main1", active="P0", priority="P0", stack_size=0,
            ),
            players=(),
            entities=tuple(entities),
        ),
        payload=ResolutionPayload(),
    )


def _held() -> HeldOutCards:
    return HeldOutCards(names=frozenset({"Serra Angel"}), script_files=frozenset())


class TestReservedShardIndices:
    """Which shards are held back for validation."""

    def test_the_reserved_shards_spread_across_the_corpus(self):
        # Consecutive shards come from one worker over one stretch of
        # collection, so taking the first four would sample one worker's
        # opening games rather than the corpus.
        picked = sorted(reserved_shard_indices(701, reserved=4))
        assert picked == [87, 262, 438, 613]

    def test_reserving_none_holds_nothing_back(self):
        assert reserved_shard_indices(701, reserved=0) == frozenset()

    def test_reserving_more_than_exist_holds_back_every_shard(self):
        assert reserved_shard_indices(3, reserved=10) == frozenset({0, 1, 2})

    def test_an_empty_corpus_reserves_nothing(self):
        assert reserved_shard_indices(0, reserved=4) == frozenset()

    def test_every_reserved_index_is_addressable(self):
        for count in range(1, 40):
            picked = reserved_shard_indices(count, reserved=4)
            assert all(0 <= index < count for index in picked)


class TestShardGames:
    """Sorting one shard's games by whether they name a held-out card."""

    def _records(self):
        return [
            _record("g0", entities=(_entity("Serra Angel"),)),
            _record("g0", entities=(_entity("Grizzly Bears"),)),
            _record("g1", entities=(_entity("Grizzly Bears"),)),
            _record("g2", entities=(_entity("Serra Angel"),)),
        ]

    def test_a_game_naming_a_held_out_card_is_tainted(self):
        tainted, _clean = shard_games(self._records(), _held())
        assert tainted == frozenset({"g0", "g2"})

    def test_the_whole_game_is_tainted_not_just_the_naming_record(self):
        # g0's second record names no held-out card and is still out.
        tainted, clean = shard_games(self._records(), _held())
        assert "g0" in tainted
        assert "g0" not in clean

    def test_a_game_naming_nothing_held_out_is_clean(self):
        _tainted, clean = shard_games(self._records(), _held())
        assert clean == frozenset({"g1"})

    def test_an_empty_shard_yields_no_games(self):
        tainted, clean = shard_games([], _held())
        assert not tainted and not clean


class TestSplitAccumulator:
    """Building the enumerated split as shards go by."""

    def test_a_tainted_game_is_card_disjoint_wherever_it_is_found(self):
        # Found in a training shard, not a reserved one: still excluded from
        # training, and still evaluable.
        acc = SplitAccumulator(held_out_cards=("Serra Angel",))
        acc.note_shard(
            [_record("g0", entities=(_entity("Serra Angel"),))], _held(),
            reserved=False,
        )
        assert acc.split().card_disjoint_games == frozenset({"g0"})

    def test_a_clean_game_in_a_reserved_shard_is_game_disjoint(self):
        acc = SplitAccumulator(held_out_cards=())
        acc.note_shard([_record("g1")], _held(), reserved=True)
        assert acc.split().game_disjoint_games == frozenset({"g1"})

    def test_a_clean_game_in_a_training_shard_is_neither(self):
        acc = SplitAccumulator(held_out_cards=())
        acc.note_shard([_record("g1")], _held(), reserved=False)
        split = acc.split()
        assert not split.card_disjoint_games and not split.game_disjoint_games
        assert split.is_training_game("g1")

    def test_the_two_strata_never_overlap(self):
        acc = SplitAccumulator(held_out_cards=("Serra Angel",))
        acc.note_shard(
            [_record("g0", entities=(_entity("Serra Angel"),)), _record("g1")],
            _held(), reserved=True,
        )
        split = acc.split()
        assert not (split.card_disjoint_games & split.game_disjoint_games)

    def test_the_accumulated_split_names_only_games_it_saw(self):
        acc = SplitAccumulator(held_out_cards=())
        acc.note_shard([_record("g1")], _held(), reserved=True)
        assert acc.split().validation_games == frozenset({"g1"})

    def test_the_held_out_card_list_is_carried_through(self):
        acc = SplitAccumulator(held_out_cards=("Serra Angel",))
        assert acc.split().held_out_cards == ("Serra Angel",)

    def test_an_inherited_split_overrides_the_local_rule(self):
        # A --split-from run reuses the checkpoint's games rather than
        # re-deriving them, so a grown corpus cannot move the boundary.
        inherited = CorpusSplit(
            held_out_cards=("Serra Angel",),
            card_disjoint_games=frozenset({"g5"}),
            game_disjoint_games=frozenset({"g6"}),
        )
        acc = SplitAccumulator.inheriting(inherited)
        acc.note_shard(
            [_record("g0", entities=(_entity("Serra Angel"),))], _held(),
            reserved=False,
        )
        # g0 names a held-out card but is not in the inherited split, so the
        # inherited boundary stands.
        assert acc.split() == inherited


class TestEpochShards:
    """Which shards an epoch reads, and how its steps divide among them."""

    def test_an_epoch_reads_its_share_of_the_corpus(self):
        shards = [Path(f"s{i}") for i in range(701)]
        assert len(epoch_shards(shards, epoch=1, per_epoch=18)) == 18

    def test_consecutive_epochs_read_different_shards(self):
        shards = [Path(f"s{i}") for i in range(701)]
        first = epoch_shards(shards, epoch=1, per_epoch=18)
        second = epoch_shards(shards, epoch=2, per_epoch=18)
        assert not set(first) & set(second)

    def test_forty_epochs_of_eighteen_cover_the_corpus_once(self):
        shards = [Path(f"s{i}") for i in range(701)]
        seen = {
            shard
            for epoch in range(1, 41)
            for shard in epoch_shards(shards, epoch=epoch, per_epoch=18)
        }
        assert seen == set(shards)

    def test_the_shard_list_wraps_rather_than_running_out(self):
        shards = [Path(f"s{i}") for i in range(10)]
        assert epoch_shards(shards, epoch=2, per_epoch=8) == [
            Path("s8"), Path("s9"), Path("s0"), Path("s1"),
            Path("s2"), Path("s3"), Path("s4"), Path("s5"),
        ]

    def test_an_empty_corpus_yields_no_shards(self):
        assert epoch_shards([], epoch=1, per_epoch=18) == []

    def test_the_steps_divide_evenly_across_the_epoch_s_shards(self):
        assert sum(steps_per_shard(5000, 18)) == 5000

    def test_every_shard_in_the_epoch_gets_steps(self):
        assert all(count > 0 for count in steps_per_shard(5000, 18))

    def test_the_remainder_spreads_rather_than_landing_on_one_shard(self):
        # 5000 over 18 is 277 with 14 left; no shard gets more than one extra.
        counts = steps_per_shard(5000, 18)
        assert max(counts) - min(counts) <= 1

    def test_fewer_steps_than_shards_still_sums_correctly(self):
        assert sum(steps_per_shard(5, 18)) == 5
