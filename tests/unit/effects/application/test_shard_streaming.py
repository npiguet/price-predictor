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
        assert len(epoch_shards(shards, epoch=1, per_epoch=18, seed=1)) == 18

    def test_an_epoch_reads_each_shard_at_most_once(self):
        shards = [Path(f"s{i}") for i in range(701)]
        picks = epoch_shards(shards, epoch=1, per_epoch=64, seed=1)
        assert len(set(picks)) == len(picks)

    def test_consecutive_epochs_read_different_shards(self):
        shards = [Path(f"s{i}") for i in range(701)]
        first = epoch_shards(shards, epoch=1, per_epoch=18, seed=1)
        second = epoch_shards(shards, epoch=2, per_epoch=18, seed=1)
        assert first != second

    def test_an_epoch_reaches_the_end_of_the_corpus(self):
        """The regression: a contiguous walk never left the opening family.

        A curated corpus is written in path order, so its depleted shards come
        first and its synthetic variants last — 69 shards of 3,194, starting at
        index 3,125. Read eighteen contiguous shards an epoch and a forty-epoch
        run walks to index 720, training a model that never saw a variant on a
        corpus built to supply them. An epoch's draw has to span the list.
        """
        shards = [Path(f"s{i}") for i in range(3194)]
        tail = set(shards[3125:])
        reached = [
            epoch
            for epoch in range(1, 11)
            if tail & set(epoch_shards(shards, epoch=epoch, per_epoch=256, seed=7))
        ]
        assert len(reached) >= 8, (
            f"only {len(reached)} of ten epochs reached the last 2% of the "
            "corpus; the draw is not spanning the list"
        )

    def test_one_seed_and_epoch_always_draw_the_same_shards(self):
        shards = [Path(f"s{i}") for i in range(701)]
        assert (
            epoch_shards(shards, epoch=3, per_epoch=18, seed=99)
            == epoch_shards(shards, epoch=3, per_epoch=18, seed=99)
        )

    def test_a_different_seed_draws_different_shards(self):
        shards = [Path(f"s{i}") for i in range(701)]
        assert (
            epoch_shards(shards, epoch=1, per_epoch=18, seed=1)
            != epoch_shards(shards, epoch=1, per_epoch=18, seed=2)
        )

    def test_a_corpus_smaller_than_the_draw_is_read_whole(self):
        shards = [Path(f"s{i}") for i in range(10)]
        picks = epoch_shards(shards, epoch=2, per_epoch=18, seed=1)
        assert sorted(picks) == sorted(shards)

    def test_an_empty_corpus_yields_no_shards(self):
        assert epoch_shards([], epoch=1, per_epoch=18, seed=1) == []

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


class TestProgressFormatting:
    """The within-shard progress line's two optional sections."""

    def test_gradient_norms_render_per_group(self):
        from effects.application.training_loop import _format_norms

        line = _format_norms({"encoder": 0.8412, "head": 1.9733})
        assert "encoder 0.84" in line
        assert "head 1.97" in line

    def test_no_gradient_section_before_the_first_clip(self):
        from effects.application.training_loop import _format_norms

        assert _format_norms({}) == ""

    def test_field_terms_are_ordered_largest_first(self):
        """A blow-up is one field carrying the loss; name order buries it."""
        from effects.application.training_loop import _format_parts

        line = _format_parts({"gate": 0.41, "zone": 0.63, "pt": 0.28})
        assert line.index("zone") < line.index("gate") < line.index("pt")

    def test_no_field_section_when_the_batch_reported_none(self):
        from effects.application.training_loop import _format_parts

        assert _format_parts({}) == ""


class TestWhenAStepReportsItsNumbers:
    """Reading a loss term back is a device sync, so one step per shard does it.

    That step is the shard's last, and its numbers ride the line the shard
    already logs when it ends. An earlier version gated this on a fifteen-second
    wall clock and went silent the moment an epoch spread over 256 shards: a
    shard's twenty steps take a few seconds, so the window never elapsed and no
    shard ever reported a loss, a gradient norm or a field breakdown.
    """

    def test_the_last_step_of_a_shard_reports(self):
        from effects.application.training_loop import reports_now

        assert reports_now(index=19, budget=20)

    def test_a_one_step_shard_still_reports(self):
        from effects.application.training_loop import reports_now

        assert reports_now(index=0, budget=1)

    def test_no_earlier_step_reports(self):
        from effects.application.training_loop import reports_now

        assert not any(reports_now(index=i, budget=20) for i in range(19))
