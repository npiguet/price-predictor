"""The fixed validation samples build-corpus writes for the trainer."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.validation_samples import _collect, class_quota, draw_samples
from effects.domain.records import CombatPayload, Moment, RecordKind
from effects.domain.sampling import (
    SamplingClass,
    sampling_class,
)
from effects.domain.strata import OUTPUT_STRATA, Stratum
from effects.infrastructure.corpus_store import CorpusStore
from effects.infrastructure.shard_io import write_shard


def _listed(store):
    """Each output stratum's shards as a build lists them (the ones written)."""
    return {str(stratum): tuple(sorted(store.output_dir(stratum).glob("*.jsonl.gz")))
            for stratum in OUTPUT_STRATA}


def test_class_quota_rounds_each_share_and_keeps_every_class():
    quota = class_quota({"a": 0.5, "b": 0.5, "c": 0.001}, 100)
    assert quota == {"a": 50, "b": 50, "c": 1}


@pytest.fixture
def strata(tmp_path, make_record):
    def res(i, game, moment=Moment.RESOLUTION):
        return make_record(record_id=f"r{i}", game_id=game, moment=moment)
    def combat(i, game):
        return make_record(record_id=f"c{i}", game_id=game, kind=RecordKind.COMBAT,
                           payload=CombatPayload())
    card = [res(i, f"cg{i % 3}") for i in range(30)] + [combat(i, f"cg{i % 3}") for i in range(30)]
    gate = card[:6]                       # six of the card-disjoint resolutions are gate-one
    game = (
        [res(i, f"gg{i % 3}") for i in range(100, 130)]
        + [combat(i, f"gg{i % 3}") for i in range(100, 130)]
    )
    store = CorpusStore(tmp_path)
    for stratum, records in (
        (Stratum.CARD_DISJOINT, card), (Stratum.GAME_DISJOINT, game), (Stratum.GATE_ONE, gate),
    ):
        write_shard(store.output_dir(stratum) / "shard-00001.jsonl.gz", records)
    return _listed(store)


MIX = {SamplingClass.RESOLUTION_EFFECT: 0.5, SamplingClass.COMBAT: 0.5}


def _one_text(record) -> str:
    """Every gate-one record acts through one held-out text, so the
    round-robin takes them by smallest hash."""
    return "held-out text"


def test_each_stratum_holds_the_quota_per_class(strata):
    samples = draw_samples(strata, mix=MIX, size=20, seed=1, held_out_text_of=_one_text)
    for stratum in ("card-disjoint", "game-disjoint"):
        by_class = Counter(sampling_class(r) for r in samples[stratum])
        assert by_class == {SamplingClass.RESOLUTION_EFFECT: 10, SamplingClass.COMBAT: 10}


def test_card_disjoint_resolutions_come_from_the_gate_one_slice_first(strata):
    samples = draw_samples(strata, mix=MIX, size=20, seed=1, held_out_text_of=_one_text)
    resolutions = {r.record_id for r in samples["card-disjoint"] if r.kind is RecordKind.RESOLUTION}
    assert {f"r{i}" for i in range(6)} <= resolutions      # all six gate-one records
    assert len(resolutions) == 10                           # topped up from the stratum


def test_the_draw_is_seeded_and_shuffled(strata):
    a = draw_samples(strata, mix=MIX, size=20, seed=1, held_out_text_of=_one_text)
    b = draw_samples(strata, mix=MIX, size=20, seed=1, held_out_text_of=_one_text)
    c = draw_samples(strata, mix=MIX, size=20, seed=2, held_out_text_of=_one_text)
    def ids(sample):
        return [r.record_id for r in sample["game-disjoint"]]

    assert ids(a) == ids(b)
    assert ids(a) != ids(c)
    classes = [sampling_class(r) for r in a["game-disjoint"]]
    assert classes != sorted(classes), "a class-sorted sample scores one class per batch"


def test_a_short_class_takes_what_there_is(strata):
    samples = draw_samples(
        strata, mix={SamplingClass.RESOLUTION_COST: 1.0}, size=5, seed=1,
        held_out_text_of=_one_text,
    )
    assert samples["card-disjoint"] == []


def test_zero_validation_sample_reads_nothing_and_draws_nothing(strata, monkeypatch):
    """``--validation-sample 0`` means no sample: not even a read of the strata."""
    from effects.infrastructure import shard_io

    def bomb(shard):
        raise AssertionError(f"draw_samples read {shard} despite size=0")

    monkeypatch.setattr(shard_io, "read_shard", bomb)
    samples = draw_samples(strata, mix=MIX, size=0, seed=1, held_out_text_of=_one_text)
    assert samples == {"card-disjoint": [], "game-disjoint": []}


def test_the_copies_of_one_record_take_one_place(tmp_path, make_record):
    """A stratum writes a record once per copy, every copy under its one
    ``record_id`` and so one hash: the sample keeps one copy, so a class
    copied four times still fills its quota with distinct records."""
    write_shard(tmp_path / "shard-00001.jsonl.gz", [
        make_record(record_id="same", game_id="g1"),
        make_record(record_id="same", game_id="g1"),
        make_record(record_id="same", game_id="g1"),
        make_record(record_id="other", game_id="g2"),
    ])
    shards = [tmp_path / "shard-00001.jsonl.gz"]
    heaps = _collect(shards, {SamplingClass.RESOLUTION_EFFECT: 2}, seed=1)
    assert sorted(r.record_id for r in heaps[SamplingClass.RESOLUTION_EFFECT].items()) == [
        "other", "same"]


def _copies(make_record, record_id: str, game_id: str, added: int, **fields) -> list:
    """A record as a stratum writes it: its natural copy and ``added`` copies
    rule budgets drew for one rule."""
    from effects.domain.records import Curation

    copies = 1 + added
    natural = Curation(natural_share=1, copies=copies, drawn_rules=("702.2b",) * bool(added))
    drawn = Curation(natural_share=1, copies=copies, drawn_rule="702.2b",
                     drawn_rules=("702.2b",))
    return ([make_record(record_id=record_id, game_id=game_id, curation=drawn, **fields)
             for _ in range(added)]
            + [make_record(record_id=record_id, game_id=game_id, curation=natural, **fields)])


def test_a_copied_class_fills_its_quota_with_distinct_natural_copies(tmp_path, make_record):
    records = [copy for i in range(12)
               for copy in _copies(make_record, f"r{i}", f"g{i % 3}", added=3)]
    store = CorpusStore(tmp_path)
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GAME_DISJOINT, Stratum.GATE_ONE):
        write_shard(store.output_dir(stratum) / "s.jsonl.gz", records)
    samples = draw_samples(
        _listed(store), mix={SamplingClass.RESOLUTION_EFFECT: 1.0}, size=8, seed=1,
        held_out_text_of=_one_text)
    for stratum in ("card-disjoint", "game-disjoint"):
        sample = samples[stratum]
        assert len({r.record_id for r in sample}) == len(sample) == 8, stratum
        assert all(not r.curation.is_added for r in sample), stratum


# ── FR-051: round-robin over held-out texts (T081) ──────────────────────


def test_round_robin_takes_four_per_text_per_round_by_smallest_value():
    from effects.application.validation_samples import round_robin

    taken = round_robin({"a": list(range(100, 110)), "b": [5, 1, 3]}, quota=10)

    # Round one: a's four smallest, then all three of b; round two: a's next four.
    assert taken[:4] == [100, 101, 102, 103]
    assert sorted(taken[4:7]) == [1, 3, 5]
    assert taken[7:] == [104, 105, 106]


def test_round_robin_stops_when_every_text_is_exhausted():
    from effects.application.validation_samples import round_robin

    assert sorted(round_robin({"a": [1], "b": [2, 3]}, quota=50)) == [1, 2, 3]


def test_card_disjoint_resolution_slots_spread_over_held_out_texts(tmp_path, make_record):
    """Spec Story 4 scenario 9: a text with many records cannot crowd one with few."""
    from effects.domain.provenance import ProvenanceKey

    common = ProvenanceKey("cardsfolder/c/common.txt", 0, "spell", 0)
    rare = ProvenanceKey("cardsfolder/r/rare.txt", 0, "spell", 0)
    records = [
        make_record(record_id=f"common{i}", game_id="g1", ability=(common,)) for i in range(40)
    ] + [
        make_record(record_id=f"rare{i}", game_id="g2", ability=(rare,)) for i in range(3)
    ]
    store = CorpusStore(tmp_path)
    for stratum in (Stratum.CARD_DISJOINT, Stratum.GATE_ONE):
        write_shard(store.output_dir(stratum) / "s.jsonl.gz", records)
    write_shard(store.game_disjoint_dir / "s.jsonl.gz", records[:1])
    text_of = {common: "common text", rare: "rare text"}

    samples = draw_samples(
        _listed(store), mix={SamplingClass.RESOLUTION_EFFECT: 1.0}, size=10, seed=1,
        held_out_text_of=lambda record: text_of[record.ability[0]],
    )
    chosen = [r.record_id for r in samples["card-disjoint"]]
    assert len(chosen) == 10
    assert sum(1 for rid in chosen if rid.startswith("rare")) == 3
