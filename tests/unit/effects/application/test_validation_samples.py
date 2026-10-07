"""The fixed validation samples build-corpus writes for the trainer."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.train_effect_model import sampling_class
from effects.application.validation_samples import _Smallest, class_quota, draw_samples
from effects.domain.effect_model import (
    CLASS_COMBAT,
    CLASS_RESOLUTION_COST,
    CLASS_RESOLUTION_EFFECT,
)
from effects.domain.records import CombatPayload, Moment, RecordKind
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.domain.conftest import ability_key, make_record, snapshot  # noqa: F401


def test_class_quota_rounds_each_share_and_keeps_every_class():
    quota = class_quota({"a": 0.5, "b": 0.5, "c": 0.001}, 100)
    assert quota == {"a": 50, "b": 50, "c": 1}


@pytest.fixture
def strata(tmp_path, make_record):  # noqa: F811 -- shadows the imported fixture by design
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
    for name, records in (("card", card), ("game", game), ("gate", gate)):
        write_shard(tmp_path / name / "shard-00001.jsonl.gz", records)
    return tmp_path


MIX = {CLASS_RESOLUTION_EFFECT: 0.5, CLASS_COMBAT: 0.5}


def test_each_stratum_holds_the_quota_per_class(strata):
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    for stratum in ("card-disjoint", "game-disjoint"):
        by_class = Counter(sampling_class(r) for r in samples[stratum])
        assert by_class == {CLASS_RESOLUTION_EFFECT: 10, CLASS_COMBAT: 10}


def test_card_disjoint_resolutions_come_from_the_gate_one_slice_first(strata):
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    resolutions = {r.record_id for r in samples["card-disjoint"] if r.kind is RecordKind.RESOLUTION}
    assert {f"r{i}" for i in range(6)} <= resolutions      # all six gate-one records
    assert len(resolutions) == 10                           # topped up from the stratum


def test_the_draw_is_seeded_and_shuffled(strata):
    a = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                     gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    b = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                     gate_one=strata / "gate", mix=MIX, size=20, seed=1)
    c = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                     gate_one=strata / "gate", mix=MIX, size=20, seed=2)
    def ids(sample):
        return [r.record_id for r in sample["game-disjoint"]]

    assert ids(a) == ids(b)
    assert ids(a) != ids(c)
    classes = [sampling_class(r) for r in a["game-disjoint"]]
    assert classes != sorted(classes), "a class-sorted sample scores one class per batch"


def test_a_short_class_takes_what_there_is(strata):
    samples = draw_samples(
        card_disjoint=strata / "card", game_disjoint=strata / "game",
        gate_one=strata / "gate", mix={CLASS_RESOLUTION_COST: 1.0}, size=5, seed=1,
    )
    assert samples["card-disjoint"] == []


def test_zero_validation_sample_reads_nothing_and_draws_nothing(strata, monkeypatch):
    """``--validation-sample 0`` means no sample: not even a read of the strata."""
    from effects.infrastructure import record_io

    def bomb(directory):
        raise AssertionError(f"draw_samples read {directory} despite size=0")

    monkeypatch.setattr(record_io, "read_records", bomb)
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix=MIX, size=0, seed=1)
    assert samples == {"card-disjoint": [], "game-disjoint": []}


def test_two_records_with_one_id_do_not_compare_the_records_themselves(make_record):  # noqa: F811
    """``_Smallest`` broke its tie on the record when hash and id both matched.

    ``EffectRecord`` is a dataclass with no ordering, so the comparison the
    heap falls through to raises ``TypeError`` rather than picking a winner.
    A monotonic counter never ties, so the record is never reached.
    """
    heap = _Smallest(4)
    heap.offer(7, make_record(record_id="same", game_id="g1"))
    heap.offer(7, make_record(record_id="same", game_id="g2"))
    assert len(heap.records()) == 2


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


def test_card_disjoint_resolution_slots_spread_over_held_out_texts(tmp_path, make_record):  # noqa: F811
    """Spec Story 4 scenario 9: a text with many records cannot crowd one with few."""
    from effects.application.build_corpus import ability_key as render
    from effects.domain.provenance import ProvenanceKey

    common = ProvenanceKey("cardsfolder/c/common.txt", 0, "spell", 0)
    rare = ProvenanceKey("cardsfolder/r/rare.txt", 0, "spell", 0)
    records = [
        make_record(record_id=f"common{i}", game_id="g1", ability=(common,)) for i in range(40)
    ] + [
        make_record(record_id=f"rare{i}", game_id="g2", ability=(rare,)) for i in range(3)
    ]
    for name in ("card", "gate"):
        write_shard(tmp_path / name / "s.jsonl.gz", records)
    write_shard(tmp_path / "game" / "s.jsonl.gz", records[:1])
    text_of_key = {
        render(records[0]): "common text", render(records[-1]): "rare text",
    }

    samples = draw_samples(
        card_disjoint=tmp_path / "card", game_disjoint=tmp_path / "game",
        gate_one=tmp_path / "gate", mix={CLASS_RESOLUTION_EFFECT: 1.0}, size=10, seed=1,
        held_out_text_of_key=text_of_key,
    )
    chosen = [r.record_id for r in samples["card-disjoint"]]
    assert len(chosen) == 10
    assert sum(1 for rid in chosen if rid.startswith("rare")) == 3
