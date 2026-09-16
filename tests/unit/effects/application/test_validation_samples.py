"""The fixed validation samples build-corpus writes for the trainer."""

from __future__ import annotations

from collections import Counter

import pytest

from effects.application.validation_samples import class_quota, draw_samples
from effects.domain.effect_model import (
    CLASS_COMBAT, CLASS_RESOLUTION_COST, CLASS_RESOLUTION_EFFECT,
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
    game = [res(i, f"gg{i % 3}") for i in range(100, 130)] + [combat(i, f"gg{i % 3}") for i in range(100, 130)]
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
    ids = lambda s: [r.record_id for r in s["game-disjoint"]]
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)
    classes = [sampling_class(r) for r in a["game-disjoint"]]
    assert classes != sorted(classes), "a class-sorted sample scores one class per batch"


def test_a_short_class_takes_what_there_is(strata):
    samples = draw_samples(card_disjoint=strata / "card", game_disjoint=strata / "game",
                           gate_one=strata / "gate", mix={CLASS_RESOLUTION_COST: 1.0}, size=5, seed=1)
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


from effects.application.train_effect_model import sampling_class  # noqa: E402
