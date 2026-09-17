"""The trainer reads its validation set from the corpus (FR-089, FR-125 withdrawn)."""

from __future__ import annotations

import logging

import pytest
import torch

from effects.application.train_effect_model import (
    HeldOutCards,
    TrainEffectModelConfig,
    run,
)
from effects.application.training_loop import (
    FloorCache,
    TrainingLoop,
    _format_floor,
    _format_parts,
)
from effects.domain.effect_model import (
    FIELDS_BY_NAME,
    SAMPLING_CLASSES,
    EntityTargetBatch,
    constant_predictor_floor,
)
from effects.domain.records import CombatPayload, RecordKind
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.domain.conftest import (  # noqa: F401
    ability_key,
    make_record,
    snapshot,
)


@pytest.fixture
def samples(tmp_path, make_record):  # noqa: F811
    card = [make_record(record_id=f"c{i}", game_id="cg") for i in range(5)]
    game = [make_record(record_id=f"g{i}", game_id="gg", kind=RecordKind.COMBAT,
                        payload=CombatPayload()) for i in range(3)]
    paths = {"card-disjoint": tmp_path / "card-disjoint.jsonl.gz",
             "game-disjoint": tmp_path / "game-disjoint.jsonl.gz"}
    write_shard(paths["card-disjoint"], card)
    write_shard(paths["game-disjoint"], game)
    return paths


def test_the_loop_loads_both_samples_and_the_probe(samples):
    loop = TrainingLoop(
        TrainEffectModelConfig(corpus="unused"),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples=samples,
        holdout_permille=20, holdout_max_carriers=8,
        gate_one_records=0, rarity={}, corpus_digest="",
    )
    loop._load_validation()
    assert [r.record_id for r in loop.card_disjoint] == [f"c{i}" for i in range(5)]
    assert len(loop.game_disjoint) == 3
    assert loop.probe and loop.probe[0].record_id == "c0"
    assert loop.present == {"resolution-effect", "combat"}


def test_run_refuses_to_train_without_a_corpus(caplog):
    assert run(TrainEffectModelConfig()) == 1
    assert "--corpus" in caplog.text


def test_the_checkpoint_records_the_corpuss_holdout_rule_not_a_constant():
    """The manifest carries the rule the corpus was composed against (FR-134).

    A constant here would be a second spelling of it, and a corpus built with
    anything but the defaults would be described by a checkpoint that never
    read it.
    """
    loop = TrainingLoop(
        TrainEffectModelConfig(corpus="unused"),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples={},
        holdout_permille=30, holdout_max_carriers=5,
        gate_one_records=0, rarity={}, corpus_digest="",
    )

    provenance = loop._provenance()

    assert provenance.holdout_permille == 30
    assert provenance.holdout_max_carriers == 5


def test_validation_batches_drop_the_augmentations_but_keep_the_withholding():
    """A validation number must not move with the draw (FR-082).

    Keyword expansion and context dropout vary what the model sees from one
    step to the next, which is what they are for in training and exactly what a
    number compared across epochs must not have. The withheld keyword is not an
    augmentation but a split, so it stays withheld: handing it back at
    validation would score the one thing training never saw.
    """
    loop = TrainingLoop(
        TrainEffectModelConfig(
            corpus="unused", keyword_expand_p=0.5, context_dropout=0.3,
            withhold_keyword="lifelink",
        ),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples={},
        holdout_permille=20, holdout_max_carriers=8,
        gate_one_records=0, rarity={}, corpus_digest="",
    )

    training = loop._batcher(None, None, {})
    assert (training.keyword_expand_p, training.context_dropout) == (0.5, 0.3)

    scoring = loop._batcher(None, None, {}, training=False)
    assert (scoring.keyword_expand_p, scoring.context_dropout) == (0.0, 0.0)
    assert scoring.withhold_keyword == "lifelink"
    # A validation pass must not advance the training stream: the batcher
    # draws from its own generator, so how many validation batches ran
    # cannot change which training records the next epoch sees.
    assert scoring.rng is not loop.rng
    assert training.rng is loop.rng


def test_a_sample_missing_a_class_says_so(tmp_path, make_record, caplog):  # noqa: F811
    """The epoch's field set comes from the validation samples (FR-085).

    A class the corpus trains on but that neither sample holds drops every
    field only it supervises, for the whole run, with nothing logged.
    """
    paths = {"card-disjoint": tmp_path / "card-disjoint.jsonl.gz",
             "game-disjoint": tmp_path / "game-disjoint.jsonl.gz"}
    write_shard(paths["card-disjoint"],
                [make_record(record_id="c0", game_id="cg")])
    write_shard(paths["game-disjoint"],
                [make_record(record_id="g0", game_id="gg",
                             kind=RecordKind.COMBAT, payload=CombatPayload())])
    loop = TrainingLoop(
        TrainEffectModelConfig(corpus="unused"),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples=paths,
        holdout_permille=20, holdout_max_carriers=8,
        gate_one_records=0, rarity={}, corpus_digest="",
    )
    loop._load_validation()

    with caplog.at_level(logging.WARNING):
        loop._warn_missing_classes()

    assert loop.present == {"resolution-effect", "combat"}
    for absent in set(SAMPLING_CLASSES) - loop.present:
        assert absent in caplog.text
    assert "no loss" in caplog.text


def test_a_sample_naming_every_class_says_nothing(caplog):
    loop = TrainingLoop(
        TrainEffectModelConfig(corpus="unused"),
        held_out=HeldOutCards(names=frozenset(), script_files=frozenset()),
        inherited=None, training_shards=[], validation_samples={},
        holdout_permille=20, holdout_max_carriers=8,
        gate_one_records=0, rarity={}, corpus_digest="",
    )
    loop.present = set(SAMPLING_CLASSES)

    with caplog.at_level(logging.WARNING):
        loop._warn_missing_classes()

    assert caplog.text == ""


class TestExplainedDeviance:
    """The card-disjoint breakdown says how far each head beat a constant.

    A field's loss is a number in nats with no scale of its own, so the epoch
    line carries `1 - loss/floor` beside it: zero is the base rate, negative is
    worse than predicting it (FR-127c).
    """

    def test_each_field_carries_its_explained_deviance(self):
        line = _format_parts(
            {"gate": 2.878, "damage_taken": 0.761},
            floor={"gate": 3.105, "damage_taken": 1.902},
        )
        assert line == "\n  fields: gate 2.878 (7%), damage_taken 0.761 (60%)"

    def test_the_largest_term_still_comes_first(self):
        line = _format_parts(
            {"damage_taken": 0.761, "gate": 2.878},
            floor={"gate": 3.105, "damage_taken": 1.902},
        )
        assert line.index("gate") < line.index("damage_taken")

    def test_a_head_worse_than_the_constant_reads_negative(self):
        line = _format_parts({"gate": 4.0}, floor={"gate": 2.0})
        assert "gate 4.000 (-100%)" in line

    def test_a_field_with_no_deviance_to_explain_reads_not_applicable(self):
        """A field whose targets never vary has nothing for a head to beat."""
        line = _format_parts({"types_gained": 0.0}, floor={"types_gained": 0.0})
        assert "types_gained 0.000 (n/a)" in line

    def test_the_per_shard_training_breakdown_is_unchanged(self):
        assert _format_parts({"gate": 2.0}) == "\n  fields: gate 2.000"

    def test_the_floor_is_logged_largest_first(self):
        assert _format_floor(
            {"damage_taken": 1.902, "gate": 3.105},
        ) == "gate 3.105, damage_taken 1.902"


class TestFloorCache:
    """The floor moves only with the active field set (FR-127c).

    The validation targets are fixed for the run, so the floor is too — until
    the curriculum step admits the sparse group, which is a different objective
    and needs its own floor.
    """

    def test_it_is_recomputed_exactly_when_the_fields_change(self):
        cache = FloorCache()
        dense = (FIELDS_BY_NAME["damage_taken"],)
        sparse = (*dense, FIELDS_BY_NAME["control_change"])

        computed = []
        for epoch, fields in ((1, dense), (2, dense), (3, sparse), (4, sparse)):
            if cache.stale(fields):
                computed.append(epoch)
                cache.update(fields, {"gate": float(epoch)})

        assert computed == [1, 3]
        assert cache.floor == {"gate": 3.0}

    def test_an_unfilled_cache_scales_nothing(self):
        assert FloorCache().floor == {}


def test_a_field_whose_targets_never_vary_reads_not_applicable_end_to_end():
    """From the sample's targets to the epoch line (FR-127c).

    A count nothing ever happens to has no floor, so a head's arbitrary log-rate
    against it is not a percentage of anything — the term says so rather than
    reporting the ratio the rate clamp would otherwise produce.
    """
    gate = torch.ones(2, 2)
    batch = EntityTargetBatch(
        gate=gate, fields={"damage_taken": torch.zeros(2, 2)}, mask=gate,
    )
    fields = (FIELDS_BY_NAME["damage_taken"],)

    floor = constant_predictor_floor([batch], fields=fields)
    line = _format_parts({"damage_taken": 0.0067, "gate": 0.5}, floor)

    assert floor["damage_taken"] == 0.0
    assert "damage_taken 0.007 (n/a)" in line
