"""The trainer reads its validation set from the corpus (FR-089, FR-125 withdrawn)."""

from __future__ import annotations

import logging
import random

import pytest
import torch

from effects.application.train_effect_model import (
    CorpusFacts,
    TrainEffectModelConfig,
    run,
)
from effects.application.training_log import format_floor, format_parts
from effects.application.validation_pass import FloorCache, ValidationSamples
from effects.domain.effect_head_layout import FIELDS_BY_NAME
from effects.domain.effect_losses import LossParts
from effects.domain.loss_floor import constant_predictor_floor
from effects.domain.records import CombatPayload, RecordKind
from effects.domain.sampling import SAMPLING_CLASSES
from effects.domain.split import HoldoutRule
from effects.infrastructure.record_io import write_shard
from tests.unit.effects.builders import (
    StubLosses,
    fields_everywhere,
    make_entity_targets,
    make_preparer,
    make_training_loop,
)


@pytest.fixture
def samples(tmp_path, make_record):
    card = [make_record(record_id=f"c{i}", game_id="cg") for i in range(5)]
    game = [make_record(record_id=f"g{i}", game_id="gg", kind=RecordKind.COMBAT,
                        payload=CombatPayload()) for i in range(3)]
    paths = {"card-disjoint": tmp_path / "card-disjoint.jsonl.gz",
             "game-disjoint": tmp_path / "game-disjoint.jsonl.gz"}
    write_shard(paths["card-disjoint"], card)
    write_shard(paths["game-disjoint"], game)
    return paths


def test_the_loop_loads_both_samples_and_the_probe(samples):
    samples = ValidationSamples.load(samples)
    assert [r.record_id for r in samples.card_disjoint] == [f"c{i}" for i in range(5)]
    assert len(samples.game_disjoint) == 3
    assert samples.probe and samples.probe[0].record_id == "c0"
    assert samples.present == {"resolution-effect", "combat"}


def test_run_refuses_to_train_without_a_corpus(caplog):
    assert run(TrainEffectModelConfig()) == 1
    assert "--corpus" in caplog.text


def test_the_checkpoint_records_the_corpuss_holdout_rule_not_a_constant():
    """The manifest carries the rule the corpus was composed against (FR-134).

    A constant here would be a second spelling of it, and a corpus built with
    anything but the defaults would be described by a checkpoint that never
    read it.
    """
    provenance = CorpusFacts(holdout=HoldoutRule(30, 5)).provenance(
        TrainEffectModelConfig(corpus="unused"),
    )

    assert provenance.holdout_permille == 30
    assert provenance.holdout_max_carriers == 5


def test_validation_batches_drop_the_augmentations_but_keep_the_withholding(tmp_path):
    """A validation number must not move with the draw (FR-082).

    Keyword expansion and context dropout vary what the model sees from one
    step to the next, which is what they are for in training and exactly what a
    number compared across epochs must not have. The withheld keyword is not an
    augmentation but a split, so it stays withheld: handing it back at
    validation would score the one thing training never saw.
    """
    preparer = make_preparer(
        tmp_path, keyword_expand_p=0.5, context_dropout=0.3, withhold_keyword="lifelink",
    )

    stream = random.Random(3)
    training = preparer.batcher(rng=stream)
    assert (training.stager.keyword_expand_p, training.context_dropout) == (0.5, 0.3)

    scoring = preparer.batcher(training=False)
    assert (scoring.stager.keyword_expand_p, scoring.context_dropout) == (0.0, 0.0)
    assert scoring.stager.withhold_keyword == "lifelink"
    # A training batcher draws from the step's own generator, and a scoring
    # one from a stream of its own, so how many validation batches ran
    # cannot change what any training step sees.
    assert training.rng is stream
    assert scoring.rng is not stream


def test_a_sample_missing_a_class_says_so(tmp_path, make_record, caplog):
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
    loop = make_training_loop(TrainEffectModelConfig(corpus="unused"))
    present = ValidationSamples.load(paths).present

    with caplog.at_level(logging.WARNING):
        loop.reporter.missing_classes(present)

    assert present == {"resolution-effect", "combat"}
    for absent in set(SAMPLING_CLASSES) - present:
        assert absent in caplog.text
    assert "no loss" in caplog.text


def test_a_sample_naming_every_class_says_nothing(caplog):
    loop = make_training_loop(TrainEffectModelConfig(corpus="unused"))

    with caplog.at_level(logging.WARNING):
        loop.reporter.missing_classes(frozenset(SAMPLING_CLASSES))

    assert caplog.text == ""


class TestExplainedDeviance:
    """The card-disjoint breakdown says how far each head beat a constant.

    A field's loss is a number in nats with no scale of its own, so the epoch
    line carries `1 - loss/floor` beside it: zero is the base rate, negative is
    worse than predicting it (FR-127c).
    """

    def test_each_field_carries_its_explained_deviance(self):
        line = format_parts(
            {"gate": 2.878, "damage_taken": 0.761},
            floor={"gate": 3.105, "damage_taken": 1.902},
        )
        assert line == "\n  fields: gate 2.878 (7%), damage_taken 0.761 (60%)"

    def test_the_largest_term_still_comes_first(self):
        line = format_parts(
            {"damage_taken": 0.761, "gate": 2.878},
            floor={"gate": 3.105, "damage_taken": 1.902},
        )
        assert line.index("gate") < line.index("damage_taken")

    def test_a_head_worse_than_the_constant_reads_negative(self):
        line = format_parts({"gate": 4.0}, floor={"gate": 2.0})
        assert "gate 4.000 (-100%)" in line

    def test_a_field_with_no_deviance_to_explain_reads_not_applicable(self):
        """A field whose targets never vary has nothing for a head to beat."""
        line = format_parts({"types_gained": 0.0}, floor={"types_gained": 0.0})
        assert "types_gained 0.000 (n/a)" in line

    def test_the_per_shard_training_breakdown_is_unchanged(self):
        assert format_parts({"gate": 2.0}) == "\n  fields: gate 2.000"

    def test_the_floor_is_logged_largest_first(self):
        assert format_floor(
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
    batch = make_entity_targets(
        gate=gate,
        fields=fields_everywhere({"damage_taken": torch.zeros(2, 2)}),
        mask=gate,
    )
    fields = (FIELDS_BY_NAME["damage_taken"],)

    floor = constant_predictor_floor([batch], fields=fields)
    line = format_parts({"damage_taken": 0.0067, "gate": 0.5}, floor)

    assert floor["damage_taken"] == 0.0
    assert "damage_taken 0.007 (n/a)" in line


def test_validation_scores_the_shipped_objective_not_the_backward_total():
    """The training-only heads join backward, never the number validation prints.

    Equal today only because a scoring step builds no training-only targets;
    summing ``total`` would make the two diverge the day one does.
    """
    from types import SimpleNamespace

    from effects.application.step_loss import StepLoss
    from effects.application.validation_pass import Validator

    loop = make_training_loop(TrainEffectModelConfig(corpus="unused", batch_size=2))
    validator = Validator(loop.config, loop.device, StubLosses(lambda prepared: StepLoss(
        total=torch.tensor(5.0), shipped=torch.tensor(1.0), parts=LossParts(), terms={},
    )))
    module = SimpleNamespace(eval=lambda: None, train=lambda: None)
    run = SimpleNamespace(
        encoder=module, model=module, step=0, scoring=None,
        preparer=SimpleNamespace(prepare_scoring_step=lambda *a, **k: SimpleNamespace()),
    )
    records = [SimpleNamespace(game_id=f"g{i}") for i in range(4)]

    assert validator.score(records, run, ()).loss == 1.0
