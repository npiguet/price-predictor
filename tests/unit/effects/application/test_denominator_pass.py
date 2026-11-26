"""The loss denominators' startup pre-pass (FR-062, research T3, T144): its
sample is prepared as training steps are (finding 15.4)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from effects.application.denominator_pass import (
    DENOMINATOR_EPOCH,
    DENOMINATOR_STEPS,
    denominator_shards,
    denominator_tasks,
    loss_denominators,
)
from effects.application.run_context import preparer_settings
from effects.application.step_preparation import ShardOpened, StepPreparer, StepReady
from effects.application.train_effect_model import TrainEffectModelConfig
from effects.domain.effect_head_layout import FIELDS_BY_NAME, PER_ENTITY_FIELDS
from effects.domain.loss_denominators import LossDenominators
from effects.domain.loss_floor import PER_ENTITY, TargetSample, constant_predictor_losses
from tests.unit.effects.builders import (
    FIXTURE_TRAINING_SEED,
    fields_everywhere,
    fixture_training_config,
    make_entity_targets,
    make_training_loop,
)
from tests.unit.effects.cli_args import parse_cli

FIELDS = (FIELDS_BY_NAME["damage_taken"], FIELDS_BY_NAME["tap_state"])


def _targets(records):
    damage = torch.tensor([[float(amount)] for amount in records])
    tapped = torch.tensor([[float(amount > 2)] for amount in records])
    return make_entity_targets(
        torch.ones(len(records), 1),
        fields_everywhere({"damage_taken": damage, "tap_state": tapped}),
    )


class _Preparer:
    """Prepares a shard's records, here damage amounts, as training steps of
    two records each, and remembers each task it was handed. Its training-only
    head's targets are no tuple a floor could read."""

    heads = frozenset({"value"})

    def __init__(self) -> None:
        self.tasks: list = []

    def shard_steps(self, task, records):
        self.tasks.append(task)
        yield ShardOpened(trainable=len(records), rarity_coverage=None)
        for index in range(task.budget):
            batch = [records[(2 * index + offset) % len(records)] for offset in (0, 1)]
            step = SimpleNamespace(targets=_targets(batch), heads={"value": "not a tuple"})
            yield StepReady(step, {}, {})


SHARDS = {
    Path("a.jsonl.gz"): [0, 0, 1, 5, 0],
    Path("b.jsonl.gz"): [3, 0, 0],
}


def _tasks(budget: int) -> list:
    return [
        SimpleNamespace(
            shard=shard, epoch=DENOMINATOR_EPOCH, position=position, budget=budget,
            fields=FIELDS,
        )
        for position, shard in enumerate(SHARDS, start=1)
    ]


def test_the_denominators_are_the_floored_constant_losses_of_the_steps():
    preparer = _Preparer()
    tasks = _tasks(budget=3)
    got = loss_denominators(
        preparer, tasks, floor=0.04, read=SHARDS.__getitem__,
    )
    assert preparer.tasks == tasks

    sample = TargetSample()
    for records in SHARDS.values():
        for index in range(3):
            sample.add(_targets([records[(2 * index + o) % len(records)] for o in (0, 1)]))
    expected = LossDenominators.of(constant_predictor_losses(sample, fields=FIELDS), floor=0.04)
    assert got == expected
    assert got.of_term(PER_ENTITY, "damage_taken") > 0.04


def test_a_term_below_the_floor_is_raised_to_it():
    """A field the sample's constant already predicts to 0.01 nats or less."""
    got = loss_denominators(
        _Preparer(), [SimpleNamespace(shard=Path("quiet"), budget=100, fields=FIELDS)],
        floor=0.5,
        read=lambda path: [0] * 199 + [1],
    )
    assert got.of_term(PER_ENTITY, "damage_taken") == 0.5
    assert got.of_term(PER_ENTITY, "tap_state") == 0.5


def _default_tasks():
    """The pre-pass's tasks at the defaults, over 1,000 shards."""
    shards = [Path(f"shard-{index:04d}.jsonl.gz") for index in range(1000)]
    return shards, denominator_tasks(
        shards, seed=11, steps_per_epoch=5000, shards_per_epoch=256,
    )


def test_the_tasks_read_a_shard_for_the_steps_an_epoch_gives_it():
    """At the defaults (5,000 steps over 256 shards, 20 a shard at most) the
    pre-pass reads 39 shards for 20 steps each, ``DENOMINATOR_STEPS`` or just
    past it."""
    shards, tasks = _default_tasks()
    assert len(tasks) == 39
    assert {task.budget for task in tasks} == {20}
    assert DENOMINATOR_STEPS <= sum(task.budget for task in tasks) < DENOMINATOR_STEPS + 20
    assert [task.shard for task in tasks] == denominator_shards(shards, seed=11, count=39)


def test_the_tasks_are_keyed_by_an_epoch_no_training_epoch_has():
    _, tasks = _default_tasks()
    assert [task.position for task in tasks] == list(range(1, 40))
    assert {task.epoch for task in tasks} == {DENOMINATOR_EPOCH} == {0}
    assert {task.fields for task in tasks} == {PER_ENTITY_FIELDS}


def test_a_shard_s_budget_never_exceeds_the_pre_pass():
    """One shard an epoch: the pre-pass reads it for ``DENOMINATOR_STEPS``
    steps, not the epoch's 5,000; with fewer shards than it wants, all."""
    shards = [Path(f"shard-{index}.jsonl.gz") for index in range(3)]
    one = denominator_tasks(shards, seed=11, steps_per_epoch=5000, shards_per_epoch=1)
    assert [task.budget for task in one] == [DENOMINATOR_STEPS]
    few = denominator_tasks(shards, seed=11, steps_per_epoch=7, shards_per_epoch=3)
    assert len(few) == 3 and {task.budget for task in few} == {3}
    assert denominator_tasks([], seed=11, steps_per_epoch=7, shards_per_epoch=3) == []


def test_the_pre_pass_draws_its_shards_by_the_run_s_seed():
    shards = [Path(f"shard-{index:03d}.jsonl.gz") for index in range(40)]
    drawn = denominator_shards(shards, seed=11, count=8)
    assert len(drawn) == 8 and set(drawn) <= set(shards)
    assert denominator_shards(list(reversed(shards)), seed=11, count=8) == drawn
    assert denominator_shards(shards, seed=12, count=8) != drawn
    assert denominator_shards(shards[:3], seed=11, count=8) == sorted(shards[:3])


def _fixture_denominators(corpus, out, caplog, **overrides):
    """The pre-pass over the fixture corpus by a real preparer, and the
    record count its log line reports."""
    config = fixture_training_config(corpus, out, prefetch_workers=0, **overrides)
    preparer = StepPreparer(preparer_settings(make_training_loop(config).spec))
    tasks = denominator_tasks(
        corpus.shards, seed=config.seed, steps_per_epoch=config.steps_per_epoch,
        shards_per_epoch=config.shards_per_epoch,
    )
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="effects.application"):
        got = loss_denominators(
            preparer, tasks, floor=config.deviance_floor,
        )
    line = next(m for m in caplog.messages if "Loss denominators from" in m)
    counted = int(re.search(r"(\d+) records", line).group(1))
    return got, counted, sum(task.budget for task in tasks) * config.batch_size


def test_the_sample_holds_the_lethality_copies_training_steps_hold(
    monkeypatch, corpus, tmp_path, caplog,
):
    """Finding 15.4: under ``--lethality-copies`` the zone and damage terms
    are divided by a constant measured on the label mix the run trains."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    plain, plain_records, planned = _fixture_denominators(corpus, tmp_path, caplog)
    copied, copied_records, _ = _fixture_denominators(
        corpus, tmp_path, caplog, lethality_copies=2,
    )
    assert plain_records == planned < copied_records
    assert copied != plain


def test_the_sample_is_drawn_by_the_run_s_seed(monkeypatch, corpus, tmp_path, caplog):
    """Seeded by the step's place, so a resume or any worker count would
    draw it again; another seed draws another."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    first, _, _ = _fixture_denominators(corpus, tmp_path, caplog)
    again, _, _ = _fixture_denominators(corpus, tmp_path, caplog)
    other, _, _ = _fixture_denominators(
        corpus, tmp_path, caplog, seed=FIXTURE_TRAINING_SEED + 1,
    )
    assert first == again != other


def test_the_deviance_floor_flag_defaults_to_four_hundredths_of_a_nat():
    parsed = parse_cli(["train-effect-model", "--corpus", "c"])
    assert parsed.deviance_floor == 0.04
    assert parse_cli(
        ["train-effect-model", "--corpus", "c", "--deviance-floor", "0.1"],
    ).deviance_floor == 0.1


def test_a_floor_of_zero_or_less_is_refused():
    with pytest.raises(ValueError, match="deviance-floor"):
        TrainEffectModelConfig.unpathed(corpus="c", deviance_floor=0.0)
