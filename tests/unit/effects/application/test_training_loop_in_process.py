"""``--prefetch-workers 0``: the host half runs in the training process, in step order.

Plain and synchronous: a shard is read when its turn comes and each step is
prepared just before its device half runs, so one shard is the most ever
resident and an exception lands at the shard or step it belongs to. That the
run it trains is the one the worker processes train, bit for bit, is
``test_training_loop_workers.py``'s; what is left here is the order things
happen in, read off a clock with both halves stubbed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from effects.application import training_loop as loop_module
from effects.application.step_preparation import (
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.train_effect_model import (
    TrainEffectModelConfig,
    epoch_shards,
)
from effects.application.training_loop import EpochTally
from effects.domain.records import CombatPayload, RecordKind
from tests.unit.effects.builders import (
    EFFECTS_FIXTURES,
    GEN1_SIDECAR_FOLDERS,
    StubLosses,
    make_cpu_training_loop,
    make_preparer,
    make_training_loop,
    one_parameter_module,
    stub_below_the_loop,
    stub_validator,
    write_validation_samples,
)

#: Shards in the epoch, and the seed whose draw orders them.
SHARDS = 4
SEED = 7
#: What each stubbed half costs, so the clock can tell one from the next.
PAUSE = 0.01


def _mark(marks: list, what: str, which, phase: str) -> None:
    marks.append((what, which, phase, time.perf_counter()))


def _at(marks, what: str, which, phase: str) -> float:
    return next(when for kind, name, stage, when in marks
                if kind == what and name == which and stage == phase)


# ── the shards ──────────────────────────────────────────────────────────


@pytest.fixture
def epoch(tmp_path, make_record, monkeypatch):
    """Run one four-shard epoch end to end; what was read and trained, and when."""
    combat = [make_record(record_id=f"r{i}", game_id=f"g{i % 2}",
                          kind=RecordKind.COMBAT, payload=CombatPayload())
              for i in range(4)]
    samples = write_validation_samples(tmp_path, combat[:2])
    shards = [tmp_path / f"train{index}.jsonl.gz" for index in range(SHARDS)]

    real_load = loop_module.load_shard
    marks: list = []
    trained: list[tuple[str, list]] = []

    def start(*, failing: str | None = None) -> int:
        def load(path):
            path = Path(path)
            if path in set(samples.values()):
                return real_load(path)
            _mark(marks, "load", path.name, "start")
            if path.name == failing:
                raise RuntimeError(f"unreadable shard {path.name}")
            return [f"{path.name} records"]

        def train(prepared):
            name, records = prepared
            _mark(marks, "train", name, "start")
            trained.append((name, records))
            time.sleep(PAUSE)
            _mark(marks, "train", name, "finish")
            return model.p.sum()

        model, _encoder = stub_below_the_loop(monkeypatch)
        monkeypatch.setattr(loop_module, "load_shard", load)
        # The shard's records stand in for the one step they would prepare.
        monkeypatch.setattr(
            StepPreparer, "shard_steps",
            lambda self, task, records, pin=False: iter([
                ShardOpened(trainable=1, held_back=0, rarity_coverage=None),
                StepReady(
                    prepared=(task.shard.name, records), buckets={}, families={},
                ),
            ]),
        )
        config = TrainEffectModelConfig(
            corpus=str(tmp_path), epochs=1, steps_per_epoch=SHARDS,
            shards_per_epoch=SHARDS, batch_size=2, seed=SEED,
            model_output=tmp_path / "out", prefetch_workers=0,
        )
        loop = make_training_loop(
            config, training_shards=shards, validation_samples=samples,
        )
        loop.losses = StubLosses(train)
        stub_validator(loop)
        return loop.execute()

    drawn = epoch_shards(shards, epoch=1, per_epoch=SHARDS, seed=SEED)
    return SimpleNamespace(
        start=start, marks=marks, trained=trained,
        drawn=[path.name for path in drawn],
    )


def test_the_shards_are_trained_in_the_drawn_order_with_their_own_records(epoch):
    """The seeded draw is the whole of an epoch's composition, so a change that
    reordered it — or handed one shard's records to another's steps — would be
    a different corpus reported under the same seed."""
    assert epoch.start() == 0
    assert [name for name, _ in epoch.trained] == epoch.drawn
    assert [records for _, records in epoch.trained] == [
        [f"{name} records"] for name in epoch.drawn
    ]


def test_a_shard_is_read_only_once_its_predecessor_has_trained(epoch):
    """So the shard in hand is the only one resident."""
    epoch.start()
    for current, following in zip(epoch.drawn, epoch.drawn[1:]):
        assert _at(epoch.marks, "train", current, "finish") < _at(
            epoch.marks, "load", following, "start",
        )


def test_a_failing_read_surfaces_at_its_own_shard(epoch):
    """One that let the exception out early would attribute it to whatever
    shard was training, and one that swallowed it would train the epoch on
    three shards while reporting four."""
    with pytest.raises(RuntimeError, match=epoch.drawn[2]):
        epoch.start(failing=epoch.drawn[2])
    assert [name for name, _ in epoch.trained] == epoch.drawn[:2]


# ── the steps of one shard ──────────────────────────────────────────────


def _shard(monkeypatch, directory, *, failing_at: int | None = None):
    """Train one fixture shard with stubbed halves; what each did, and when."""
    marks: list = []
    model = one_parameter_module()

    def prepare(self, plan, *args, **kwargs):
        which = len([m for m in marks if m[0] == "prepare"])
        _mark(marks, "prepare", which, "start")
        if which == failing_at:
            raise RuntimeError(f"step {which} could not be prepared")
        return which, 0

    def device_loss(prepared):
        _mark(marks, "train", prepared, "start")
        time.sleep(PAUSE)
        return model.p.sum()

    monkeypatch.setattr(StepPreparer, "prepare_training_step", prepare)
    loop = make_cpu_training_loop(monkeypatch, batch_size=8)
    loop.losses = StubLosses(device_loss)
    preparer = make_preparer(directory, cards_folders=GEN1_SIDECAR_FOLDERS, batch_size=8)
    task = ShardTask(
        shard=EFFECTS_FIXTURES / "gen1-records.jsonl.gz", epoch=1, position=1,
        budget=4, fields=(),
    )
    run = SimpleNamespace(
        step=0, warmup=1, encoder=one_parameter_module(), model=model, training=None,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
    )
    loop._train_on_shard(
        task, loop._local_steps(task, preparer), run,
        EpochTally(running=torch.zeros(())), of=1,
    )
    return marks


def test_each_step_is_prepared_just_before_it_trains(monkeypatch, tmp_path):
    marks = _shard(monkeypatch, tmp_path)
    assert [(kind, which) for kind, which, _, _ in marks] == [
        (kind, which) for which in range(4) for kind in ("prepare", "train")
    ]


def test_a_failing_preparation_surfaces_at_its_own_step(monkeypatch, tmp_path, caplog):
    with (
        caplog.at_level(logging.ERROR, logger="effects.application.training_loop"),
        pytest.raises(RuntimeError, match="step 2"),
    ):
        _shard(monkeypatch, tmp_path, failing_at=2)
    assert any("preparing it failed" in message for message in caplog.messages)


def test_the_shard_line_reports_the_batch_wait(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="effects.application.training_loop"):
        _shard(monkeypatch, tmp_path)
    assert any("batch wait" in message for message in caplog.messages)
