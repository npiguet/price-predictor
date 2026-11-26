"""Gradient accumulation counts the epoch's own steps (``--grad-accum``).

Each epoch drops a partial accumulation it began, so an optimizer step that
counted the run's global step would fall on the wrong micro-step whenever an
epoch's step count is not a multiple of the accumulation: the next epoch's
first update would carry fewer micro-steps than the rest.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from effects.application import training_loop as loop_module
from effects.application.run_reporter import EpochTally, ShardStats
from effects.application.training_loop import EpochScope
from effects.application.weight_average import WeightAverage
from tests.unit.effects.builders import StubLosses, make_cpu_training_loop


def test_every_epoch_steps_the_optimizer_on_its_own_accumulation_boundary(monkeypatch):
    loop = make_cpu_training_loop(monkeypatch, grad_accum=2, steps_per_epoch=4)
    weight = torch.nn.Parameter(torch.ones(()))
    loop.losses = StubLosses(lambda prepared: weight * 2.0)
    updates: list[tuple[int, int]] = []
    monkeypatch.setattr(
        loop_module, "_optimizer_step",
        lambda optimizer, tables=None: updates.append((epoch, tally.taken)) or {},
    )
    run = SimpleNamespace(
        step=0, encoder=None, model=torch.nn.Module(), optimizer=None, training=None,
        # Averages nothing: these tests read the optimizer step alone.
        average=WeightAverage({}, rate=1.0, start=1),
    )
    for epoch in (1, 2):
        tally = EpochTally(running=torch.zeros(()))
        for _ in range(3):  # an epoch shortened to an odd count
            loop._train_step(
                object(), EpochScope(run, tally, 1), ShardStats(loss=torch.zeros(())), False,
            )
    # Two micro-steps per update in both epochs; the third of each is dropped.
    assert updates == [(1, 1), (2, 1)]


def test_a_shard_ending_between_boundaries_still_reports_its_last_norms(monkeypatch):
    """The shard's last step reads back the norms of the last optimizer step,
    whether or not that step was itself an accumulation boundary."""
    loop = make_cpu_training_loop(monkeypatch, grad_accum=2, steps_per_epoch=4)
    weight = torch.nn.Parameter(torch.ones(()))
    loop.losses = StubLosses(lambda prepared: weight * 2.0)
    run = SimpleNamespace(
        step=0, encoder=None, model=torch.nn.Module(), training=None,
        # Averages nothing: these tests read the optimizer step alone.
        average=WeightAverage({}, rate=1.0, start=1),
        optimizer=torch.optim.SGD([{"params": [weight], "name": "head"}], lr=0.1),
    )
    tally = EpochTally(running=torch.zeros(()))
    stats = ShardStats(loss=torch.zeros(()))
    for index in range(3):  # shortened to an odd count
        loop._train_step(object(), EpochScope(run, tally, 1), stats, index == 2)
    assert set(stats.norms) == {"head"}


def test_an_odd_epoch_whose_last_step_is_no_optimizer_step_still_starts_the_average(
    monkeypatch,
):
    """Epoch 2 ends at step 6 = the start, on a dropped micro-step: the average
    starts from the weights at the epoch's end, before validation reads them."""
    loop = make_cpu_training_loop(monkeypatch, grad_accum=2, steps_per_epoch=4)
    weight = torch.nn.Parameter(torch.ones(()))
    loop.losses = StubLosses(lambda prepared: weight * 2.0)
    monkeypatch.setattr(loop_module, "_optimizer_step", lambda optimizer, tables=None: {})
    module = torch.nn.Module()
    module.weight = weight
    run = MagicMock(step=0, model=module, encoder=None)
    run.average = WeightAverage({"model": module}, rate=0.5, start=6)
    seen: list[bool] = []
    loop.validator = MagicMock()
    loop.validator.epoch.side_effect = lambda *args: (
        seen.append(run.average.started), MagicMock())[1]
    loop._save_epoch = MagicMock(return_value=False)
    monkeypatch.setattr(loop_module, "save_resume", lambda *args, **kwargs: None)
    loop.reporter = MagicMock()
    for epoch in (1, 2):
        loop._train_epoch = lambda epoch, run, tally: [
            loop._train_step(
                object(), EpochScope(run, tally, 1), ShardStats(loss=torch.zeros(())), False,
            )
            for _ in range(3)
        ]
        loop._run_epoch(epoch, run, None)
    assert seen == [False, True]
    assert run.average.start_step == 6
