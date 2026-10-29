"""Gradient accumulation counts the epoch's own steps (``--grad-accum``).

Each epoch drops a partial accumulation it began, so an optimizer step that
counted the run's global step would fall on the wrong micro-step whenever an
epoch's step count is not a multiple of the accumulation: the next epoch's
first update would carry fewer micro-steps than the rest.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from effects.application import training_loop as loop_module
from effects.application.training_loop import EpochTally, ShardStats
from tests.unit.effects.builders import StubLosses, make_cpu_training_loop


def test_every_epoch_steps_the_optimizer_on_its_own_accumulation_boundary(monkeypatch):
    loop = make_cpu_training_loop(monkeypatch, grad_accum=2, steps_per_epoch=3)
    weight = torch.nn.Parameter(torch.ones(()))
    loop.losses = StubLosses(lambda prepared: weight * 2.0)
    updates: list[tuple[int, int]] = []
    monkeypatch.setattr(
        loop_module, "_optimizer_step",
        lambda optimizer: updates.append((epoch, tally.taken)) or {},
    )
    run = SimpleNamespace(
        step=0, encoder=None, model=None, optimizer=None, training=None,
    )
    for epoch in (1, 2):
        tally = EpochTally(running=torch.zeros(()))
        for _ in range(3):
            loop._train_step(
                object(), run, tally, ShardStats(loss=torch.zeros(())), False,
            )
    # Two micro-steps per update in both epochs; the third of each is dropped.
    assert updates == [(1, 1), (2, 1)]


def test_a_shard_ending_between_boundaries_still_reports_its_last_norms(monkeypatch):
    """The shard's last step reads back the norms of the last optimizer step,
    whether or not that step was itself an accumulation boundary."""
    loop = make_cpu_training_loop(monkeypatch, grad_accum=2, steps_per_epoch=3)
    weight = torch.nn.Parameter(torch.ones(()))
    loop.losses = StubLosses(lambda prepared: weight * 2.0)
    run = SimpleNamespace(
        step=0, encoder=None, model=None, training=None,
        optimizer=torch.optim.SGD([{"params": [weight], "name": "head"}], lr=0.1),
    )
    tally = EpochTally(running=torch.zeros(()))
    stats = ShardStats(loss=torch.zeros(()))
    for index in range(3):
        loop._train_step(object(), run, tally, stats, index == 2)
    assert set(stats.norms) == {"head"}
