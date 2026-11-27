"""Two real epochs of training on the gen-3 fixture records, end to end.

What ``test_training_loop_workers.py`` and ``test_curriculum_resume.py`` run:
every loss, the final weights, the averaged weights and the log lines a run
leaves, for a :class:`RunPlan` that names the worker count, the seed, where to
interrupt and whether to resume.
"""

from __future__ import annotations

import contextlib
import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from effects.application import run_context
from effects.application import training_loop as loop_module
from effects.application.step_loss import StepLossComputer
from effects.domain.ability_encoder import AbilityEncoder
from effects.domain.effect_model import EffectModel
from effects.domain.split import CorpusSplit
from tests.unit.effects.builders import FIXTURE_TRAINING_SEED, curated_fixture_records
from tests.unit.effects.selection_builders import FIXTURE_RULE_OUTPUTS, fixture_selection
from tests.unit.effects.training_builders import (
    LoopCorpus,
    fixture_training_config,
    make_training_loop,
)

SEED = FIXTURE_TRAINING_SEED


@functools.cache
def every_record_selection():
    """The control score's selection set over every fixture record: the
    validation sample alone holds too few games for every component."""
    return fixture_selection(curated_fixture_records())


class Interrupted(Exception):
    """Stands in for the machine turning off after an epoch."""


class RecordedLosses(StepLossComputer):
    """The real device half, noting every loss it computes, training's and
    validation's, and pressing Ctrl+C on the ``interrupt_at``-th."""

    def __init__(self, config, device, sink: list[float], interrupt_at) -> None:
        super().__init__(config, device)
        self.sink = sink
        self.interrupt_at = interrupt_at
        self.calls = 0

    def device_loss(self, *args, **kwargs):
        self.calls += 1
        if self.calls == self.interrupt_at:
            raise KeyboardInterrupt
        computed = super().device_loss(*args, **kwargs)
        self.sink.append(float(computed.total.detach()))
        return computed


def record_losses(loop, sink: list[float], *, interrupt_at=None) -> None:
    """Swap ``loop``'s device half, training's and validation's, for one that
    records into ``sink``."""
    recorded = RecordedLosses(loop.config, loop.device, sink, interrupt_at)
    loop.losses = loop.validator.losses = recorded


@dataclass(frozen=True)
class RunPlan:
    """What one run is asked to do.

    ``interrupt_after`` stops the run once that epoch's resume file is written,
    as a machine turning off would; ``resume`` continues from that file.
    ``interrupt_at`` presses Ctrl+C on that device loss, counted from 1.
    ``config`` holds the training flags that differ from the run's defaults.
    """

    workers: int
    seed: int = SEED
    interrupt_after: int | None = None
    resume: bool = False
    interrupt_at: int | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


class BuiltModules:
    """The encoder and model a run builds, and the weights the model started with."""

    def __init__(self) -> None:
        self.modules: dict[str, torch.nn.Module] = {}
        self.initial: dict[str, torch.Tensor] = {}

    def encoder(self, config):
        self.modules["encoder"] = AbilityEncoder(config)
        return self.modules["encoder"]

    def model(self, config):
        # The trunk at a size a unit suite can train on a CPU; the workers
        # never build it, so the patch reaches every run alike.
        self.modules["model"] = EffectModel(replace(
            config, d_model=32, n_layers=1, n_heads=2, ff_dim=64,
        ))
        self.initial.update({
            name: value.detach().clone()
            for name, value in self.modules["model"].state_dict().items()
        })
        return self.modules["model"]

    def parameters(self) -> dict[str, torch.Tensor]:
        return {
            f"{owner}.{name}": value.detach().clone()
            for owner, module in self.modules.items()
            for name, value in module.state_dict().items()
        }


def _stop_after_epoch(monkeypatch, epoch: int) -> None:
    """Make the run raise ``Interrupted`` once ``epoch``'s resume file is written."""
    real_save = loop_module.save_resume

    def save_then_stop(run, saved_epoch, **kwargs):
        real_save(run, saved_epoch, **kwargs)
        if saved_epoch == epoch:
            raise Interrupted

    monkeypatch.setattr(loop_module, "save_resume", save_then_stop)


def _execute(loop) -> int | None:
    """The loop's exit code, None when the run was interrupted."""
    try:
        return loop.execute()
    except Interrupted:
        return None


def _resume_part(out: Path, key: str):
    """``key`` of the last resume file ``out`` holds, None without one."""
    from effects.infrastructure.effect_model_store import load_resume_state

    return load_resume_state(out)[key] if (out / "resume.pt").exists() else None


@dataclass(frozen=True)
class TrainingRunner:
    """What a run needs of the test: the patcher, the corpus, the log."""

    monkeypatch: pytest.MonkeyPatch
    corpus: Any
    caplog: Any

    def run(self, out: Path, plan: RunPlan) -> dict:
        """Two epochs end to end; every loss, the final weights, the share lines."""
        self.monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if plan.interrupt_after is not None:
            _stop_after_epoch(self.monkeypatch, plan.interrupt_after)
        built = BuiltModules()
        losses: list[float] = []
        self.monkeypatch.setattr(run_context, "AbilityEncoder", built.encoder)
        self.monkeypatch.setattr(run_context, "EffectModel", built.model)
        self.caplog.clear()
        with self.caplog.at_level(logging.INFO, logger="effects.application"):
            loop = self._loop(out, plan)
            record_losses(loop, losses, interrupt_at=plan.interrupt_at)
            code = _execute(loop)
        self.monkeypatch.undo()
        assert code in (0, None)
        return self._outcome(out, losses, built)

    def _loop(self, out: Path, plan: RunPlan):
        # Gradient accumulation (6 steps, an even count, so each epoch ends on an
        # optimizer step) and lethality copies on, so the equalities cover
        # the micro-batch sum and the counterfactual draws.
        config = {"grad_accum": 2, "lethality_copies": 2, "steps_per_epoch": 6, **plan.config}
        return make_training_loop(
            fixture_training_config(
                self.corpus, out, prefetch_workers=plan.workers, seed=plan.seed,
                resume=plan.resume, **config,
            ),
            LoopCorpus(
                training_shards=self.corpus.shards,
                validation_samples={
                    "card-disjoint": self.corpus.sample, "game-disjoint": self.corpus.sample,
                },
                selection=every_record_selection(),
            ),
            split=CorpusSplit(
                held_out_cards=(), card_disjoint_games=frozenset(),
                game_disjoint_games=frozenset({self.corpus.held}),
            ),
            rule_outputs=FIXTURE_RULE_OUTPUTS,
        )

    def _outcome(self, out: Path, losses: list[float], built: BuiltModules) -> dict:
        messages = self.caplog.messages
        return {
            # The averaged weights the last resume file holds, None without one.
            "average": _resume_part(out, "weight_average"),
            "stopper": _resume_part(out, "stopper"),
            "control": [m for m in messages if "| best " in m],
            "losses": losses,
            "parameters": built.parameters(),
            "shares": [m for m in messages if "by rarity bucket" in m],
            "workers": any("worker process" in m for m in messages),
            "initial": built.initial,
        }


class ListHandler(logging.Handler):
    def __init__(self, sink: list[str]) -> None:
        super().__init__()
        self.sink = sink

    def emit(self, record) -> None:
        self.sink.append(record.getMessage())


class Capture:
    """``caplog`` for a module-scoped fixture, which cannot request it."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._handler = ListHandler(self.messages)

    def clear(self) -> None:
        self.messages.clear()

    def at_level(self, level, logger):
        @contextlib.contextmanager
        def scope():
            target = logging.getLogger(logger)
            previous = target.level
            target.setLevel(level)
            target.addHandler(self._handler)
            try:
                yield
            finally:
                target.removeHandler(self._handler)
                target.setLevel(previous)

        return scope()
