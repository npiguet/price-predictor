"""The training loop: the optimizer, the epochs and the shards between steps.

The schedule is ``training_schedule``'s pure functions and corpus reading
``train_effect_model``'s; each step's host half is a ``StepPreparer``'s and its device half a
``StepLossComputer``'s, validation is a ``Validator``'s and every line between
steps a ``RunReporter``'s. An epoch is a step count, not a pass over the
corpus, and walks ``--shards-per-epoch`` shards one at a time. The best
checkpoint is chosen by the card-disjoint validation loss.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from effects.application.resume_state import restore_resume, save_resume
from effects.application.run_context import (
    RunContext,
    build_run,
    preparer_settings,
)
from effects.application.run_reporter import EpochTally, RunReporter, ShardStats
from effects.application.step_loss import StepLossComputer, read_back
from effects.application.step_preparation import (
    DeviceStep,
    PreparerSettings,
    ShardClosed,
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.step_workers import StepWorkerPool
from effects.application.surface_batching import IDENTITY_TABLE_KEY
from effects.application.train_effect_model import (
    CorpusFacts,
    MissingTrainingInputError,
    RunSpec,
    TrainEffectModelConfig,
    check_holdout,
)
from effects.application.training_schedule import (
    MAX_GRAD_NORM,
    Curriculum,
    epoch_shards,
    learning_rate_at,
    steps_per_shard,
)
from effects.application.validation_pass import ValidationSamples, Validator
from effects.domain.effect_head_input import feature_widths
from effects.infrastructure.effect_model_store import (
    EPOCHS_DIR,
    EffectCheckpoint,
    EffectModelStore,
)
from effects.infrastructure.shard_io import load_shard
from effects.infrastructure.torch_device import default_device
from price_predictor.infrastructure.torch_training import clip_per_group

logger = logging.getLogger(__name__)

#: What a shard that never opened reports: no records, so no line of steps.
UNOPENED = ShardOpened(trainable=0, rarity_coverage=None)

#: The most prepared steps one worker process may hold queued: room for one
#: shard's steps, capped to bound the shared memory of a run with long shards.
MAX_STEPS_AHEAD = 32

#: Cap on the fraction of CUDA memory the caching allocator may reserve
#: (FR-095). On Windows the driver pages excess reservation out to host RAM
#: rather than raising out-of-memory, so an unbounded allocator makes the run
#: slow instead of failing.
CUDA_MEMORY_FRACTION = 0.9


class EpochPlan(NamedTuple):
    """One epoch's shard tasks in the drawn order, and how many shards were
    drawn: a shard given no steps gets no task."""

    drawn: int
    tasks: list[ShardTask]


def reports_now(*, index: int, budget: int) -> bool:
    """Whether the step at zero-based ``index`` reads its numbers back: only a
    shard's last, since each read is a device synchronization."""
    return index + 1 >= budget


def local_steps(task: ShardTask, preparer: StepPreparer, *, pin: bool) -> Iterator:
    """One shard's messages prepared in this process, the shard read on the
    first message and released with the generator."""
    yield from preparer.shard_steps(task, load_shard(task.shard), pin=pin)


def _timed(steps, task: ShardTask, of: int):
    """Each of a shard's messages with how long it was waited for."""
    while True:
        waiting = time.perf_counter()
        try:
            message = next(steps, None)
        except Exception:
            logger.error(
                "epoch %d | shard %d/%d %s | preparing it failed",
                task.epoch, task.position, of, Path(task.shard).name,
            )
            raise
        if message is None:
            return
        yield message, time.perf_counter() - waiting


def _optimizer_step(optimizer) -> dict[str, torch.Tensor]:
    """Clip each group, step and zero; the pre-clip norms unread, since each
    read stalls until the GPU drains."""
    norms = clip_per_group(optimizer, max_norm=MAX_GRAD_NORM, read=False)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norms


class TrainingLoop:
    """Owns one training run end to end: builds it, then trains, validates,
    reports and saves each epoch through its three collaborators."""

    def __init__(
        self,
        config: TrainEffectModelConfig,
        *,
        facts: CorpusFacts,
        training_shards: Sequence[Path],
        validation_samples: Mapping[str, Path],
    ) -> None:
        self.config = config
        #: Processes preparing steps; 0 prepares them in this one. Not a
        #: training setting: it moves where the host half runs, not what it
        #: computes.
        self.prefetch_workers = max(int(config.prefetch_workers), 0)
        self.training_shards = list(training_shards)
        self.validation_samples = dict(validation_samples)
        self.spec = RunSpec(
            config, facts,
            config.seed if config.seed is not None
            else random.SystemRandom().getrandbits(32),
        )
        torch.manual_seed(self.spec.seed)
        self.device = default_device()
        #: The device half of every step, training's and validation's.
        self.losses = StepLossComputer(config, self.device)
        self.validator = Validator(config, self.device, self.losses)
        self.reporter = RunReporter(config, facts)
        self.curriculum = Curriculum(config.curriculum_epoch)

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def execute(self) -> int:
        samples = ValidationSamples.load(self.validation_samples)
        self.reporter.missing_classes(samples.present)
        if not samples.probe:
            raise MissingTrainingInputError(
                "The corpus's validation samples hold no records, so there is "
                "nothing to measure feature widths from."
            )
        self.reporter.run_start(self.spec.seed, samples, shards=len(self.training_shards))
        # Raises on an empty card-disjoint stratum (FR-088b); the gate-1 slice
        # is the build's count, not this run's.
        self.reporter.thin_holdout(check_holdout(
            card_disjoint_records=len(samples.card_disjoint),
            unique_text_records=self.spec.facts.gate_one_records,
        ))
        self._cap_cuda_memory()
        settings = preparer_settings(self.spec, widths=feature_widths(samples.probe))
        run = build_run(
            self.spec, StepPreparer(settings, truncations=self.reporter.truncations),
            samples.present, device=self.device,
        )
        first_epoch = 1
        if self.config.resume:
            first_epoch = restore_resume(run)
            if first_epoch is None:
                return 0

        run.pool = self._start_workers(settings, run, first_epoch=first_epoch)
        try:
            for epoch in range(first_epoch, self.config.epochs + 1):
                if self._run_epoch(epoch, run, samples):
                    break
        finally:
            if run.pool is not None:
                run.pool.close()
        return 0

    def _cap_cuda_memory(self) -> None:
        if self.is_cuda:
            torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)

    def _run_epoch(
        self, epoch: int, run: RunContext, samples: ValidationSamples,
    ) -> bool:
        """Train, validate, report and save one epoch; True when it stops the run."""
        # Drops a partial --grad-accum accumulation the last epoch left staged,
        # so it never crosses into a different objective.
        run.optimizer.zero_grad(set_to_none=True)
        tally = EpochTally(running=torch.zeros((), device=self.device))
        if self.curriculum.resets_at(epoch):
            run.stopper.reset()
            self.reporter.curriculum(epoch, len(run.fields[epoch]))
        self._train_epoch(epoch, run, tally)
        card, game = self.validator.epoch(run, samples, run.fields[epoch])
        self.reporter.epoch(epoch, tally, card, game, self.validator.floor.floor)
        self.reporter.unresolved(run.preparer)
        return self._close_epoch(epoch, run, card.loss)

    def _train_epoch(self, epoch: int, run: RunContext, tally: EpochTally) -> None:
        plan = self._epoch_plan(epoch, run)
        for task in plan.tasks:
            steps = (
                local_steps(task, run.preparer, pin=self.is_cuda)
                if run.pool is None else run.pool.shard()
            )
            self._train_on_shard(task, steps, run, tally, of=plan.drawn)
            self._release_cache()
        if run.pool is not None and epoch < self.config.epochs:
            # Queued before validation so the workers prepare while this
            # process validates; nothing validation decides changes them.
            for task in self._epoch_plan(epoch + 1, run).tasks:
                run.pool.submit(task)

    def _close_epoch(self, epoch: int, run: RunContext, card_disjoint_loss: float) -> bool:
        """Save what the epoch earned and its resume state; True to stop early."""
        stopper = run.stopper
        best = stopper.update(card_disjoint_loss)
        if best or self.config.save_every_epoch:
            checkpoint = self._checkpoint(epoch, run)
            output = self.config.resolved_model_output()
            if best:
                EffectModelStore(output).save(checkpoint)
            if self.config.save_every_epoch:
                EffectModelStore(output / EPOCHS_DIR).save(checkpoint)
        save_resume(run, epoch, best=best)
        if stopper.should_stop:
            self.reporter.early_stop()
        return stopper.should_stop

    def _checkpoint(self, epoch: int, run: RunContext) -> EffectCheckpoint:
        return EffectCheckpoint(
            encoder_config=run.encoder_config,
            model_config=run.model_config,
            encoder_state=run.encoder.state_dict(),
            model_state=run.model.state_dict(),
            provenance=self.spec.facts.provenance(self.config),
            variant=self.config.variant,
            best_val_loss=run.stopper.best,
            epoch=epoch,
            extra=(
                {IDENTITY_TABLE_KEY: run.identity_table.state()}
                if run.identity_table is not None else {}
            ),
            cards_folders=tuple(str(folder) for folder in self.config.cards_folders),
            training_settings=dict(run.training_settings),
        )

    def _epoch_plan(self, epoch: int, run: RunContext) -> EpochPlan:
        """One epoch's plan. Every task keeps its drawn ``position``, which
        seeds its generators, whether or not the shards before it got steps."""
        shards = epoch_shards(
            self.training_shards, epoch=epoch,
            per_epoch=self.config.shards_per_epoch, seed=self.spec.seed,
        )
        allocation = steps_per_shard(self.config.steps_per_epoch, len(shards))
        return EpochPlan(len(shards), [
            ShardTask(
                shard=Path(shard), epoch=epoch, position=position,
                budget=budget, fields=run.fields[epoch],
            )
            for position, (shard, budget) in enumerate(
                zip(shards, allocation), start=1,
            )
            if budget > 0
        ])

    def _start_workers(
        self, settings: PreparerSettings, run: RunContext, *, first_epoch: int,
    ) -> StepWorkerPool | None:
        """The step-preparing processes, with ``first_epoch`` queued; None for
        none. Each may hold one shard's steps ahead, or the workers whose
        shards come later would fill their queues and stop."""
        if self.prefetch_workers <= 0 or not self.training_shards:
            return None
        drawn = min(self.config.shards_per_epoch, len(self.training_shards))
        depth = min(
            max(steps_per_shard(self.config.steps_per_epoch, drawn), default=1),
            MAX_STEPS_AHEAD,
        )
        self.reporter.workers(self.prefetch_workers, depth)
        pool = StepWorkerPool(
            settings, workers=self.prefetch_workers, depth=max(depth, 1),
            pin=self.is_cuda,
        )
        try:
            for task in self._epoch_plan(first_epoch, run).tasks:
                pool.submit(task)
        except BaseException:
            pool.close()
            raise
        return pool

    def _release_cache(self) -> None:
        """Release the CUDA allocator's cache between shards: one shard's
        cached blocks rarely fit the next one's batch shapes."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def _train_on_shard(
        self, task: ShardTask, steps: Iterator, run: RunContext,
        tally: EpochTally, *, of: int,
    ) -> None:
        """Take ``task.budget`` steps on one shard.

        ``steps`` yields a ``ShardOpened``, each step's ``StepReady``, and
        from a worker a ``ShardClosed``. The line's ``wait`` is the time to
        the opening message; ``batch wait`` is the steps' wait for their
        preparation.
        """
        stats = ShardStats(loss=torch.zeros((), device=self.device))
        started = time.perf_counter()
        opened = UNOPENED
        index = 0
        try:
            for message, elapsed in _timed(steps, task, of):
                match message:
                    case ShardOpened():
                        stats.waited += elapsed
                        started = time.perf_counter()
                        opened = message
                        self.reporter.shard_opened(message)
                    case ShardClosed():
                        self.reporter.shard_closed(message)
                    case StepReady():
                        stats.batch_waited += elapsed
                        stats.learning_rate = self._set_learning_rate(run)
                        tally.add_shares(message)
                        # Decided before the forward pass, which is when the
                        # per-field decomposition has to be asked for.
                        due = reports_now(index=index, budget=task.budget)
                        index += 1
                        self._train_step(message.prepared, run, tally, stats, due)
                    case _:
                        raise TypeError(f"unexpected shard message {type(message).__name__}")
        finally:
            steps.close()
        self.reporter.shard(task, of, opened, stats, trained=time.perf_counter() - started)

    def _set_learning_rate(self, run: RunContext) -> float:
        rate = learning_rate_at(run.step, warmup=run.warmup, scale=self.config.lr_scale)
        for group in run.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def _train_step(
        self, prepared: DeviceStep, run: RunContext, tally: EpochTally,
        stats: ShardStats, due: bool,
    ) -> None:
        """One step's device half: forward, backward, and the optimizer on the
        accumulation boundary."""
        computed = self.losses.device_loss(
            prepared, run, run.training, step=run.step, report_parts=due,
        )
        (computed.total / self.config.grad_accum).backward()
        self.losses.mlm_backward(computed, run.encoder, run.model)
        # Counted within the epoch, not by run.step: each epoch drops any
        # partial accumulation, so the run's step would misplace boundaries.
        if (tally.taken + 1) % self.config.grad_accum == 0:
            stats.unread_norms = _optimizer_step(run.optimizer)
        if due:
            stats.norms = read_back(stats.unread_norms)
        tally.add_step(computed)
        stats.add_step(computed, reported=due)
        run.step += 1
