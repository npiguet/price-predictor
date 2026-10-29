"""The training loop: each step's device half, the optimizer, and validation.

The schedule and corpus reading are pure functions in ``train_effect_model``;
each step's host half is a ``StepPreparer``'s. An epoch is a step count, not a
pass over the corpus, and walks ``--shards-per-epoch`` shards one at a time.
The best checkpoint is chosen by the card-disjoint validation loss.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch

from effects.application.resume_state import restore_resume, save_resume
from effects.application.run_context import (
    RunContext,
    build_run,
    preparer_settings,
)
from effects.application.step_loss import StepLoss, StepLossComputer, read_back
from effects.application.step_preparation import (
    NO_TEXT_BUCKET,
    DeviceStep,
    PreparerSettings,
    ShardClosed,
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
)
from effects.application.surface_batching import (
    IDENTITY_TABLE_KEY,
)
from effects.application.train_effect_model import (
    MAX_GRAD_NORM,
    CorpusFacts,
    EpochResult,
    TrainEffectModelConfig,
    check_holdout,
    epoch_shards,
    learning_rate_at,
    load_shard,
    steps_per_shard,
)
from effects.application.training_log import (
    format_norms,
    format_parts,
    format_shares,
    format_terms,
)
from effects.application.validation_pass import ValidationSamples, Validator
from effects.domain.ability_encoder import (
    TruncationLog,
    log_truncation,
)
from effects.domain.effect_head_input import (
    feature_widths,
)
from effects.domain.rarity import RARITY_BUCKETS
from effects.domain.sampling import SAMPLING_CLASSES
from effects.infrastructure.effect_model_store import (
    EPOCHS_DIR,
    EffectCheckpoint,
    EffectModelStore,
)
from effects.infrastructure.step_workers import StepWorkerPool
from effects.infrastructure.torch_device import default_device
from price_predictor.infrastructure.torch_training import clip_per_group

logger = logging.getLogger(__name__)

#: The most prepared steps one worker process may hold queued: room for one
#: shard's steps, capped to bound the shared memory of a run with long shards.
MAX_STEPS_AHEAD = 32

#: Cap on the fraction of CUDA memory the caching allocator may reserve
#: (FR-095). On Windows the driver pages excess reservation out to host RAM
#: rather than raising out-of-memory, so an unbounded allocator makes the run
#: slow instead of failing.
CUDA_MEMORY_FRACTION = 0.9


def reports_now(*, index: int, budget: int) -> bool:
    """Whether the step at zero-based ``index`` reads its numbers back: only a
    shard's last, since each read is a device synchronization."""
    return index + 1 >= budget


def resets_at(config, *, epoch: int) -> bool:
    """Whether the early stopper forgets its best at the start of ``epoch``:
    at the curriculum epoch, where the objective gains the sparse fields."""
    return config.curriculum_epoch > 1 and epoch == config.curriculum_epoch


@dataclass
class EpochTally:
    """One epoch's training numbers, summed on the device and read once."""

    running: torch.Tensor
    taken: int = 0
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Trained records by rarity bucket and by rule family (FR-055).
    buckets: Counter[str] = field(default_factory=Counter)
    families: Counter[str] = field(default_factory=Counter)

    def add_shares(self, ready: StepReady) -> None:
        self.buckets.update(ready.buckets)
        self.families.update(ready.families)

    def add_step(self, loss: StepLoss) -> None:
        self.running += loss.shipped.detach()
        self.taken += 1
        for term, value in loss.terms.items():
            held = self.terms.get(term)
            self.terms[term] = value.detach() if held is None else held + value.detach()

    @property
    def train_loss(self) -> float:
        return float(self.running) / max(self.taken, 1)

    def mean_terms(self) -> dict[str, float]:
        steps = max(self.taken, 1)
        return {name: value / steps for name, value in read_back(self.terms).items()}


@dataclass
class ShardStats:
    """One shard's training numbers, for the line it logs when it ends."""

    loss: torch.Tensor
    waited: float = 0.0
    batch_waited: float = 0.0
    steps: int = 0
    learning_rate: float = 0.0
    #: The last reported step's loss by field, gradient norms and head terms.
    parts: dict[str, float] = field(default_factory=dict)
    norms: dict[str, float] = field(default_factory=dict)
    heads: dict[str, float] = field(default_factory=dict)
    #: The last optimizer step's norms, still on the device until reported.
    unread_norms: dict[str, torch.Tensor] = field(default_factory=dict)

    def add_step(self, loss: StepLoss, *, reported: bool) -> None:
        self.loss += loss.shipped.detach()
        self.steps += 1
        if reported:
            self.heads = read_back(loss.terms)
        if loss.parts:
            self.parts = loss.parts.as_dict()

    def log(self, task: ShardTask, of: int, opened: ShardOpened | None, *, trained: float) -> None:
        name = Path(task.shard).name
        trainable = opened.trainable if opened is not None else 0
        held_back = opened.held_back if opened is not None else 0
        if not trainable:
            logger.info(
                "epoch %d | shard %d/%d %s | nothing trainable, all %d records "
                "held back | skipped",
                task.epoch, task.position, of, name, held_back,
            )
            return
        logger.info(
            "epoch %d | shard %d/%d %s | %d records trainable, %d held back | "
            "%d steps | loss %.4f | lr %.2e | %.1f steps/s | "
            "wait %.1fs, batch wait %.1fs, train %.1fs%s%s%s",
            task.epoch, task.position, of, name, trainable, held_back, task.budget,
            float(self.loss) / self.steps if self.steps else float("nan"),
            self.learning_rate, task.budget / trained if trained > 0 else float("nan"),
            self.waited, self.batch_waited, trained, format_norms(self.norms),
            format_parts(self.parts), format_terms(self.heads),
        )


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
    """Owns one training run end to end: the device half of each step and
    everything between steps."""

    def __init__(
        self,
        config: TrainEffectModelConfig,
        *,
        facts: CorpusFacts,
        training_shards: Sequence[Path],
        validation_samples: Mapping[str, Path],
    ) -> None:
        self.config = config
        self.facts = facts
        #: Processes preparing steps; 0 prepares them in this one. Not a
        #: training setting: it moves where the host half runs, not what it
        #: computes.
        self.prefetch_workers = max(int(config.prefetch_workers), 0)
        self.training_shards = list(training_shards)
        self.validation_samples = dict(validation_samples)
        self._rarity_reported = False
        #: Unconverted-script lookups counted by the workers' own sidecar caches.
        self._unresolved_elsewhere: Counter[str] = Counter()
        #: One per run, so a truncated text is reported once (FR-006).
        self.truncations = TruncationLog(
            log_truncation(logger, "train-effect-model"),
        )
        #: Resolved once, so the seed logged and checkpointed is the one every
        #: draw used.
        self.seed = (
            config.seed if config.seed is not None
            else random.SystemRandom().getrandbits(32)
        )
        torch.manual_seed(self.seed)
        self.device = default_device()
        #: The device half of every step, training's and validation's.
        self.losses = StepLossComputer(config, self.device)
        self.validator = Validator(config, self.device, self.losses)

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _warn_missing_classes(self, present: frozenset[str]) -> None:
        """Say which sampling classes the validation samples do not name: the
        fields only those classes supervise silently drop for the whole run."""
        missing = [name for name in SAMPLING_CLASSES if name not in present]
        if not missing:
            return
        logger.warning(
            "The corpus's validation samples name %d of the %d sampling "
            "classes; %s are missing, so the fields only those classes "
            "supervise carry no loss this run.",
            len(present), len(SAMPLING_CLASSES), ", ".join(missing),
        )

    def execute(self) -> int:
        samples = ValidationSamples.load(self.validation_samples)
        self._warn_missing_classes(samples.present)
        if not samples.probe:
            logger.error(
                "The corpus's validation samples hold no records, so there is "
                "nothing to measure feature widths from.",
            )
            return 1
        self._announce_run(samples)
        self._warn_on_thin_holdout(samples)
        self._cap_cuda_memory()
        settings = preparer_settings(
            self.config, seed=self.seed, facts=self.facts,
            widths=feature_widths(samples.probe),
        )
        run = build_run(
            self.config, StepPreparer(settings, truncations=self.truncations),
            samples.present, facts=self.facts, seed=self.seed, device=self.device,
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

    def _announce_run(self, samples: ValidationSamples) -> None:
        logger.info(
            "Seed %d (%s), %d of %d training shards per epoch, drawn across "
            "the corpus.",
            self.seed,
            "given" if self.config.seed is not None else "drawn; pass "
            f"--seed {self.seed} to repeat this run",
            min(self.config.shards_per_epoch, len(self.training_shards)),
            len(self.training_shards),
        )
        logger.info("Classes present: %s", ", ".join(sorted(samples.present)))

    def _warn_on_thin_holdout(self, samples: ValidationSamples) -> None:
        """Warn on a thin holdout, raise on an empty one (FR-088b); the gate-1
        slice is the build's count, not this run's."""
        warning = check_holdout(
            card_disjoint_records=len(samples.card_disjoint),
            unique_text_records=self.facts.gate_one_records,
        )
        if warning:
            logger.warning("%s", warning)

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
        if resets_at(self.config, epoch=epoch):
            run.stopper.reset()
            logger.info(
                "epoch %d enables the sparse field group (%d fields now); "
                "the early stopper starts over", epoch, len(run.fields[epoch]),
            )
        self._train_epoch(epoch, run, tally)
        card, game = self.validator.epoch(run, samples, run.fields[epoch])
        result = EpochResult(
            epoch=epoch, train_loss=tally.train_loss,
            card_disjoint_loss=card.loss, game_disjoint_loss=game.loss,
        )
        self._log_epoch(result, card.progress(), card.parts, tally)
        self._log_unresolved(run.preparer)
        return self._close_epoch(epoch, run, result)

    def _train_epoch(self, epoch: int, run: RunContext, tally: EpochTally) -> None:
        drawn, tasks = self._epoch_tasks(epoch, run)
        for task in tasks:
            steps = (
                self._local_steps(task, run.preparer) if run.pool is None
                else run.pool.shard()
            )
            self._train_on_shard(task, steps, run, tally, of=drawn)
            self._release_cache()
        if run.pool is not None and epoch < self.config.epochs:
            # Queued before validation so the workers prepare while this
            # process validates; nothing validation decides changes them.
            for task in self._epoch_tasks(epoch + 1, run)[1]:
                run.pool.submit(task)

    def _log_epoch(
        self, result: EpochResult, metrics, card_parts: dict[str, float],
        tally: EpochTally,
    ) -> None:
        logger.info(
            "epoch %d | train %.4f | card-disjoint %.4f | "
            "game-disjoint %.4f | gate F1 %.3f | zone acc %.3f | "
            "deviance %.3f%s",
            result.epoch, result.train_loss, result.card_disjoint_loss,
            result.game_disjoint_loss, metrics.affected_gate_f1,
            metrics.zone_outcome_accuracy, metrics.mean_poisson_deviance,
            format_parts(card_parts, self.validator.floor.floor),
        )
        logger.info(
            "epoch %d | trained records by rarity bucket: %s%s\n"
            "  by rule family: %s",
            result.epoch,
            format_shares(tally.buckets, (*RARITY_BUCKETS, NO_TEXT_BUCKET)),
            format_terms(tally.mean_terms()),
            format_shares(tally.families),
        )

    def _log_unresolved(self, preparer: StepPreparer) -> None:
        """Scripts the converted corpus does not hold, asked for by any process."""
        unresolved = Counter(preparer.sidecars.unresolved) + self._unresolved_elsewhere
        if not unresolved:
            return
        logger.info(
            "%d ability scripts the converted corpus does not hold, "
            "%d lookups so far; those abilities reach the model as no "
            "text. Most asked: %s. A `_token` stem under cardsfolder/ "
            "is a token keyed to the wrong tree — it is the token's own "
            "cast-spell key, which maps to no text either way, while "
            "its ability lines are keyed under tokenscripts/ and do "
            "resolve; a variant-scripts/ key needs --variant-scripts.",
            len(unresolved),
            sum(unresolved.values()),
            ", ".join(f"{name} ({hits})" for name, hits in unresolved.most_common(3)),
        )

    def _close_epoch(self, epoch: int, run: RunContext, result: EpochResult) -> bool:
        """Save what the epoch earned and its resume state; True to stop early."""
        stopper = run.stopper
        best = stopper.update(result.card_disjoint_loss)
        if best or self.config.save_every_epoch:
            checkpoint = self._checkpoint(epoch, run)
            output = self.config.resolved_model_output()
            if best:
                EffectModelStore(output).save(checkpoint)
            if self.config.save_every_epoch:
                EffectModelStore(output / EPOCHS_DIR).save(checkpoint)
        save_resume(run, epoch, finished=stopper.should_stop, best=best)
        if stopper.should_stop:
            logger.info(
                "Early stop: %d epochs without a new card-disjoint best",
                self.config.patience,
            )
        return stopper.should_stop

    def _checkpoint(self, epoch: int, run: RunContext) -> EffectCheckpoint:
        return EffectCheckpoint(
            encoder_config=run.encoder_config,
            model_config=run.model_config,
            encoder_state=run.encoder.state_dict(),
            model_state=run.model.state_dict(),
            provenance=self.facts.provenance(self.config),
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

    def _epoch_tasks(self, epoch: int, run: RunContext) -> tuple[int, list[ShardTask]]:
        """``(shards drawn, tasks)`` for one epoch, in the drawn order.

        A shard given no steps gets no task, but every task keeps its drawn
        ``position``, which seeds its generators.
        """
        shards = epoch_shards(
            self.training_shards, epoch=epoch,
            per_epoch=self.config.shards_per_epoch, seed=self.seed,
        )
        allocation = steps_per_shard(self.config.steps_per_epoch, len(shards))
        return len(shards), [
            ShardTask(
                shard=Path(shard), epoch=epoch, position=position,
                budget=budget, fields=run.fields[epoch],
            )
            for position, (shard, budget) in enumerate(
                zip(shards, allocation), start=1,
            )
            if budget > 0
        ]

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
        logger.info(
            "Preparing steps in %d worker process(es), up to %d step(s) ahead "
            "each.", self.prefetch_workers, depth,
        )
        pool = StepWorkerPool(
            settings, workers=self.prefetch_workers, depth=max(depth, 1),
            pin=self.is_cuda,
        )
        try:
            for task in self._epoch_tasks(first_epoch, run)[1]:
                pool.submit(task)
        except BaseException:
            pool.close()
            raise
        return pool

    def _local_steps(self, task: ShardTask, preparer: StepPreparer) -> Iterator:
        """One shard's messages prepared in this process, the shard read on
        the first message and released with the generator."""
        yield from preparer.shard_steps(task, load_shard(task.shard), pin=self.is_cuda)

    def _release_cache(self) -> None:
        """Release the CUDA allocator's cache between shards: one shard's
        cached blocks rarely fit the next one's batch shapes."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def _note_opened(self, opened: ShardOpened) -> None:
        """Say, once per run, how much of a shard the rarity table covers: a
        table matching nothing otherwise looks exactly like a healthy one."""
        if (
            opened.rarity_coverage is None or not opened.trainable
            or self._rarity_reported
        ):
            return
        self._rarity_reported = True
        found, distinct = opened.rarity_coverage
        share = 100.0 * found / distinct if distinct else 0.0
        report = logger.info if found else logger.warning
        report(
            "Rarity table names %d of this shard's %d distinct ability "
            "text(s) (%.1f%%); the rest weigh by this shard's own game count.",
            found, distinct, share,
        )

    def _note_closed(self, closed: ShardClosed) -> None:
        """What a worker's own caches counted, into this process's reports."""
        self._unresolved_elsewhere.update(closed.unresolved)
        for text, keys in closed.truncated:
            self.truncations.note(text, keys)

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
        opened: ShardOpened | None = None
        index = 0
        try:
            for message, elapsed in _timed(steps, task, of):
                if isinstance(message, ShardOpened):
                    stats.waited += elapsed
                    started = time.perf_counter()
                    opened = message
                    self._note_opened(message)
                    continue
                if isinstance(message, ShardClosed):
                    self._note_closed(message)
                    continue
                stats.batch_waited += elapsed
                stats.learning_rate = self._set_learning_rate(run)
                tally.add_shares(message)
                # Decided before the forward pass, which is when the per-field
                # decomposition has to be asked for.
                due = reports_now(index=index, budget=task.budget)
                index += 1
                self._train_step(message.prepared, run, tally, stats, due)
        finally:
            steps.close()
        stats.log(task, of, opened, trained=time.perf_counter() - started)

    def _set_learning_rate(self, run: RunContext) -> float:
        rate = learning_rate_at(run.step, warmup=run.warmup) * self.config.lr_scale
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
