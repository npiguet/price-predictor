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
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch

from effects.application.ability_text import known_texts
from effects.application.gate_one import GateOneMetrics, MarginAccumulator
from effects.application.resume_state import ResumeState
from effects.application.step_preparation import (
    NO_TEXT_BUCKET,
    DeviceStep,
    HeadTargets,
    PreparerSettings,
    ShardClosed,
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
    training_heads,
)
from effects.application.surface_batching import (
    IDENTITY_TABLE_KEY,
    EncodedBatch,
    IdentityTable,
    Materializer,
    NoiseState,
)
from effects.application.text_staging import MaskedTokens, MlmMasker
from effects.application.train_effect_model import (
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    CorpusFacts,
    EarlyStopper,
    EpochResult,
    TrainEffectModelConfig,
    check_holdout,
    epoch_shards,
    fields_for_epoch,
    learning_rate_at,
    load_shard,
    steps_per_shard,
    warmup_steps,
)
from effects.application.training_log import (
    format_floor,
    format_norms,
    format_parts,
    format_shares,
    format_terms,
)
from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    TruncationLog,
    log_truncation,
    surface_of,
)
from effects.domain.ability_tokenizer import (
    TOKENIZER_RULES,
    TOKENIZER_RULES_KEY,
)
from effects.domain.aux_heads import (
    AUX_HEAD,
    AUX_HEADS,
    SHIPPED_HEADS,
    TRAINING_HEADS,
    Head,
    HeadInput,
)
from effects.domain.baselines import baseline_masks
from effects.domain.effect_head_input import (
    INPUT_LAYOUT,
    INPUT_LAYOUT_KEY,
    SlotKind,
    feature_widths,
)
from effects.domain.effect_head_layout import FieldSpec
from effects.domain.effect_losses import (
    LossParts,
    mlm_loss,
    per_entity_loss,
)
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.loss_floor import PartsMean, constant_predictor_floor
from effects.domain.rarity import RARITY_BUCKETS
from effects.domain.sampling import SAMPLING_CLASSES, BatchPlan, sampling_class
from effects.domain.strata import VALIDATION_STRATA, Stratum
from effects.domain.surface_collation import gather_entities
from effects.infrastructure.effect_model_store import (
    EPOCHS_DIR,
    EffectCheckpoint,
    EffectModelStore,
)
from effects.infrastructure.sidecar_io import (
    script_vocabularies,
    sidecar_roots,
)
from effects.infrastructure.step_workers import StepWorkerPool
from effects.infrastructure.torch_device import default_device
from price_predictor.infrastructure.torch_training import clip_per_group

logger = logging.getLogger(__name__)

#: Records kept aside to measure feature widths from, before any shard loads.
PROBE_RECORDS = 64

#: The most prepared steps one worker process may hold queued: room for one
#: shard's steps, capped to bound the shared memory of a run with long shards.
MAX_STEPS_AHEAD = 32

#: Cap on the fraction of CUDA memory the caching allocator may reserve
#: (FR-095). On Windows the driver pages excess reservation out to host RAM
#: rather than raising out-of-memory, so an unbounded allocator makes the run
#: slow instead of failing.
CUDA_MEMORY_FRACTION = 0.9


def autocast_enabled(device: torch.device) -> bool:
    """Whether the forward pass and loss run under bf16 autocast: CUDA only,
    and only when the card supports bf16.

    Guarded on ``device_count()`` because under ``CUDA_VISIBLE_DEVICES=""``
    ``is_available()`` can still be true and ``is_bf16_supported()`` would
    raise rather than read as unsupported.
    """
    return (
        device.type == "cuda"
        and torch.cuda.device_count() > 0
        and torch.cuda.is_bf16_supported()
    )


def reports_now(*, index: int, budget: int) -> bool:
    """Whether the step at zero-based ``index`` reads its numbers back: only a
    shard's last, since each read is a device synchronization."""
    return index + 1 >= budget


def resets_at(config, *, epoch: int) -> bool:
    """Whether the early stopper forgets its best at the start of ``epoch``:
    at the curriculum epoch, where the objective gains the sparse fields."""
    return config.curriculum_epoch > 1 and epoch == config.curriculum_epoch


def parameter_groups(encoder, model, identity_table=None) -> list[dict]:
    """One optimizer group per module, named for ``clip_per_group``.

    Clipped apart (FR-095): the head's gradient norm runs several times the
    encoder's, and a joint clip would shrink the encoder's update by it.
    """
    groups = [
        {"name": "encoder", "params": list(encoder.parameters())},
        {"name": "head", "params": list(model.parameters())},
    ]
    if identity_table is not None:
        groups.append({"name": "identity", "params": list(identity_table.parameters())})
    return groups


@dataclass
class StepLoss:
    """One batch's losses, on the device."""

    #: What backward runs on: ``shipped`` plus the weighted training-only heads.
    total: torch.Tensor
    #: The objective validation scores and the train line prints: the
    #: per-entity loss and the shipped heads.
    shipped: torch.Tensor
    parts: LossParts
    terms: dict[str, torch.Tensor]
    #: The MLM pass's masked tokens, left for after the main backward.
    pending_mlm: MaskedTokens | None = None
    #: The per-entity head's output at each target entity, detached.
    gathered: torch.Tensor | None = None


def combine_terms(
    loss: torch.Tensor, terms: Mapping[str, torch.Tensor],
    weights: Mapping[Head, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(total, shipped)`` from the per-entity loss and the head terms (FR-060a).

    Shipped heads join at unit weight, training-only heads at their own
    weights for backward alone. Summed in :data:`AUX_HEADS` order, which is
    part of what makes a run repeat bit for bit.
    """
    shipped = loss
    for spec in SHIPPED_HEADS:
        if spec.head in terms:
            shipped = shipped + terms[spec.head]
    total = shipped
    for spec in TRAINING_HEADS:
        if spec.head in terms:
            total = total + weights[spec.head] * terms[spec.head]
    return total, shipped


@dataclass(frozen=True)
class ValidationSamples:
    """The corpus's fixed validation samples, one per stratum, read once and
    never redrawn (FR-089)."""

    card_disjoint: list
    game_disjoint: list
    #: Records to measure feature widths from, before any shard loads.
    probe: list
    #: The sampling classes the two samples hold; every epoch's field set is
    #: built from these.
    present: frozenset[str]


@dataclass
class RunContext:
    """What every step of a run trains with, and where the run has got to."""

    encoder_config: AbilityEncoderConfig
    model_config: EffectModelConfig
    encoder: AbilityEncoder
    model: EffectModel
    optimizer: torch.optim.Optimizer
    warmup: int
    #: The host half of the steps this process prepares, and of validation.
    preparer: StepPreparer
    #: The device halves: training's carries the noise on ``e``, scoring's none.
    training: Materializer
    scoring: Materializer
    #: The identity baseline's table; None for every other variant.
    identity_table: IdentityTable | None
    resume: ResumeState
    #: What the checkpoint and the resume file record of the run (FR-063).
    training_settings: dict
    #: Each epoch's field set, fixed up front so tasks queued for the workers
    #: before an epoch began read the same one.
    fields: Mapping[int, tuple[FieldSpec, ...]]
    #: Training steps taken over the whole run.
    step: int = 0


@contextmanager
def evaluating(run: RunContext) -> Iterator[None]:
    """Both modules in evaluation mode for the block, back in training mode
    however it ends: every training step assumes training mode."""
    run.encoder.eval()
    run.model.eval()
    try:
        yield
    finally:
        run.encoder.train()
        run.model.train()


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
        return {name: value / steps for name, value in _read_back(self.terms).items()}


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
            self.heads = _read_back(loss.terms)
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


def _read_back(scalars: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """Device scalars as floats by name, in one synchronization.

    Stacked in float64, so each value is the one ``float()`` would have read;
    moved to the first one's device, because a group with no gradient reports
    its norm as a host zero.
    """
    if not scalars:
        return {}
    values = list(scalars.values())
    device = values[0].device
    stacked = torch.stack([
        value.detach().to(device, torch.float64) for value in values
    ])
    return dict(zip(scalars, stacked.tolist()))


def _progress_metrics(margins: MarginAccumulator) -> GateOneMetrics:
    """Gate 1's three numbers on the whole card-disjoint sample, every epoch.

    A progress reading, not the gate: the evaluator scores only the
    unique-text stratum, and in float32 where this pass runs under bf16
    autocast, so a logit near zero can land on the other side here.
    """
    return margins.metrics()


@dataclass(frozen=True, slots=True)
class _HostCopy:
    """One validation batch's gathered outputs on their way to the host,
    queued without waiting and read a batch later so the device never idles."""

    gathered: torch.Tensor
    done: torch.cuda.Event | None
    prepared: DeviceStep

    @classmethod
    def queue(cls, gathered: torch.Tensor, prepared: DeviceStep) -> _HostCopy:
        host = gathered.float().to("cpu", non_blocking=True)
        done = None
        if gathered.device.type == "cuda":
            done = torch.cuda.Event()
            done.record()
        return cls(host, done, prepared)

    def fold_into(self, margins: MarginAccumulator) -> None:
        if self.done is not None:
            self.done.synchronize()
        margins.add(self.gathered, self.prepared.targets)


def _optimizer_step(optimizer) -> dict[str, torch.Tensor]:
    """Clip each group, step and zero; the pre-clip norms unread, since each
    read stalls until the GPU drains."""
    norms = clip_per_group(optimizer, max_norm=MAX_GRAD_NORM, read=False)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norms


class FloorCache:
    """The constant-predictor floor, keyed by the active field set.

    The validation targets never move (FR-089), so the floor changes only when
    the field set does, at the curriculum epoch.
    """

    def __init__(self) -> None:
        self._fields: tuple[FieldSpec, ...] | None = None
        self.floor: dict[str, float] = {}

    def stale(self, fields: tuple[FieldSpec, ...]) -> bool:
        return self._fields != fields

    def update(
        self, fields: tuple[FieldSpec, ...], floor: Mapping[str, float],
    ) -> None:
        self._fields = fields
        self.floor = dict(floor)


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
        self.masks = baseline_masks(config.variant)
        self.surface = surface_of(config.vocab_path)
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
        self.autocast: bool = autocast_enabled(self.device)
        self.floor = FloorCache()
        #: Noise on every training batch's ``e`` (FR-056); never on a scoring
        #: batch.
        self.noise = NoiseState(config.e_noise, config.steps_per_epoch)

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    def _script_vocabularies(self) -> tuple[list[str], list[str]]:
        """The script-API head's two vocabularies; read only when the head
        trains, since the read is every sidecar of every tree."""
        if self.config.api_weight <= 0.0:
            return [], []
        folders = list(self.config.cards_folders)
        if self.config.variant_scripts:
            folders.append(Path(self.config.variant_scripts))
        return script_vocabularies(folders)

    def _preparer_settings(self, widths: dict[SlotKind, int]) -> PreparerSettings:
        """What every preparer of this run is built from, here or in a worker."""
        api_types, param_keys = self._script_vocabularies()
        if api_types:
            logger.info(
                "Script-API head over %d API types and %d parameter keys.",
                len(api_types), len(param_keys),
            )
        return PreparerSettings(
            config=self.config, seed=self.seed, split=self.facts.split,
            rarity=self.facts.rarity, api_types=tuple(api_types),
            param_keys=tuple(param_keys), widths=dict(widths),
            heads=training_heads(self.config, api_types=api_types),
            legality_mode_counts=self.facts.legality_mode_counts,
        )

    def _load_validation(self) -> ValidationSamples:
        logger.info(
            "Reading the corpus's validation samples: %s.",
            ", ".join(str(path) for path in self.validation_samples.values()),
        )
        card_disjoint = load_shard(self.validation_samples[Stratum.CARD_DISJOINT])
        game_disjoint = load_shard(self.validation_samples[Stratum.GAME_DISJOINT])
        for stratum, records in zip(VALIDATION_STRATA, (card_disjoint, game_disjoint)):
            logger.info(
                "%s validation: %d records over %d games, %d classes",
                stratum, len(records), len({r.game_id for r in records}),
                len({sampling_class(r) for r in records}),
            )
        return ValidationSamples(
            card_disjoint=card_disjoint,
            game_disjoint=game_disjoint,
            probe=card_disjoint[:PROBE_RECORDS] or game_disjoint[:PROBE_RECORDS],
            present=frozenset(
                sampling_class(record) for record in (*card_disjoint, *game_disjoint)
            ),
        )

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

    def _autocast(self):
        """bf16 autocast on CUDA (FR-095); a no-op context elsewhere."""
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast,
        )

    def _head_weights(self) -> dict[Head, float]:
        return {spec.head: spec.weight(self.config) for spec in TRAINING_HEADS}

    def _device_loss(
        self, prepared: DeviceStep, run: RunContext, materializer: Materializer,
        *, report_parts: bool = False,
    ) -> StepLoss:
        """One prepared step's losses: the device half, for training and
        validation alike.

        ``report_parts`` keeps each field's term, stacked on the device.
        Autocast encloses ``materialize`` too, because the encoder's forward
        runs there; backward and the optimizer step stay outside it (bf16
        needs no grad scaler).
        """
        with self._autocast():
            batch, _surfaces, encoded = materializer.materialize(
                prepared.batch, run.encoder,
            )
            hidden = run.model(**batch)
            outputs = run.model.per_entity(hidden)
            targets = prepared.targets
            gathered = gather_entities(
                outputs, targets.index.to(self.device, non_blocking=True),
            )
            loss, parts = per_entity_loss(
                gathered, targets.fields_on(self.device, non_blocking=True),
                fields=prepared.fields, report_parts=report_parts,
            )
            terms = self._head_terms(prepared.heads, encoded, run.model, hidden)
        total, shipped = combine_terms(loss, terms, self._head_weights())
        return StepLoss(
            total=total, shipped=shipped, parts=parts, terms=terms,
            pending_mlm=prepared.heads.get(Head.MLM), gathered=gathered.detach(),
        )

    def _head_terms(
        self, targets: HeadTargets, encoded: EncodedBatch, model, hidden,
    ) -> dict[str, torch.Tensor]:
        """Every head's loss term but the MLM pass's, as device tensors; a head
        the host half prepared no targets for is not computed."""
        inputs = {HeadInput.TRUNK: hidden, HeadInput.ENCODED: encoded.matrix}
        terms: dict[str, torch.Tensor] = {}
        for spec in AUX_HEADS:
            given = targets.get(spec.head)
            source = inputs.get(spec.reads)
            if spec.loss is None or given is None or source is None:
                continue
            terms[spec.head] = spec.loss(
                model, source, *spec.to_device(given, self.device),
            )
        return terms

    def _mlm_backward(self, loss: StepLoss, encoder, model) -> None:
        """The MLM term's forward and backward, after the main loss's backward
        has freed its activations: with both graphs alive at once, long texts
        overflowed 8 GB. The accumulated gradient is the same sum either way.
        """
        masked: MaskedTokens | None = loss.pending_mlm
        if masked is None:
            return
        loss.pending_mlm = None
        with self._autocast():
            token_hidden, token_targets = MlmMasker.forward(encoder, masked, self.device)
            term = mlm_loss(
                model.mlm_head(token_hidden).float(), token_targets, rows=masked.rows,
            )
        weight = AUX_HEAD[Head.MLM].weight(self.config)
        (weight * term / self.config.grad_accum).backward()
        loss.terms[Head.MLM] = term.detach()

    def _training_settings(self, api_types, param_keys) -> dict:
        """The flags a sweep arm differs by, the seed, and the script-API
        vocabularies (FR-063). ``lethality_copies`` is written only when set,
        so a run without copies resumes against the settings it always had."""
        copies = (
            {"lethality_copies": self.config.lethality_copies}
            if self.config.lethality_copies else {}
        )
        return {
            **copies,
            "seed": self.seed,
            "e_noise": self.config.e_noise,
            "value_weight": self.config.value_weight,
            "mlm_weight": self.config.mlm_weight,
            "mlm_mask_prob": self.config.mlm_mask_prob,
            "api_weight": self.config.api_weight,
            "api_types": list(api_types),
            "param_keys": list(param_keys),
            TOKENIZER_RULES_KEY: TOKENIZER_RULES,
            INPUT_LAYOUT_KEY: INPUT_LAYOUT.name,
        }

    def execute(self) -> int:
        samples = self._load_validation()
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
        settings = self._preparer_settings(feature_widths(samples.probe))
        run = self._build_run(
            StepPreparer(settings, truncations=self.truncations), samples.present,
        )
        stopper = EarlyStopper(self.config.patience)
        first_epoch = 1
        if self.config.resume:
            first_epoch = run.resume.restore(run, stopper)
            if first_epoch is None:
                return 0

        pool = self._start_workers(settings, run, first_epoch=first_epoch)
        try:
            for epoch in range(first_epoch, self.config.epochs + 1):
                if self._run_epoch(epoch, run, samples, stopper, pool):
                    break
        finally:
            if pool is not None:
                pool.close()
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

    def _build_run(self, preparer: StepPreparer, present: frozenset[str]) -> RunContext:
        """The encoder, the model, the optimizer and both device halves, sized
        by ``preparer``."""
        vocab_size = preparer.tokenizer.vocab_size
        widths = preparer.widths
        encoder_config = AbilityEncoderConfig(
            vocab_size=vocab_size,
            e_dim=self.config.e_dim,
            d_model=self.config.encoder_d_model,
            n_layers=self.config.encoder_layers,
        )
        model_config = EffectModelConfig(
            global_features=widths[SlotKind.GLOBAL],
            act_features=widths[SlotKind.ACT],
            player_features=widths[SlotKind.PLAYER],
            card_features=widths[SlotKind.CARD],
            e_dim=self.config.e_dim,
            vocab_size=vocab_size,
            n_api_types=len(preparer.api_types),
            n_param_keys=len(preparer.param_keys),
            encoder_d_model=encoder_config.d_model,
        )
        encoder = AbilityEncoder(encoder_config).to(self.device)
        model = EffectModel(model_config).to(self.device)
        # The identity baseline's table replaces the encoder's output, so the
        # optimizer has to see it.
        identity_table = (
            self._identity_table() if self.masks.identity_embedding else None
        )
        recorded = self._training_settings(preparer.api_types, preparer.param_keys)
        return RunContext(
            encoder_config=encoder_config, model_config=model_config,
            encoder=encoder, model=model,
            optimizer=torch.optim.AdamW(
                parameter_groups(encoder, model, identity_table),
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            ),
            warmup=warmup_steps(
                epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
            ),
            preparer=preparer,
            training=Materializer(
                self.device, self.masks, identity_table, noise=self.noise,
            ),
            scoring=Materializer(self.device, self.masks, identity_table),
            identity_table=identity_table,
            resume=ResumeState(
                self.config, seed=self.seed, corpus_digest=self.facts.digest,
                training_settings=partial(dict, recorded), noise=self.noise,
                identity_table=identity_table, device=self.device,
            ),
            training_settings=recorded,
            fields={
                epoch: fields_for_epoch(self.config, present=present, epoch=epoch)
                for epoch in range(1, self.config.epochs + 1)
            },
        )

    def _identity_table(self) -> IdentityTable:
        """The identity baseline's table, one row per text the sidecars hold."""
        texts = known_texts(
            sidecar_roots(self.config.cards_folders, self.config.variant_scripts),
            self.surface,
        )
        logger.info("Identity table over %d ability texts.", len(texts))
        return IdentityTable(texts, self.config.e_dim).to(self.device)

    def _run_epoch(
        self, epoch: int, run: RunContext, samples: ValidationSamples,
        stopper: EarlyStopper, pool: StepWorkerPool | None,
    ) -> bool:
        """Train, validate, report and save one epoch; True when it stops the run."""
        # Drops a partial --grad-accum accumulation the last epoch left staged,
        # so it never crosses into a different objective.
        run.optimizer.zero_grad(set_to_none=True)
        tally = EpochTally(running=torch.zeros((), device=self.device))
        if resets_at(self.config, epoch=epoch):
            stopper.reset()
            logger.info(
                "epoch %d enables the sparse field group (%d fields now); "
                "the early stopper starts over", epoch, len(run.fields[epoch]),
            )
        self._train_epoch(epoch, run, tally, pool)
        result, card_parts, metrics = self._validate_epoch(epoch, run, tally, samples)
        self._log_epoch(result, metrics, card_parts, tally)
        self._log_unresolved(run.preparer)
        return self._close_epoch(epoch, run, stopper, result)

    def _train_epoch(
        self, epoch: int, run: RunContext, tally: EpochTally,
        pool: StepWorkerPool | None,
    ) -> None:
        drawn, tasks = self._epoch_tasks(epoch, run)
        for task in tasks:
            steps = (
                self._local_steps(task, run.preparer) if pool is None
                else pool.shard()
            )
            self._train_on_shard(task, steps, run, tally, of=drawn)
            self._release_cache()
        if pool is not None and epoch < self.config.epochs:
            # Queued before validation so the workers prepare while this
            # process validates; nothing validation decides changes them.
            for task in self._epoch_tasks(epoch + 1, run)[1]:
                pool.submit(task)

    def _validate_epoch(
        self, epoch: int, run: RunContext, tally: EpochTally,
        samples: ValidationSamples,
    ) -> tuple[EpochResult, dict[str, float], GateOneMetrics]:
        """Both strata's loss, the card-disjoint one by field, the floor, and
        gate 1's progress margins."""
        fields = run.fields[epoch]
        card_parts: dict[str, float] = {}
        margins = MarginAccumulator(fields)
        # Collected only on an epoch with a floor to compute: a copy of the
        # whole sample's targets.
        collected: list[EntityTargetBatch] | None = (
            [] if self.floor.stale(fields) else None
        )
        result = EpochResult(
            epoch=epoch,
            train_loss=tally.train_loss,
            card_disjoint_loss=self._validate(
                samples.card_disjoint, run, parts=card_parts, fields=fields,
                collect=collected, margins=margins,
            ),
            game_disjoint_loss=self._validate(samples.game_disjoint, run, fields=fields),
        )
        if collected is not None:
            self.floor.update(
                fields, constant_predictor_floor(collected, fields=fields),
            )
            logger.info(
                "Constant-predictor floor (card-disjoint): %s",
                format_floor(self.floor.floor),
            )
        return result, card_parts, _progress_metrics(margins)

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
            format_parts(card_parts, self.floor.floor),
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

    def _close_epoch(
        self, epoch: int, run: RunContext, stopper: EarlyStopper, result: EpochResult,
    ) -> bool:
        """Save what the epoch earned and its resume state; True to stop early."""
        best = stopper.update(result.card_disjoint_loss)
        if best or self.config.save_every_epoch:
            checkpoint = self._checkpoint(epoch, run, stopper)
            output = self.config.resolved_model_output()
            if best:
                EffectModelStore(output).save(checkpoint)
            if self.config.save_every_epoch:
                EffectModelStore(output / EPOCHS_DIR).save(checkpoint)
        run.resume.save(epoch, run, stopper, finished=stopper.should_stop, best=best)
        if stopper.should_stop:
            logger.info(
                "Early stop: %d epochs without a new card-disjoint best",
                self.config.patience,
            )
        return stopper.should_stop

    def _checkpoint(
        self, epoch: int, run: RunContext, stopper: EarlyStopper,
    ) -> EffectCheckpoint:
        return EffectCheckpoint(
            encoder_config=run.encoder_config,
            model_config=run.model_config,
            encoder_state=run.encoder.state_dict(),
            model_state=run.model.state_dict(),
            provenance=self.facts.provenance(self.config),
            variant=self.config.variant,
            best_val_loss=stopper.best,
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
        self.noise.step = run.step
        computed = self._device_loss(prepared, run, run.training, report_parts=due)
        (computed.total / self.config.grad_accum).backward()
        self._mlm_backward(computed, run.encoder, run.model)
        # Counted within the epoch, not by run.step: each epoch drops any
        # partial accumulation, so the run's step would misplace boundaries.
        if (tally.taken + 1) % self.config.grad_accum == 0:
            stats.unread_norms = _optimizer_step(run.optimizer)
        if due:
            stats.norms = _read_back(stats.unread_norms)
        tally.add_step(computed)
        stats.add_step(computed, reported=due)
        run.step += 1

    def _validate(
        self, records, run: RunContext, *, fields,
        parts: dict[str, float] | None = None,
        collect: list[EntityTargetBatch] | None = None,
        margins: MarginAccumulator | None = None,
    ) -> float:
        """Mean loss over the stratum, in fixed-size batches, not one per game:
        one game's outlier would move a per-game mean many times over.

        ``fields`` is the epoch's set, so validation and training score one
        objective. ``parts`` collects the loss by field, averaged over batches;
        ``collect`` takes each batch's host targets for the floor and
        ``margins`` its outputs for gate 1's progress margins, so neither needs
        a second pass. Records are grouped by game within a batch, so one
        encode of a text serves the whole game.
        """
        if not records:
            return float("nan")
        size = self.config.batch_size
        total = torch.zeros((), device=self.device)
        batches = 0
        means = PartsMean()
        pending: _HostCopy | None = None
        with evaluating(run), torch.no_grad():
            for start in range(0, len(records), size):
                grouped: dict[str, list] = defaultdict(list)
                for record in records[start:start + size]:
                    grouped[record.game_id].append(record)
                prepared = run.preparer.prepare_scoring_step(
                    BatchPlan(dict(grouped)), fields,
                )
                if collect is not None:
                    collect.append(prepared.targets)
                computed = self._device_loss(
                    prepared, run, run.scoring, report_parts=parts is not None,
                )
                total += computed.shipped
                batches += 1
                means.add(computed.parts)
                if margins is not None:
                    copy = _HostCopy.queue(computed.gathered, prepared)
                    if pending is not None:
                        pending.fold_into(margins)
                    pending = copy
            if pending is not None:
                pending.fold_into(margins)
        if parts is not None:
            parts.update(means.means())
        return float(total) / batches if batches else float("nan")
