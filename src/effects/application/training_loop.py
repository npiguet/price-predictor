"""The training loop itself, separated from the pure functions it drives.

``train_effect_model`` owns reading the corpus and the schedule — all pure
functions of their inputs, all unit-testable with no torch. This module owns the
part that needs a GPU: running each step's device half, stepping the optimizer,
and validating between epochs. Each step's host half — assembling the batch
into tensors — is a ``StepPreparer``'s, which the loop owns and which runs in
worker processes (``step_workers``) or on a thread here.

The loop's shape follows four constraints from the spec:

- **An epoch is a step count, not a pass over the corpus.** The corpus reaches
  10^8 records; a pass would be a week and would make ``--patience`` meaningless.
- **The corpus is read one shard at a time.** Parsed records cost about 45 KB
  each, so holding the whole corpus would need hundreds of gigabytes. A shard
  costs about one, and it is released once its steps are taken. An epoch walks
  ``--shards-per-epoch`` of them and the walk advances, so a long run covers the
  corpus rather than re-reading its opening slice. The shards read ahead while
  the resident one trains are the single exception — one on a thread, or one per
  step-preparing worker process — and they multiply the resident cost by a
  small constant rather than raising it by a corpus.
- **Validation runs on both strata every epoch, and the best checkpoint is
  chosen by the card-disjoint one** — the number that stands in for deployment
  to an unseen set, rather than in-distribution fit. Its records are the
  corpus's own fixed samples, read once and never redrawn (FR-089): a
  validation set the trainer drew for itself would make ``--patience`` measure
  which records got drawn rather than whether the model improved, and the
  card-disjoint sample is the build's gate-1 slice rather than whatever
  arrived first.
- **The batch groups each game's records together**, so one encode of an ability
  text serves every record in that game that mentions it.
"""

from __future__ import annotations

import logging
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from effects.application.ability_text import known_texts
from effects.application.gate_one import measure
from effects.application.resume_state import ResumeState
from effects.application.step_preparation import (
    NO_TEXT_BUCKET,
    SHIPPED_HEADS,
    TRAINING_HEADS,
    Head,
    HeadTargets,
    PreparedStep,
    PreparerSettings,
    ShardClosed,
    ShardOpened,
    ShardTask,
    StepPreparer,
    StepReady,
    one_ahead,
    training_heads,
)
from effects.application.surface_batching import (
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
    api_loss,
    created_objects_loss,
    mlm_loss,
    per_entity_loss,
    value_loss,
    verdict_loss,
)
from effects.domain.effect_model import EffectModel, EffectModelConfig
from effects.domain.loss_floor import (
    EntityTargetBatch,
    PartsMean,
    constant_predictor_floor,
)
from effects.domain.rarity import RARITY_BUCKETS
from effects.domain.sampling import SAMPLING_CLASSES, BatchPlan, sampling_class
from effects.domain.split import CorpusSplit
from effects.domain.strata import VALIDATION_STRATA, Stratum
from effects.domain.surface_collation import gather_entities
from effects.infrastructure.effect_model_store import (
    EPOCHS_DIR,
    EffectCheckpoint,
    EffectModelStore,
    SplitProvenance,
    content_hash,
)
from effects.infrastructure.model_runner import IDENTITY_TABLE_KEY
from effects.infrastructure.sidecar_io import (
    script_vocabularies,
    sidecar_roots,
)
from price_predictor.infrastructure.torch_training import clip_per_group

logger = logging.getLogger(__name__)

#: Records kept aside to measure feature widths from, before any shard loads.
PROBE_RECORDS = 64

#: The most prepared steps one worker process may hold queued. A shard's
#: budget is about twenty steps at the default flags, and a worker needs a
#: shard's worth of room to prepare its next shard whole while the training
#: loop is on another worker's; the cap bounds the shared memory a run with
#: few shards and long ones would otherwise queue.
MAX_STEPS_AHEAD = 32

#: Cap on the fraction of CUDA memory the caching allocator may reserve for
#: this process (FR-095). On Windows the driver pages excess reservation out
#: to host RAM instead of raising an out-of-memory error, so an allocator left
#: unbounded fragments across the corpus's variable batch shapes and reserves
#: far more than it ever allocates while the run looks merely slow rather than
#: broken; the cap forces it to free its cache before the card is full instead
#: of trusting the driver to fail loudly.
CUDA_MEMORY_FRACTION = 0.9


def autocast_enabled(device: torch.device) -> bool:
    """Whether the forward pass and loss should run under bf16 autocast.

    CUDA only, and only when the card itself supports bf16. Pulled out of
    `TrainingLoop.__init__` as a pure function so a test can drive the two
    checks without constructing a loop against a real device.

    Guarded on ``device_count()`` because ``is_available()`` and
    ``device_count()`` can disagree under ``CUDA_VISIBLE_DEVICES=""``:
    ``is_available()`` reads the driver's raw, unmasked count, while
    ``device_count()`` applies the visibility mask that
    ``is_bf16_supported()`` (via ``get_device_properties``) asserts against —
    calling it unguarded raises on a masked-empty device rather than reading
    as unsupported.
    """
    return (
        device.type == "cuda"
        and torch.cuda.device_count() > 0
        and torch.cuda.is_bf16_supported()
    )


def reports_now(*, index: int, budget: int) -> bool:
    """Whether the step at ``index`` should read its numbers back.

    The last step of the shard, and only that one. Reading a loss term back is
    a device synchronization and the shard already logs a line when it ends, so
    the numbers ride that line and cost one synchronized step per shard.

    Args:
        index: zero-based step within the shard.
        budget: steps this shard gets.
    """
    return index + 1 >= budget


def resets_at(config, *, epoch: int) -> bool:
    """Whether the early stopper forgets its best at the start of ``epoch``.

    True once, at the curriculum epoch: the loss gains twenty-five field terms
    there, so an earlier best would be a different objective's number.
    """
    return config.curriculum_epoch > 1 and epoch == config.curriculum_epoch


def parameter_groups(encoder, model, identity_table=None) -> list[dict]:
    """One optimizer group per module, named for ``clip_per_group``.

    Clipped apart rather than together (FR-095): the head's gradient norm runs
    several times the encoder's, and a joint clip at 1.0 would scale the
    encoder's update by the head's norm — the one path the effect loss reaches
    the encoder through.
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
    #: The objective validation scores: the per-entity loss and the shipped
    #: heads, so the train and validation numbers printed side by side measure
    #: one objective.
    shipped: torch.Tensor
    #: The per-entity loss by field, on the device, when the step asked for them.
    parts: LossParts
    #: Every head's own term, by head.
    terms: dict[str, torch.Tensor]
    #: The MLM pass's masked tokens, left for after the main backward.
    pending_mlm: MaskedTokens | None = None


def combine_terms(
    loss: torch.Tensor, terms: Mapping[str, torch.Tensor],
    weights: Mapping[Head, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(total, shipped)`` from the per-entity loss and the head terms (FR-060a).

    The shipped heads join the per-entity loss at unit weight; the
    training-only heads are added at their own weights for backward alone.
    """
    shipped = loss
    for head in SHIPPED_HEADS:
        if head in terms:
            shipped = shipped + terms[head]
    total = shipped
    for head in TRAINING_HEADS:
        if head in terms:
            total = total + weights[head] * terms[head]
    return total, shipped


@dataclass
class RunContext:
    """What every step of a run trains, and where the run has got to."""

    encoder_config: AbilityEncoderConfig
    model_config: EffectModelConfig
    encoder: AbilityEncoder
    model: EffectModel
    optimizer: torch.optim.Optimizer
    warmup: int
    #: Training steps taken over the whole run; every step's draws are seeded
    #: by its place in the run.
    step: int = 0


@dataclass
class EpochTally:
    """One epoch's training numbers, summed on the device and read once.

    Read per step, each would synchronize once per batch for a number nothing
    looks at until the epoch closes.
    """

    running: torch.Tensor
    taken: int = 0
    terms: dict[str, torch.Tensor] = field(default_factory=dict)
    #: Trained records by rarity bucket and by rule family (FR-055), counted on
    #: the host from each batch's plan.
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

    #: Summed on the device like the epoch's running loss; read on that line.
    loss: torch.Tensor
    waited: float = 0.0
    batch_waited: float = 0.0
    steps: int = 0
    learning_rate: float = 0.0
    #: The last reported step's loss by field, gradient norms and head terms.
    parts: dict[str, float] = field(default_factory=dict)
    norms: dict[str, float] = field(default_factory=dict)
    heads: dict[str, float] = field(default_factory=dict)

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
    """Device scalars as floats by name, in one synchronization rather than one each.

    Stacked in float64, so each value is the one ``float()`` would have read
    on its own, whatever its dtype; moved to the first one's device, because a
    group with no gradient reports its norm as a host zero.
    """
    if not scalars:
        return {}
    values = list(scalars.values())
    device = values[0].device
    stacked = torch.stack([
        value.detach().to(device, torch.float64) for value in values
    ])
    return dict(zip(scalars, stacked.tolist()))


def _optimizer_step(optimizer, *, read: bool) -> dict[str, float]:
    """Clip each group, step and zero; the pre-clip norms only when ``read``.

    Read back only on the step that logs them: each read stalls the step until
    the GPU drains.
    """
    clipped = clip_per_group(optimizer, max_norm=MAX_GRAD_NORM, read=False)
    norms = _read_back(clipped) if read else {}
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return norms


class FloorCache:
    """The constant-predictor floor, held across the epochs that share it.

    The floor is a property of the validation targets and the active field set.
    The first never moves — the corpus's samples are fixed (FR-089) — and the
    second moves once, at the curriculum step, where the sparse group arrives
    and a floor computed before it would scale the new terms against nothing.
    So the field set is the cache key, and an epoch that shares it reuses the
    floor rather than paying a second pass over the sample for the same numbers.
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
    """Owns one training run end to end.

    Holds one shard of the corpus at a time. The corpus is 14M records and would
    need hundreds of gigabytes resident; a shard needs about one, and an epoch is
    a walk over ``--shards-per-epoch`` of them rather than a pass over the whole.

    Each step's host half is its ``preparer``'s — built once the run knows its
    feature widths and script-API vocabularies, and run here, on a thread
    (``--prefetch-workers 0``), or as one per worker process — and this class
    owns the device half and everything between steps.
    """

    def __init__(
        self,
        config: TrainEffectModelConfig,
        *,
        split: CorpusSplit,
        training_shards: Sequence[Path],
        validation_samples: Mapping[str, Path],
        holdout_permille: int,
        holdout_max_carriers: int,
        gate_one_records: int = 0,
        rarity: Mapping[str, int] | None = None,
        legality_mode_counts: Mapping[str, int] | None = None,
        corpus_digest: str = "",
        prefetch_batches: bool = True,
    ) -> None:
        self.config = config
        #: Under ``--prefetch-workers 0``, whether each step's host half is
        #: prepared on a background thread one step ahead. Off, the same
        #: function runs in place; every draw a step takes is seeded by its
        #: place in the run, so this changes the speed of a run and nothing it
        #: computes.
        self.prefetch_batches = prefetch_batches
        #: Processes preparing steps; 0 prepares them in this one. Like the
        #: thread above it moves where the host half runs and not what it
        #: computes, so it is not a training setting.
        self.prefetch_workers = max(int(config.prefetch_workers), 0)
        #: The corpus's own split, read from its manifest: the host half keeps
        #: its validation games out of training, and the checkpoint records it.
        self.split = split
        self.training_shards = list(training_shards)
        #: The corpus's fixed validation samples, one file per stratum
        #: (FR-089). Read once by ``_load_validation`` and never redrawn: the
        #: build decided what is in them, so two runs of the same corpus
        #: select their checkpoints on the same records.
        self.validation_samples = dict(validation_samples)
        #: The holdout rule the corpus was composed against, read from its
        #: manifest and stamped onto the checkpoint. The trainer has no flags
        #: for it: a depleted collection run was generated against these two
        #: numbers, and re-deriving them here could only disagree.
        self.holdout_permille = holdout_permille
        self.holdout_max_carriers = holdout_max_carriers
        #: Records in the manifest's gate-1 slice — resolution records whose
        #: acting text is on no training card. Read rather than counted here
        #: (FR-088b): the build already sized the slice.
        self.gate_one_records = gate_one_records
        #: The curated dataset's corpus-wide ``text -> games`` table (FR-146),
        #: threaded into every ``sample_weights`` call this run makes.
        self.rarity = rarity
        #: The manifest's records per legality mode, so the epoch line places
        #: a legality record in the family the build balanced it in.
        self.legality_mode_counts = legality_mode_counts
        #: Whether this run has already said how much of a shard the
        #: rarity table names. Once per run rather than once per shard:
        #: a table matching nothing is otherwise invisible, since every
        #: weight falls back to the shard's own count and the run looks
        #: exactly like a healthy one.
        self._rarity_reported = False
        #: Unconverted-script lookups the worker processes counted, which
        #: their own sidecar caches hold rather than this process's.
        self._unresolved_elsewhere: Counter[str] = Counter()
        #: The curated dataset's manifest digest at read time (FR-147), read
        #: into the checkpoint's provenance by ``_provenance`` below.
        self.corpus_digest = corpus_digest
        self.masks = baseline_masks(config.variant)
        # The surface follows the vocabulary the run loads, so the text an
        # ability is encoded from and the vocabulary it is tokenized against can
        # never disagree.
        self.surface = surface_of(config.vocab_path)
        #: One per run, handed to the preparer, so a text the encoder's window
        #: cuts short is reported once rather than every batch (FR-006).
        self.truncations = TruncationLog(
            log_truncation(logger, "train-effect-model"),
        )
        self.identity_table: IdentityTable | None = None
        #: Resolved once here rather than read from the config at each use, so
        #: the value logged at startup and written to the checkpoint is the one
        #: every draw actually used.
        self.seed = (
            config.seed if config.seed is not None
            else random.SystemRandom().getrandbits(32)
        )
        torch.manual_seed(self.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        #: Whether the device half runs the forward pass and loss under bf16
        #: autocast (FR-095). Resolved once here rather than at each call, so
        #: the value logged at startup and the value every batch runs under
        #: are the same one.
        self.autocast: bool = autocast_enabled(self.device)
        #: What each card-disjoint field term is scaled against, recomputed
        #: when the curriculum changes the field set and not otherwise.
        self.floor = FloorCache()
        self.card_disjoint: list = []
        self.game_disjoint: list = []
        self.probe: list = []
        self.present: set[str] = set()
        #: Noise on every training batch's ``e`` (FR-056), its ratio ramping
        #: in over the first epoch's steps. Never handed to a scoring batch.
        self.noise = NoiseState(config.e_noise, config.steps_per_epoch)
        #: The script-API head's classes (FR-060b), fixed before the model is
        #: built and recorded on the checkpoint.
        self.api_types: list[str] = []
        self.param_keys: list[str] = []
        #: The host half, built once the widths and vocabularies are known.
        self.preparer: StepPreparer | None = None
        #: The device halves: training's carries the noise on ``e``, scoring's
        #: none. Built with the run, once the identity table exists.
        self._training: Materializer | None = None
        self._scoring: Materializer | None = None
        self._resume: ResumeState | None = None
        #: Each epoch's field set, built once; see ``_fields_for``.
        self._epoch_fields: dict[int, tuple[FieldSpec, ...]] = {}

    # ── setup ───────────────────────────────────────────────────────────

    def _script_vocabularies(self) -> tuple[list[str], list[str]]:
        """The script-API head's two vocabularies, read off the sidecars.

        Only when the head trains: the read is every sidecar of every tree.
        """
        if self.config.api_weight <= 0.0:
            return [], []
        folders = list(self.config.cards_folders)
        if self.config.variant_scripts:
            folders.append(Path(self.config.variant_scripts))
        return script_vocabularies(folders)

    def _preparer_settings(self, widths: dict[SlotKind, int]) -> PreparerSettings:
        """What every preparer of this run is built from, here or in a worker."""
        return PreparerSettings(
            config=self.config, seed=self.seed, split=self.split,
            rarity=self.rarity, api_types=tuple(self.api_types),
            param_keys=tuple(self.param_keys), widths=dict(widths),
            heads=training_heads(self.config, api_types=self.api_types),
            legality_mode_counts=self.legality_mode_counts,
        )

    def _load_validation(self) -> None:
        """Read the corpus's fixed samples; nothing is drawn here (FR-089)."""
        self.card_disjoint = load_shard(self.validation_samples[Stratum.CARD_DISJOINT])
        self.game_disjoint = load_shard(self.validation_samples[Stratum.GAME_DISJOINT])
        self.probe = self.card_disjoint[:PROBE_RECORDS] or self.game_disjoint[:PROBE_RECORDS]
        self.present = {
            sampling_class(record) for record in (*self.card_disjoint, *self.game_disjoint)
        }
        for stratum, records in zip(
            VALIDATION_STRATA, (self.card_disjoint, self.game_disjoint),
        ):
            logger.info(
                "%s validation: %d records over %d games, %d classes",
                stratum, len(records), len({r.game_id for r in records}),
                len({sampling_class(r) for r in records}),
            )

    def _warn_missing_classes(self) -> None:
        """Say which sampling classes the validation samples do not name.

        ``self.present`` comes from those two samples alone, and it is what
        ``fields_for_epoch`` builds every epoch's field set from. A class the
        training shards carry but neither sample happens to hold drops each
        field only that class supervises — for the whole run, with nothing
        else saying so, because every number printed still looks valid.
        """
        missing = [name for name in SAMPLING_CLASSES if name not in self.present]
        if not missing:
            return
        logger.warning(
            "The corpus's validation samples name %d of the %d sampling "
            "classes; %s are missing, so the fields only those classes "
            "supervise carry no loss this run.",
            len(self.present), len(SAMPLING_CLASSES), ", ".join(missing),
        )

    # ── stepping ────────────────────────────────────────────────────────

    def _autocast(self):
        """bf16 autocast on CUDA (FR-095); a no-op context elsewhere."""
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast,
        )

    def _head_weights(self) -> dict[Head, float]:
        """The training-only heads' weights in the backward pass."""
        return {
            Head.VALUE: self.config.value_weight,
            Head.MLM: self.config.mlm_weight,
            Head.API: self.config.api_weight,
        }

    def _device_loss(
        self, prepared: PreparedStep, run: RunContext, materializer: Materializer,
        *, report_parts: bool = False,
    ) -> StepLoss:
        """One prepared step's losses: the device half.

        ``report_parts`` keeps each field's term, stacked on the device.
        Training asks for it only on a shard's last step, whose line reports
        the breakdown; validation asks on every card-disjoint batch and reads
        the terms back once, when the pass ends.

        Both training and validation go through here, so both run under
        autocast. Backward and the optimizer step stay outside it (bf16 needs
        no grad scaler); the encoder's own forward runs inside
        ``materialize``, which is why the context encloses the whole device
        half and not just the model.
        """
        with self._autocast():
            batch, _surfaces, encoded = materializer.materialize(
                prepared.batch, run.encoder,
            )
            hidden = run.model(**batch)
            outputs = run.model.per_entity(hidden)
            gathered = gather_entities(
                outputs, prepared.index.to(self.device, non_blocking=True),
            )
            # The gate, the entity mask and the presence masks stay on the
            # host, where they were built: the loss picks its rows there, so
            # the step never waits for the device to say which entities were
            # real, affected or supervised.
            loss, parts = per_entity_loss(
                gathered, prepared.gate,
                prepared.field_targets.to(self.device, non_blocking=True),
                prepared.mask, fields=prepared.fields,
                report_parts=report_parts,
            )
            terms = self._head_terms(prepared.heads, encoded, run.model, hidden)
        total, shipped = combine_terms(loss, terms, self._head_weights())
        return StepLoss(
            total=total, shipped=shipped, parts=parts, terms=terms,
            pending_mlm=prepared.heads.mlm,
        )

    def _head_terms(
        self, targets: HeadTargets, encoded: EncodedBatch, model, hidden,
    ) -> dict[str, torch.Tensor]:
        """Every loss term beside the per-entity one and the MLM pass's, as
        device tensors.

        Scores what the step's host half prepared: a head with no targets there
        is not computed at all, which is how a zero weight and a scoring pass
        both reach this. The MLM pass is a second encoder forward over the
        batch's texts, left for ``_mlm_backward`` to run after the main loss's
        backward has freed the first pass's activations: run here, both
        passes' graphs were alive at once and a batch of long texts overflowed
        8 GB. The gradient is the same sum either way.
        """
        terms: dict[str, torch.Tensor] = {}
        if targets.verdict is not None:
            target, mask = targets.verdict
            terms[Head.VERDICT] = verdict_loss(
                model.verdict(hidden).float(),
                target.to(self.device, non_blocking=True), mask,
            )
        if targets.created is not None:
            target, mask = targets.created
            terms[Head.CREATED_OBJECTS] = created_objects_loss(
                model.created_objects(hidden).float(),
                target.to(self.device, non_blocking=True), mask,
            )
        matrix = encoded.matrix
        if matrix is None:
            return terms
        if targets.value is not None:
            values, mask = targets.value
            terms[Head.VALUE] = value_loss(
                model.value_head(matrix).float(),
                values.to(self.device, non_blocking=True),
                # Kept on the host: the loss decides which rows it scores there.
                mask,
            )
        if targets.api is not None:
            types, keys = targets.api
            terms[Head.API] = api_loss(
                model.api_type_head(matrix).float(), types,
                model.param_key_head(matrix).float()
                if model.param_key_head is not None else None,
                keys.to(self.device, non_blocking=True)
                if model.param_key_head is not None else None,
            )
        return terms

    def _mlm_backward(self, loss: StepLoss, encoder, model) -> None:
        """The MLM term's forward and backward, after the main loss's backward.

        Its weighted gradient adds to the one already accumulated, which is the
        gradient of the summed objective; the term is recorded with the step's
        other heads for the epoch line.
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
        (self.config.mlm_weight * term / self.config.grad_accum).backward()
        loss.terms[Head.MLM] = term.detach()

    def _training_settings(self) -> dict:
        """The flags a sweep arm differs by, the seed, and the script-API vocabularies (FR-063).

        ``lethality_copies`` is written only when set, so a run without copies
        records, and resumes against, exactly the settings it always did.
        """
        copies = (
            {"lethality_copies": self.config.lethality_copies}
            if self.config.lethality_copies else {}
        )
        return {
            **copies,
            # The drawn one when --seed was absent: the only way to repeat it.
            "seed": self.seed,
            "e_noise": self.config.e_noise,
            "value_weight": self.config.value_weight,
            "mlm_weight": self.config.mlm_weight,
            "mlm_mask_prob": self.config.mlm_mask_prob,
            "api_weight": self.config.api_weight,
            "api_types": list(self.api_types),
            "param_keys": list(self.param_keys),
            TOKENIZER_RULES_KEY: TOKENIZER_RULES,
            INPUT_LAYOUT_KEY: INPUT_LAYOUT,
        }

    # ── the run ─────────────────────────────────────────────────────────

    def execute(self) -> int:
        if not self._announce_run():
            return 1
        widths = feature_widths(self.probe)
        self._warn_on_thin_holdout()
        self._cap_cuda_memory()
        self.api_types, self.param_keys = self._script_vocabularies()
        if self.api_types:
            logger.info(
                "Script-API head over %d API types and %d parameter keys.",
                len(self.api_types), len(self.param_keys),
            )
        settings = self._preparer_settings(widths)
        self.preparer = StepPreparer(settings, truncations=self.truncations)
        run = self._build_run(self.preparer.tokenizer.vocab_size, widths)
        stopper = EarlyStopper(self.config.patience)
        first_epoch = 1
        if self.config.resume:
            first_epoch = self._resume.restore(run, stopper)
            if first_epoch is None:
                return 0

        pool = self._start_workers(settings, first_epoch=first_epoch)
        try:
            for epoch in range(first_epoch, self.config.epochs + 1):
                if self._run_epoch(epoch, run, stopper, pool):
                    break
        finally:
            # On every way out — the last epoch, an early stop, an exception
            # or an interrupt — so no worker outlives the run.
            if pool is not None:
                pool.close()
        return 0

    def _announce_run(self) -> bool:
        """Load the validation samples and log the run's draw; False when empty."""
        logger.info(
            "Reading the corpus's validation samples: %s.",
            ", ".join(str(path) for path in self.validation_samples.values()),
        )
        self._load_validation()
        self._warn_missing_classes()
        if not self.probe:
            logger.error(
                "The corpus's validation samples hold no records, so there is "
                "nothing to measure feature widths from.",
            )
            return False
        logger.info(
            "Seed %d (%s), %d of %d training shards per epoch, drawn across "
            "the corpus.",
            self.seed,
            "given" if self.config.seed is not None else "drawn; pass "
            f"--seed {self.seed} to repeat this run",
            min(self.config.shards_per_epoch, len(self.training_shards)),
            len(self.training_shards),
        )
        logger.info("Classes present: %s", ", ".join(sorted(self.present)))
        return True

    def _warn_on_thin_holdout(self) -> None:
        """The gate-1 slice is the build's count, not this run's (FR-088b).

        An empty card-disjoint sample stops the run: left alone it trains for
        its full patience, validates `nan` every epoch, and saves no
        checkpoint at all.
        """
        warning = check_holdout(
            card_disjoint_records=len(self.card_disjoint),
            unique_text_records=self.gate_one_records,
        )
        if warning:
            logger.warning("%s", warning)

    def _cap_cuda_memory(self) -> None:
        """See :data:`CUDA_MEMORY_FRACTION`."""
        if self.device.type == "cuda":
            torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)

    def _build_run(self, vocab_size: int, widths) -> RunContext:
        """The encoder, the model, the optimizer and both device halves."""
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
            n_api_types=len(self.api_types),
            n_param_keys=len(self.param_keys),
            encoder_d_model=encoder_config.d_model,
        )
        encoder = AbilityEncoder(encoder_config).to(self.device)
        model = EffectModel(model_config).to(self.device)
        # The identity baseline learns a vector per ability text instead of
        # reading one. It replaces the encoder rather than joining it, so the
        # encoder's parameters go unused for that variant and its own table is
        # what the optimizer has to see.
        self.identity_table = (
            self._identity_table() if self.masks.identity_embedding else None
        )
        self._training = Materializer(
            self.device, self.masks, self.identity_table, noise=self.noise,
        )
        self._scoring = Materializer(self.device, self.masks, self.identity_table)
        self._resume = ResumeState(
            self.config, seed=self.seed, corpus_digest=self.corpus_digest,
            training_settings=self._training_settings, noise=self.noise,
            identity_table=self.identity_table, device=self.device,
        )
        return RunContext(
            encoder_config=encoder_config, model_config=model_config,
            encoder=encoder, model=model,
            optimizer=torch.optim.AdamW(
                parameter_groups(encoder, model, self.identity_table),
                lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
            ),
            warmup=warmup_steps(
                epochs=self.config.epochs,
                steps_per_epoch=self.config.steps_per_epoch,
            ),
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
        self, epoch: int, run: RunContext, stopper: EarlyStopper, pool,
    ) -> bool:
        """Train, validate, report and save one epoch; True when it stops the run."""
        run.encoder.train()
        run.model.train()
        # With --grad-accum > 1 the last steps of an epoch can leave a partial
        # accumulation staged for a step that never comes; dropping it here
        # keeps it from crossing into the next epoch or over the curriculum
        # switch, where the objective is no longer the same one.
        run.optimizer.zero_grad(set_to_none=True)
        tally = EpochTally(running=torch.zeros((), device=self.device))
        fields = self._fields_for(epoch)
        if resets_at(self.config, epoch=epoch):
            stopper.reset()
            logger.info(
                "epoch %d enables the sparse field group (%d fields now); "
                "the early stopper starts over", epoch, len(fields),
            )
        self._train_epoch(epoch, run, tally, pool)
        result, card_parts = self._validate_epoch(epoch, run, tally, fields)
        metrics = self._progress_metrics(run, fields)
        self._log_epoch(result, metrics, card_parts, tally)
        self._log_unresolved()
        return self._close_epoch(epoch, run, stopper, result)

    def _train_epoch(self, epoch: int, run: RunContext, tally: EpochTally, pool) -> None:
        drawn, tasks = self._epoch_tasks(epoch)
        if pool is None:
            self._train_in_process(tasks, drawn, run, tally)
            return
        for task in tasks:
            self._train_on_shard(task, pool.shard(), 0.0, run, tally, of=drawn)
            self._release_cache()
        if epoch < self.config.epochs:
            # Queued before validation rather than after it, so the workers
            # read and prepare the next epoch's first shards while this
            # process validates. Its draw and field set are functions of the
            # seed and the epoch alone, so nothing validation decides could
            # change them; an early stop discards them with the pool.
            for task in self._epoch_tasks(epoch + 1)[1]:
                pool.submit(task)

    def _validate_epoch(
        self, epoch: int, run: RunContext, tally: EpochTally, fields,
    ) -> tuple[EpochResult, dict[str, float]]:
        """Both strata's loss, the card-disjoint one by field, and the floor."""
        card_parts: dict[str, float] = {}
        # Collected only on the epoch that has a floor to compute: the targets
        # are a copy of the whole sample, and holding them past the one pass
        # that reads them would cost that for nothing.
        collected: list[EntityTargetBatch] | None = (
            [] if self.floor.stale(fields) else None
        )
        result = EpochResult(
            epoch=epoch,
            train_loss=tally.train_loss,
            card_disjoint_loss=self._validate(
                self.card_disjoint, run, parts=card_parts, fields=fields,
                collect=collected,
            ),
            game_disjoint_loss=self._validate(self.game_disjoint, run, fields=fields),
        )
        if collected is not None:
            self.floor.update(
                fields, constant_predictor_floor(collected, fields=fields),
            )
            logger.info(
                "Constant-predictor floor (card-disjoint): %s",
                format_floor(self.floor.floor),
            )
        return result, card_parts

    def _progress_metrics(self, run: RunContext, fields):
        """Gate 1's three numbers on the whole card-disjoint sample, every epoch.

        The loss says the objective fell; these say whether the model knows
        *that* something happens, *what*, and *how much*. A progress reading,
        not the gate itself: the evaluator's gate 1 scores only the unique-text
        stratum (records whose acting text is on no training card), while this
        sample also carries card-disjoint records whose text the model has
        seen elsewhere. A run whose loss falls while all three sit still is
        worth seeing early regardless.
        """
        metrics = measure(
            self.card_disjoint, run.encoder, run.model,
            self.preparer.batcher(training=False, materializer=self._scoring),
            fields=fields,
        )
        # `measure` calls `eval()` and does not call `train()` back — its
        # other caller is the evaluator, which never trains.
        run.encoder.train()
        run.model.train()
        return metrics

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

    def _log_unresolved(self) -> None:
        """Scripts the converted corpus does not hold, asked for by any process."""
        # This process's lookups and every worker's, which each counted
        # against a sidecar cache of its own.
        unresolved = Counter(self.preparer.sidecars.unresolved) + self._unresolved_elsewhere
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
        if best:
            EffectModelStore(self.config.resolved_model_output()).save(
                self._checkpoint(epoch, run, stopper),
            )
        if self.config.save_every_epoch:
            EffectModelStore(
                self.config.resolved_model_output() / EPOCHS_DIR,
            ).save(self._checkpoint(epoch, run, stopper))
        self._resume.save(epoch, run, stopper, finished=stopper.should_stop, best=best)
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
            provenance=self._provenance(),
            variant=self.config.variant,
            best_val_loss=stopper.best,
            epoch=epoch,
            # The identity baseline's e lives in this table rather than in the
            # encoder, so a checkpoint without it reloads as random vectors and
            # hands gate 1 a margin nobody earned.
            extra=(
                {IDENTITY_TABLE_KEY: self.identity_table.state()}
                if self.identity_table is not None else {}
            ),
            cards_folders=tuple(str(folder) for folder in self.config.cards_folders),
            training_settings=self._training_settings(),
        )

    # ── one epoch's shards ──────────────────────────────────────────────

    def _fields_for(self, epoch: int) -> tuple[FieldSpec, ...]:
        """The field set every step of ``epoch`` trains with, and validation scores.

        Memoised so every pass of an epoch reads the one object — including an
        epoch whose tasks were queued for the workers before it began.
        """
        fields = self._epoch_fields.get(epoch)
        if fields is None:
            fields = self._epoch_fields[epoch] = fields_for_epoch(
                self.config, present=frozenset(self.present), epoch=epoch,
            )
        return fields

    def _epoch_tasks(self, epoch: int) -> tuple[int, list[ShardTask]]:
        """``(shards drawn, tasks)`` for one epoch, in the drawn order.

        A shard the allocation gave no steps was skipped without being read
        and still is, so it is dropped here rather than prepared and thrown
        away; its ``position`` comes along so the log line still numbers
        shards the way the draw did, and so its generators are seeded by it.
        """
        shards = epoch_shards(
            self.training_shards, epoch=epoch,
            per_epoch=self.config.shards_per_epoch, seed=self.seed,
        )
        allocation = steps_per_shard(self.config.steps_per_epoch, len(shards))
        fields = self._fields_for(epoch)
        return len(shards), [
            ShardTask(
                shard=Path(shard), epoch=epoch, position=position,
                budget=budget, fields=fields,
            )
            for position, (shard, budget) in enumerate(
                zip(shards, allocation), start=1,
            )
            if budget > 0
        ]

    def _start_workers(self, settings: PreparerSettings, *, first_epoch: int = 1):
        """The step-preparing processes, with ``first_epoch`` queued; None for none.

        Each worker may hold one shard's steps prepared ahead (up to
        ``MAX_STEPS_AHEAD``): with fewer, the workers whose shards come later
        fill their queues and stop, and only the worker serving the shard in
        training is preparing anything.
        """
        if self.prefetch_workers <= 0 or not self.training_shards:
            return None
        from effects.infrastructure.step_workers import StepWorkerPool

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
            pin=self.device.type == "cuda",
        )
        try:
            for task in self._epoch_tasks(first_epoch)[1]:
                pool.submit(task)
        except BaseException:
            pool.close()
            raise
        return pool

    def _train_in_process(
        self, tasks: Sequence[ShardTask], drawn: int, run: RunContext,
        tally: EpochTally,
    ) -> None:
        """One epoch's shards with the host half in this process.

        One shard is read on a thread while the previous one trains. Reading
        one is a gigabyte of gzip and JSON with the GPU idle, and the gzip and
        the file read release the interpreter lock while the JSON decode does
        not, so the overlap is partial, which is still most of the read. One
        thread, so the shards are read in the drawn order and one extra shard
        is the most that is ever resident.
        """
        loader = ThreadPoolExecutor(max_workers=1)
        try:
            pending = loader.submit(load_shard, tasks[0].shard) if tasks else None
            for index, task in enumerate(tasks):
                waited = time.perf_counter()
                try:
                    records = pending.result()
                except Exception:
                    # Raised here rather than where it was read, so it belongs
                    # to the shard whose turn it is; the future's own
                    # traceback names the loader and no shard at all.
                    logger.error(
                        "epoch %d | shard %d/%d %s | reading it failed",
                        task.epoch, task.position, drawn, task.shard.name,
                    )
                    raise
                waited = time.perf_counter() - waited
                pending = (
                    loader.submit(load_shard, tasks[index + 1].shard)
                    if index + 1 < len(tasks) else None
                )
                steps = self._local_steps(task, records)
                self._train_on_shard(task, steps, waited, run, tally, of=drawn)
                self._release_cache()
                # Before the next shard is waited for, so the one in hand and
                # the one being read are the only two resident.
                del steps, records
        finally:
            # Including on the way out of an exception, where a prefetch may
            # still be reading a shard nobody will train on.
            loader.shutdown(wait=True, cancel_futures=True)

    def _local_steps(self, task: ShardTask, records):
        """The preparer's ``shard_steps`` in this process, one step ahead on a thread.

        The thread prepares the next step's host half while the main thread
        runs this one's device half, whose torch calls release the
        interpreter lock. Every draw a step takes is seeded by its place in
        the run, so the thread changes when a step is prepared and not what
        it holds; ``prefetch_batches`` off runs the same generator in place.
        """
        steps = self.preparer.shard_steps(
            task, records, pin=self.device.type == "cuda",
        )
        if not self.prefetch_batches:
            yield from steps
            return
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="batch-prefetch")
        try:
            yield from one_ahead(steps, pool)
        finally:
            # Including on the way out of an exception, where the thread may
            # still be preparing a step nobody will take.
            pool.shutdown(wait=True, cancel_futures=True)

    def _release_cache(self) -> None:
        """Release the CUDA allocator's cache between shards.

        The batch shapes change every shard, so the cached blocks from the one
        just trained rarely fit the next; releasing them once per shard costs
        a few milliseconds against the paging it prevents.
        """
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _note_opened(self, opened: ShardOpened) -> None:
        """Say, once per run, how much of a shard the rarity table covers.

        The coverage is said once per run rather than once per shard: a table
        matching nothing is otherwise invisible, since every weight falls back
        to the shard's own count and the run looks exactly like a healthy one.
        """
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
        self, task: ShardTask, steps, waited: float, run: RunContext,
        tally: EpochTally, *, of: int,
    ) -> None:
        """Take ``task.budget`` steps on one shard, from its prepared steps.

        ``steps`` yields the shard's messages in order — a ``ShardOpened``,
        then each step's ``StepReady``, and from a worker a ``ShardClosed`` —
        whether they come from a thread of this process or from a worker
        process. Only the device half runs here: the step's tensors were all
        built where the step was prepared, and this process copies them, runs
        the encoder with autograd, adds the noise to ``e``, and steps.

        ``waited`` is how long the epoch loop waited for the shard's read,
        and the time to the shard's opening message is added to it: what the
        line reports as ``wait`` is the part of reading and splitting a shard
        the steps before it did not hide. ``batch wait`` is how long the steps
        waited for their preparation — near nothing when the preparation
        keeps ahead of the device, and the host half's whole cost when it
        runs in place.
        """
        stats = ShardStats(loss=torch.zeros((), device=self.device), waited=waited)
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
                # Counted from the plan where it was prepared: no device read
                # (FR-055).
                tally.add_shares(message)
                # Decided before the forward pass: the decomposition has to be
                # asked for while the loss is being computed, not after. The
                # last step of every shard reports, so the shard's own line
                # always carries one.
                due = reports_now(index=index, budget=task.budget)
                index += 1
                if message.prepared is not None:
                    self._train_step(message.prepared, run, tally, stats, due)
        finally:
            # Including on the way out of an exception, where a thread may
            # still be preparing a step nobody will take.
            close = getattr(steps, "close", None)
            if close is not None:
                close()
        stats.log(task, of, opened, trained=time.perf_counter() - started)

    def _set_learning_rate(self, run: RunContext) -> float:
        rate = learning_rate_at(run.step, warmup=run.warmup) * self.config.lr_scale
        for group in run.optimizer.param_groups:
            group["lr"] = rate
        return rate

    def _train_step(
        self, prepared: PreparedStep, run: RunContext, tally: EpochTally,
        stats: ShardStats, due: bool,
    ) -> None:
        """One step's device half: forward, backward, and the optimizer on the
        accumulation boundary."""
        self.noise.step = run.step
        computed = self._device_loss(prepared, run, self._training, report_parts=due)
        # `total` adds the training-only heads to the objective validation
        # scores, `shipped`, for the backward pass.
        (computed.total / self.config.grad_accum).backward()
        self._mlm_backward(computed, run.encoder, run.model)
        # Counted within the epoch, which drops any partial accumulation it
        # began: the run's global step would put every later epoch's boundary
        # on the wrong micro-step.
        if (tally.taken + 1) % self.config.grad_accum == 0:
            norms = _optimizer_step(run.optimizer, read=due)
            if due:
                stats.norms = norms
        tally.add_step(computed)
        stats.add_step(computed, reported=due)
        run.step += 1

    def _validate(
        self, records, run: RunContext, *, fields,
        parts: dict[str, float] | None = None,
        collect: list[EntityTargetBatch] | None = None,
    ) -> float:
        """Mean loss over the stratum, batched the way training batches.

        Fixed-size batches over the sample (shuffled when the corpus was
        built), not one batch per game: a game is a handful of numbers, one of
        which blowing up moves a per-game mean many times over, and its class
        composition would decide which fields carry loss at all. ``fields`` is
        the epoch's own set rather than each batch's, so the validation number
        and the training number beside it score the same objective even where a
        batch happens to hold no record of some class. ``parts`` collects the
        loss by field, averaged over batches, which is what says *which* term
        is not moving when the total is not moving. ``collect`` takes each
        batch's targets along the way, kept on the host, so the
        constant-predictor floor those terms are scaled by is scored on this
        sample and this masking rather than on a second pass's.

        Records are still grouped by game *within* a batch, for the same reason
        training groups them: one encode of an ability text then serves every
        record of that game which references it.
        """
        if not records:
            return float("nan")
        run.encoder.eval()
        run.model.eval()

        size = self.config.batch_size
        total = torch.zeros((), device=self.device)
        batches = 0
        means = PartsMean()
        with torch.no_grad():
            for start in range(0, len(records), size):
                chunk = records[start:start + size]
                grouped: dict[str, list] = defaultdict(list)
                for record in chunk:
                    grouped[record.game_id].append(record)
                prepared = self.preparer.prepare_step(
                    BatchPlan(dict(grouped)), step=run.step, fields=fields,
                    training=False,
                )
                if prepared is None:
                    continue
                if collect is not None:
                    collect.append(EntityTargetBatch(
                        prepared.gate, prepared.field_targets, prepared.mask,
                    ))
                computed = self._device_loss(
                    prepared, run, self._scoring, report_parts=parts is not None,
                )
                # Accumulated on the device; read once below.
                total += computed.shipped
                batches += 1
                means.add(computed.parts)
        run.encoder.train()
        run.model.train()
        if parts is not None:
            parts.update(means.means())
        return float(total) / batches if batches else float("nan")

    def _provenance(self) -> SplitProvenance:
        """What the checkpoint records about the data it saw.

        Both strata are the corpus's own, read from the manifest rather than
        derived here, so every run against one dataset records the same split
        whatever it read or how early it stopped. That is what the evaluator
        needs: it scores the games the checkpoint names, and a boundary
        recomputed against a grown corpus would not be the one this model
        trained against.
        """
        split = self.split
        return SplitProvenance(
            held_out_cards=split.held_out_cards,
            card_disjoint_games=tuple(sorted(split.card_disjoint_games)),
            game_disjoint_games=tuple(sorted(split.game_disjoint_games)),
            vocab_path=str(self.config.vocab_path),
            keyword_definitions_path=str(self.config.keyword_definitions),
            vocab_hash=content_hash(Path(self.config.vocab_path)),
            keyword_definitions_hash=content_hash(
                Path(self.config.keyword_definitions)
            ),
            withheld_keyword=self.config.withhold_keyword,
            # The corpus's own, straight off its manifest: `build-corpus`
            # composed the holdout, and a constant here would be a second
            # spelling of it that nothing would report disagreeing.
            holdout_permille=self.holdout_permille,
            holdout_max_carriers=self.holdout_max_carriers,
            corpus_path=self.config.corpus or "",
            corpus_digest=self.corpus_digest,
            # Resolved from the manifest at startup when the flag was absent.
            holdout_unit=self.config.holdout_unit or "text",
        )
