"""The training loop itself, separated from the pure functions it drives.

``train_effect_model`` owns reading the corpus and the schedule — all pure
functions of their inputs, all unit-testable with no torch. This module owns the
part that needs a GPU: running each step's device half, stepping the optimizer,
and validating between epochs. Each step's host half — assembling the batch
into tensors — is ``step_preparation``'s, and runs in worker processes
(``step_workers``) or on a thread here.

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
from dataclasses import asdict
from pathlib import Path

import torch

from effects.application.gate_one import measure
from effects.application.step_preparation import (
    NO_TEXT_BUCKET,
    HeadTargets,
    PreparedStep,
    PreparerSettings,
    ShardClosed,
    ShardOpened,
    ShardTask,
    StepPreparation,
    one_ahead,
)
from effects.application.surface_batching import (
    IDENTITY_TABLE_SIZE,
    NoiseState,
    SurfaceBatcher,
)
from effects.application.train_effect_model import (
    LEARNING_RATE,
    MAX_GRAD_NORM,
    WEIGHT_DECAY,
    ContextCache,
    CorpusSplit,
    EarlyStopper,
    EpochResult,
    HeldOutCards,
    SplitAccumulator,
    TrainEffectModelConfig,
    check_holdout,
    epoch_shards,
    fields_for_epoch,
    learning_rate_at,
    load_shard,
    sampling_class,
    steps_per_shard,
    variant_masks,
    warmup_steps,
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
from effects.domain.effect_head_input import (
    SlotKind,
    act_features,
    card_features,
    global_features,
    player_features,
)
from effects.domain.effect_model import (
    FLOOR_EPSILON,
    SAMPLING_CLASSES,
    EffectModel,
    EffectModelConfig,
    EntityTargetBatch,
    FieldSpec,
    api_loss,
    constant_predictor_floor,
    created_objects_loss,
    mlm_loss,
    per_entity_loss,
    value_loss,
    verdict_loss,
)
from effects.domain.rarity import RARITY_BUCKETS
from effects.infrastructure.effect_model_store import (
    EffectCheckpoint,
    EffectModelStore,
    SplitProvenance,
    content_hash,
)
from effects.infrastructure.model_runner import IDENTITY_TABLE_KEY
from effects.infrastructure.sidecar_io import (
    script_vocabularies,
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

#: Re-exported: the training loop's tests and callers name them here.
__all__ = ["NO_TEXT_BUCKET", "HeadTargets", "PreparedStep", "TrainingLoop"]

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

    Clipped apart rather than together (FR-095): the head's gradient norm ran
    five to twenty times the encoder's in the first run, and a joint clip at
    1.0 scaled the encoder's update by the head's norm — the one path the
    effect loss reaches the encoder through, cut to a twentieth.
    """
    groups = [
        {"name": "encoder", "params": list(encoder.parameters())},
        {"name": "head", "params": list(model.parameters())},
    ]
    if identity_table is not None:
        groups.append({"name": "identity", "params": list(identity_table.parameters())})
    return groups


def _format_norms(norms: Mapping[str, float]) -> str:
    """The pre-clip gradient norms, or nothing before the first backward."""
    if not norms:
        return ""
    body = ", ".join(f"{name} {value:.2f}" for name, value in norms.items())
    return f"\n  |g| {body} (clipped per group at {MAX_GRAD_NORM:g})"


def _explained(name: str, value: float, floor: Mapping[str, float] | None) -> str:
    """How much of the constant predictor's loss this field's head removed.

    ``1 - loss/floor`` against the best constant prediction for the field
    (FR-127c): zero is the base rate, negative is worse than predicting it. A
    field's loss is a number in nats with no scale of its own, so on its own it
    cannot say whether the head learned the field or learned how often it fires.

    A field whose targets never vary has no deviance to explain and reads
    ``n/a`` rather than dividing by nothing at all.
    """
    if floor is None:
        return ""
    reference = floor.get(name)
    if reference is None or reference < FLOOR_EPSILON:
        return " (n/a)"
    return f" ({round(100.0 * (1.0 - value / reference))}%)"


def _format_parts(
    parts: Mapping[str, float], floor: Mapping[str, float] | None = None,
) -> str:
    """Each field's loss term, largest first, and what it explains.

    Ordered by size rather than by name because the question this line answers
    is which term the loss is made of: a single field carrying nearly all of it
    is what a blow-up looks like, and alphabetical order buries that.

    ``floor`` is the card-disjoint constant-predictor floor and is passed only
    for that breakdown; the per-shard training line has no floor of its own —
    its records change every shard — and prints the terms alone.
    """
    if not parts:
        return ""
    ranked = sorted(parts.items(), key=lambda kv: -kv[1])
    body = ", ".join(
        f"{name} {value:.3f}{_explained(name, value, floor)}"
        for name, value in ranked
    )
    return f"\n  fields: {body}"


def _format_floor(floor: Mapping[str, float]) -> str:
    """The floor itself, ordered like the breakdown it scales."""
    ranked = sorted(floor.items(), key=lambda kv: -kv[1])
    return ", ".join(f"{name} {value:.3f}" for name, value in ranked)


#: The loss terms beside the per-entity loss (FR-060a, FR-058, FR-060b). The
#: first two are shipped heads and enter the validation loss; the last three
#: are training-only and enter the training loss at their flags' weights.
SHIPPED_TERMS: tuple[str, ...] = ("verdict", "created_objects")
TRAINING_TERMS: tuple[str, ...] = ("value", "mlm", "api")


def format_shares(counts: Mapping[str, int], order: Sequence[str] = ()) -> str:
    """``label share%`` per label, in ``order`` first and then largest first."""
    total = sum(counts.values())
    if not total:
        return "none"
    ordered = [label for label in order if label in counts] + sorted(
        (label for label in counts if label not in order),
        key=lambda label: -counts[label],
    )
    return ", ".join(
        f"{label} {100.0 * counts[label] / total:.1f}%" for label in ordered
    )


def _format_terms(terms: Mapping[str, float]) -> str:
    """The head terms of the loss, or nothing when there were none."""
    if not terms:
        return ""
    body = ", ".join(f"{name} {value:.3f}" for name, value in terms.items())
    return f"\n  heads: {body}"


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


class TrainingLoop(StepPreparation):
    """Owns one training run end to end.

    Holds one shard of the corpus at a time. The corpus is 14M records and would
    need hundreds of gigabytes resident; a shard needs about one, and an epoch is
    a walk over ``--shards-per-epoch`` of them rather than a pass over the whole.

    The host half of each step is ``StepPreparation``'s, run either here, on a
    thread (``--prefetch-workers 0``), or in worker processes; this class owns
    the device half and everything between steps.
    """

    def __init__(
        self,
        config: TrainEffectModelConfig,
        *,
        held_out: HeldOutCards,
        inherited: CorpusSplit | None,
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
        self.held_out = held_out
        #: The corpus's own split, which every ``--corpus`` run inherits; the
        #: host half reads it to keep validation games out of training.
        self.inherited = inherited
        self.training_shards = list(training_shards)
        #: The corpus's fixed validation samples, one file per stratum
        #: (FR-089). Read once by ``_load_validation`` and never redrawn: the
        #: build decided what is in them, so two runs of the same corpus
        #: select their checkpoints on the same records.
        self.validation_samples = dict(validation_samples)
        #: The holdout rule the corpus was composed against, read from its
        #: manifest and stamped onto the checkpoint. The trainer has no flags
        #: for it any more: a depleted collection run was generated against
        #: these two numbers, and re-deriving them here could only disagree.
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
        self.accumulator = (
            SplitAccumulator.inheriting(inherited) if inherited is not None
            else SplitAccumulator(held_out_cards=tuple(sorted(held_out.names)))
        )
        self.masks = variant_masks(config.variant)
        # The surface follows the vocabulary the run loads, so the text an
        # ability is encoded from and the vocabulary it is tokenized against can
        # never disagree.
        self.surface = surface_of(config.vocab_path)
        #: One per run, handed to every batcher, so a text the encoder's window
        #: cuts short is reported once rather than every batch (FR-006).
        self.truncations = TruncationLog(
            log_truncation(logger, "train-effect-model"),
        )
        self.identity_table = None
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
        #: Whether `_loss_for` wraps the forward pass and loss in bf16
        #: autocast (FR-095). Resolved once here rather than at each call, so
        #: the value logged at startup and the value every batch runs under
        #: are the same one.
        self.autocast: bool = autocast_enabled(self.device)
        self.context_cache = (
            ContextCache(config.cache_refresh) if config.context_cache else None
        )
        #: What each card-disjoint field term is scaled against, recomputed
        #: when the curriculum changes the field set and not otherwise.
        self.floor = FloorCache()
        self.card_disjoint: list = []
        self.game_disjoint: list = []
        self.probe: list = []
        self.present: set[str] = set()
        #: Noise on every training batch's ``e`` (FR-056), its ratio ramping
        #: in over the first epoch's steps. Never handed to a scoring batcher.
        self.noise = NoiseState(config.e_noise, config.steps_per_epoch)
        #: The script-API head's classes (FR-060b), fixed before the model is
        #: built and recorded on the checkpoint.
        self.api_types: list[str] = []
        self.param_keys: list[str] = []
        self._init_preparation()
        #: The last ``_loss_for``'s head terms, as device tensors.
        self._last_terms: dict[str, torch.Tensor] = {}
        #: The step's masked tokens and batcher, for `_mlm_backward`.
        self._pending_mlm = None
        #: This epoch's head terms, summed on the device and read once.
        self._epoch_terms: dict[str, torch.Tensor] = {}
        self._epoch_term_steps = 0
        #: This epoch's trained records by rarity bucket and by rule family
        #: (FR-055), counted on the host from each batch's plan.
        self._bucket_counts: defaultdict[str, int] = defaultdict(int)
        self._family_counts: defaultdict[str, int] = defaultdict(int)
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

    def _load_validation(self) -> None:
        """Read the corpus's fixed samples; nothing is drawn here (FR-089)."""
        self.card_disjoint = load_shard(self.validation_samples["card-disjoint"])
        self.game_disjoint = load_shard(self.validation_samples["game-disjoint"])
        self.probe = self.card_disjoint[:PROBE_RECORDS] or self.game_disjoint[:PROBE_RECORDS]
        self.present = {
            sampling_class(record) for record in (*self.card_disjoint, *self.game_disjoint)
        }
        for stratum, records in (("card-disjoint", self.card_disjoint),
                                 ("game-disjoint", self.game_disjoint)):
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

    def _feature_widths(self, records: Sequence) -> dict[SlotKind, int]:
        """Measure each slot kind's width from a real record.

        Measured rather than declared: the vocabularies the features are built
        from live in one module, and a constant here would be a second place to
        keep them in step.
        """
        probe = records[0]
        widths = {
            SlotKind.GLOBAL: len(global_features(probe)),
            SlotKind.ACT: len(act_features(probe)),
            SlotKind.PLAYER: 0,
            SlotKind.CARD: 0,
        }
        for record in records:
            if record.state.players:
                widths[SlotKind.PLAYER] = len(
                    player_features(record.state.players[0], record)
                )
            if record.state.entities:
                widths[SlotKind.CARD] = len(
                    card_features(record.state.entities[0], record)
                )
            if widths[SlotKind.PLAYER] and widths[SlotKind.CARD]:
                break
        return widths

    # ── stepping ────────────────────────────────────────────────────────

    def _training_heads(self, model) -> frozenset[str]:
        """The training-only heads a step computes; a zero weight computes none."""
        heads = set()
        if self.config.value_weight > 0.0:
            heads.add("value")
        if self.config.api_weight > 0.0 and model.api_type_head is not None:
            heads.add("api")
        if self.config.mlm_weight > 0.0 and model.mlm_head is not None:
            heads.add("mlm")
        return frozenset(heads)

    def _loss_for(
        self, plan, encoder, model, tokenizer, sidecars, widths, step,
        *, report_parts: bool = False, fields=None, training: bool = True,
        collect: list[EntityTargetBatch] | None = None,
        prepared: PreparedStep | None = None,
        batcher: SurfaceBatcher | None = None,
    ):
        """``(loss, parts)`` for one planned batch, or ``None`` when empty.

        ``report_parts`` reads each field's term back as a float, which is one
        device synchronization per active field. Passed only on the batch a
        progress line is about to report, so the cost lands a few times a
        minute rather than on all five thousand steps of an epoch.

        ``fields`` is the epoch's own field set (FR-082): every batch of an
        epoch scores the same objective, and validation scores that same one,
        so the two numbers printed side by side are comparable. Left ``None``
        the batch derives its own from the classes it holds, which is what
        other callers building their own batches need.

        ``collect`` receives this batch's targets, which is how the
        constant-predictor floor is scored over exactly the entities the loss
        was. The targets do not depend on the weights, so the pass that reads
        the loss hands them over rather than a second pass re-deriving them.

        ``prepared`` is the step's host half when the caller built it ahead of
        time, as training does; without it the host half runs here first,
        which is what validation does. Either way this is the device half.
        A training step's host half carries no batcher of its own, and
        ``batcher`` is the device-side one the loop hands every step.
        """
        if prepared is None:
            prepared = self._prepare_step(
                plan, tokenizer, sidecars, widths, step, fields=fields,
                training=training,
                heads=self._training_heads(model) if training else frozenset(),
            )
            if prepared is None:
                return None
        if collect is not None:
            # Kept on the host: the floor is scored once per field set, and a
            # copy of every validation target on the device would sit beside
            # the model for the rest of the run.
            collect.append(EntityTargetBatch(
                prepared.gate, prepared.field_targets, prepared.mask,
            ))
        # Both training and validation go through here, so both run under
        # autocast. Backward and the optimizer step stay outside this context
        # (bf16 needs no grad scaler); the encoder's own forward happens
        # inside `.materialize(...)` (`SurfaceBatcher._encode_staged` calls
        # `encoder(**batch)`), which is why the context has to enclose the
        # batch's device half and not just `model(**batch)`.
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast,
        ):
            if prepared.batcher is not None:
                batcher = prepared.batcher
            batch, _surfaces = batcher.materialize(prepared.batch, encoder)
            hidden = model(**batch)
            outputs = model.per_entity(hidden)
            # One batched gather rather than a slice per row: `index` selects
            # each surface's [CARD] and [PLAYER] columns, and the rows are
            # independent.
            index = prepared.index.to(self.device, non_blocking=True)
            gathered = outputs.gather(
                1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]),
            )
            # The gate and the entity mask stay on the host, where they were
            # built: the loss picks its rows there, so the step never waits
            # for the device to say which entities were real or affected.
            loss, parts = per_entity_loss(
                gathered, prepared.gate,
                {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in prepared.field_targets.items()
                },
                prepared.mask, fields=prepared.fields,
                report_parts=report_parts,
            )
            terms = self._head_terms(
                prepared.heads, batcher, encoder, model, hidden,
            )
        self._last_terms = terms
        # The shipped heads join the per-entity loss at unit weight (FR-060a)
        # and are what validation scores; the training-only heads are added
        # for the backward pass alone, so the train and validation numbers
        # printed side by side measure one objective.
        shipped = loss
        for name in SHIPPED_TERMS:
            if name in terms:
                shipped = shipped + terms[name]
        total = shipped
        weights = {
            "value": self.config.value_weight,
            "mlm": self.config.mlm_weight,
            "api": self.config.api_weight,
        }
        for name in TRAINING_TERMS:
            if name in terms:
                total = total + weights[name] * terms[name]
        return total, parts, shipped

    def _head_terms(
        self, targets: HeadTargets, batcher, encoder, model, hidden,
    ) -> dict[str, torch.Tensor]:
        """Every loss term beside the per-entity one, as device tensors.

        Scores what ``_head_targets`` prepared: a head with no targets there is
        not computed at all, which is how a zero weight and a scoring pass both
        reach this.
        """
        terms: dict[str, torch.Tensor] = {}
        if targets.verdict is not None:
            target, mask = targets.verdict
            terms["verdict"] = verdict_loss(
                model.verdict(hidden).float(),
                target.to(self.device, non_blocking=True), mask,
            )
        if targets.created is not None:
            target, mask = targets.created
            terms["created_objects"] = created_objects_loss(
                model.created_objects(hidden).float(),
                target.to(self.device, non_blocking=True), mask,
            )
        encoded = getattr(batcher, "encoded", None)
        if encoded is None or encoded.matrix is None:
            return terms
        matrix = encoded.matrix
        if targets.value is not None:
            values, mask = targets.value
            terms["value"] = value_loss(
                model.value_head(matrix).float(),
                values.to(self.device, non_blocking=True),
                # Kept on the host: the loss decides which rows it scores there.
                mask,
            )
        if targets.api is not None:
            types, keys = targets.api
            terms["api"] = api_loss(
                model.api_type_head(matrix).float(), types,
                model.param_key_head(matrix).float()
                if model.param_key_head is not None else None,
                keys.to(self.device, non_blocking=True)
                if model.param_key_head is not None else None,
            )
        # The MLM pass is a second encoder forward over the batch's texts. It is
        # left for `_mlm_backward`, which runs it after the main loss's backward
        # has freed the first pass's activations: run here, both passes' graphs
        # were alive at once and a batch of long texts overflowed 8 GB. The
        # gradient is the same sum either way.
        self._pending_mlm = (
            (targets.mlm, batcher) if targets.mlm is not None else None
        )
        return terms

    def _mlm_backward(self, encoder, model) -> None:
        """The MLM term's forward and backward, after the main loss's backward.

        Its weighted gradient adds to the one already accumulated, which is the
        gradient of the summed objective; the term is recorded with the step's
        other heads for the epoch line.
        """
        pending, self._pending_mlm = self._pending_mlm, None
        if pending is None:
            return
        masked, batcher = pending
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast,
        ):
            token_hidden, token_targets, token_mask = batcher.mlm_forward(
                encoder, masked,
            )
            term = mlm_loss(
                model.mlm_head(token_hidden).float(), token_targets, token_mask,
            )
        (self.config.mlm_weight * term / self.config.grad_accum).backward()
        self._last_terms = {**self._last_terms, "mlm": term.detach()}

    def _training_settings(self) -> dict:
        """The flags a sweep arm differs by, and the script-API vocabularies (FR-063)."""
        return {
            "e_noise": self.config.e_noise,
            "value_weight": self.config.value_weight,
            "mlm_weight": self.config.mlm_weight,
            "mlm_mask_prob": self.config.mlm_mask_prob,
            "api_weight": self.config.api_weight,
            "api_types": list(self.api_types),
            "param_keys": list(self.param_keys),
            TOKENIZER_RULES_KEY: TOKENIZER_RULES,
        }

    # ── the run ─────────────────────────────────────────────────────────

    def execute(self) -> int:
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
            return 1

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

        tokenizer = self._build_tokenizer()
        sidecars = self._build_sidecars()
        widths = self._feature_widths(self.probe)

        # The gate-1 slice is the build's count, not this run's (FR-088b). An
        # empty card-disjoint sample stops the run: left alone it trains for
        # its full patience, validates `nan` every epoch, and saves no
        # checkpoint at all.
        warning = check_holdout(
            card_disjoint_records=len(self.card_disjoint),
            unique_text_records=self.gate_one_records,
        )
        if warning:
            logger.warning("%s", warning)

        if self.device.type == "cuda":
            # The Windows driver pages excess reservation out to host RAM
            # instead of raising, so the allocator needs an explicit cap: it
            # fragments on the corpus's variable batch shapes and reserves far
            # more than it ever allocates, and only a hard cap forces it to
            # free that cache before the card is full.
            torch.cuda.set_per_process_memory_fraction(CUDA_MEMORY_FRACTION)

        encoder_config = AbilityEncoderConfig(
            vocab_size=tokenizer.vocab_size,
            e_dim=self.config.e_dim,
            d_model=self.config.encoder_d_model,
            n_layers=self.config.encoder_layers,
        )
        self.api_types, self.param_keys = self._script_vocabularies()
        if self.api_types:
            logger.info(
                "Script-API head over %d API types and %d parameter keys.",
                len(self.api_types), len(self.param_keys),
            )
        model_config = EffectModelConfig(
            global_features=widths[SlotKind.GLOBAL],
            act_features=widths[SlotKind.ACT],
            player_features=widths[SlotKind.PLAYER],
            card_features=widths[SlotKind.CARD],
            e_dim=self.config.e_dim,
            vocab_size=tokenizer.vocab_size,
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
            torch.nn.Embedding(IDENTITY_TABLE_SIZE, self.config.e_dim).to(
                self.device
            )
            if self.masks.identity_embedding
            else None
        )

        optimizer = torch.optim.AdamW(
            parameter_groups(encoder, model, self.identity_table),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        warmup = warmup_steps(
            epochs=self.config.epochs, steps_per_epoch=self.config.steps_per_epoch,
        )

        stopper = EarlyStopper(self.config.patience)
        store = EffectModelStore(self.config.resolved_model_output())
        step = 0

        heads = self._training_heads(model)
        pool = self._start_workers(widths, heads)
        try:
            for epoch in range(1, self.config.epochs + 1):
                encoder.train()
                model.train()
                # With --grad-accum > 1 the last steps of an epoch can leave a
                # partial accumulation staged for a step that never comes; dropping
                # it here keeps it from crossing into the next epoch or over the
                # curriculum switch, where the objective is no longer the same one.
                optimizer.zero_grad(set_to_none=True)
                # Accumulated on the device and read once at the end of the epoch.
                # Reading it per step would synchronize once per batch for a number
                # nothing looks at until the epoch closes.
                running = torch.zeros((), device=self.device)
                taken = 0
                self._epoch_terms = {}
                self._epoch_term_steps = 0
                self._bucket_counts.clear()
                self._family_counts.clear()
                fields = self._fields_for(epoch)
                if resets_at(self.config, epoch=epoch):
                    stopper.reset()
                    logger.info(
                        "epoch %d enables the sparse field group (%d fields now); "
                        "the early stopper starts over", epoch, len(fields),
                    )
                drawn, tasks = self._epoch_tasks(epoch)
                if pool is not None:
                    for task in tasks:
                        step, taken = self._train_on_shard(
                            task, pool.shard(), 0.0, encoder=encoder, model=model,
                            tokenizer=tokenizer, sidecars=sidecars, widths=widths,
                            optimizer=optimizer, warmup=warmup, running=running,
                            step=step, taken=taken, of=drawn,
                        )
                        self._release_cache()
                    if epoch < self.config.epochs:
                        # Queued before validation rather than after it, so the
                        # workers read and prepare the next epoch's first shards
                        # while this process validates. Its draw and field set are
                        # functions of the seed and the epoch alone, so nothing
                        # validation decides could change them; an early stop
                        # discards them with the pool.
                        for task in self._epoch_tasks(epoch + 1)[1]:
                            pool.submit(task)
                else:
                    step, taken = self._train_in_process(
                        tasks, drawn, encoder=encoder, model=model,
                        tokenizer=tokenizer, sidecars=sidecars, widths=widths,
                        optimizer=optimizer, warmup=warmup, running=running,
                        step=step, taken=taken, heads=heads,
                    )

                card_parts: dict[str, float] = {}
                # Collected only on the epoch that has a floor to compute: the
                # targets are a copy of the whole sample, and holding them past the
                # one pass that reads them would cost that for nothing.
                collected: list[EntityTargetBatch] | None = (
                    [] if self.floor.stale(fields) else None
                )
                result = EpochResult(
                    epoch=epoch,
                    train_loss=float(running) / max(taken, 1),
                    card_disjoint_loss=self._validate(
                        self.card_disjoint, encoder, model, tokenizer, sidecars,
                        widths, step, parts=card_parts, fields=fields,
                        collect=collected,
                    ),
                    game_disjoint_loss=self._validate(
                        self.game_disjoint, encoder, model, tokenizer, sidecars,
                        widths, step, fields=fields,
                    ),
                )
                if collected is not None:
                    self.floor.update(
                        fields, constant_predictor_floor(collected, fields=fields),
                    )
                    del collected
                    logger.info(
                        "Constant-predictor floor (card-disjoint): %s",
                        _format_floor(self.floor.floor),
                    )
                # Gate 1's three numbers on the whole card-disjoint sample, every
                # epoch. The loss says the objective fell; these say whether the
                # model knows *that* something happens, *what*, and *how much* —
                # a progress reading, not the gate itself: the evaluator's actual
                # gate 1 scores only the unique-text stratum (records whose acting
                # text is on no training card), while this sample also carries
                # card-disjoint records whose text the model has seen elsewhere. A
                # run whose loss falls while all three sit still is worth seeing
                # early regardless.
                metrics = measure(
                    self.card_disjoint, encoder, model,
                    self._batcher(tokenizer, sidecars, widths, training=False),
                    fields=fields,
                )
                # `measure` calls `eval()` and does not call `train()` back — its
                # other caller is the evaluator, which never trains. The next
                # epoch's first line does, so this only matters for the save
                # below, but a mode left flipped by a diagnostic is not a thing to
                # leave for the next reader to rediscover.
                encoder.train()
                model.train()
                logger.info(
                    "epoch %d | train %.4f | card-disjoint %.4f | "
                    "game-disjoint %.4f | gate F1 %.3f | zone acc %.3f | "
                    "deviance %.3f%s",
                    result.epoch, result.train_loss, result.card_disjoint_loss,
                    result.game_disjoint_loss, metrics.affected_gate_f1,
                    metrics.zone_outcome_accuracy, metrics.mean_poisson_deviance,
                    _format_parts(card_parts, self.floor.floor),
                )
                # One read per term per epoch: summed on the device all epoch.
                steps = max(self._epoch_term_steps, 1)
                logger.info(
                    "epoch %d | trained records by rarity bucket: %s%s\n"
                    "  by rule family: %s",
                    epoch,
                    format_shares(self._bucket_counts, (*RARITY_BUCKETS, NO_TEXT_BUCKET)),
                    _format_terms({
                        name: float(value) / steps
                        for name, value in self._epoch_terms.items()
                    }),
                    format_shares(self._family_counts),
                )
                # This process's lookups and every worker's, which each
                # counted against a sidecar cache of its own.
                unresolved = Counter(sidecars.unresolved) + self._unresolved_elsewhere
                if unresolved:
                    worst = sorted(
                        unresolved.items(), key=lambda kv: -kv[1],
                    )[:3]
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
                        ", ".join(f"{name} ({hits})" for name, hits in worst),
                    )
                if stopper.update(result.card_disjoint_loss):
                    store.save(EffectCheckpoint(
                        encoder_config=encoder_config,
                        model_config=model_config,
                        encoder_state=encoder.state_dict(),
                        model_state=model.state_dict(),
                        provenance=self._provenance(),
                        variant=self.config.variant,
                        best_val_loss=stopper.best,
                        epoch=epoch,
                        # The identity baseline's e lives in this table rather than
                        # in the encoder, so a checkpoint without it reloads as
                        # random vectors and hands gate 1 a margin nobody earned.
                        extra=(
                            {IDENTITY_TABLE_KEY: self.identity_table.state_dict()}
                            if self.identity_table is not None else {}
                        ),
                        cards_folders=tuple(
                            str(folder) for folder in self.config.cards_folders
                        ),
                        training_settings=self._training_settings(),
                    ))
                if stopper.should_stop:
                    logger.info(
                        "Early stop: %d epochs without a new card-disjoint best",
                        self.config.patience,
                    )
                    break
        finally:
            # On every way out — the last epoch, an early stop, an exception
            # or an interrupt — so no worker outlives the run.
            if pool is not None:
                pool.close()
        return 0

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

    def _start_workers(self, widths, heads: frozenset[str]):
        """The step-preparing processes, with epoch 1 queued; None for none.

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
            PreparerSettings(
                config=self.config, seed=self.seed, held_out=self.held_out,
                inherited=self.inherited, rarity=self.rarity,
                api_types=tuple(self.api_types),
                param_keys=tuple(self.param_keys), widths=dict(widths),
                heads=heads, legality_mode_counts=self.legality_mode_counts,
            ),
            workers=self.prefetch_workers, depth=max(depth, 1),
            pin=self.device.type == "cuda",
        )
        try:
            for task in self._epoch_tasks(1)[1]:
                pool.submit(task)
        except BaseException:
            pool.close()
            raise
        return pool

    def _train_in_process(
        self, tasks: Sequence[ShardTask], drawn: int, *, encoder, model,
        tokenizer, sidecars, widths, optimizer, warmup, running, step, taken,
        heads: frozenset[str],
    ) -> tuple[int, int]:
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
                steps = self._local_steps(
                    task, records, tokenizer, sidecars, widths, heads=heads,
                )
                step, taken = self._train_on_shard(
                    task, steps, waited, encoder=encoder, model=model,
                    tokenizer=tokenizer, sidecars=sidecars, widths=widths,
                    optimizer=optimizer, warmup=warmup, running=running,
                    step=step, taken=taken, of=drawn,
                )
                self._release_cache()
                # Before the next shard is waited for, so the one in hand and
                # the one being read are the only two resident.
                del steps, records
        finally:
            # Including on the way out of an exception, where a prefetch may
            # still be reading a shard nobody will train on.
            loader.shutdown(wait=True, cancel_futures=True)
        return step, taken

    def _local_steps(
        self, task: ShardTask, records, tokenizer, sidecars, widths, *,
        heads: frozenset[str],
    ):
        """``shard_steps`` in this process, one step ahead on a thread.

        The thread prepares the next step's host half while the main thread
        runs this one's device half, whose torch calls release the
        interpreter lock. Every draw a step takes is seeded by its place in
        the run, so the thread changes when a step is prepared and not what
        it holds; ``prefetch_batches`` off runs the same generator in place.
        """
        steps = self.shard_steps(
            task, records, tokenizer, sidecars, widths, heads=heads,
            pin=self.device.type == "cuda",
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
        """Fold a shard's split into the run's; say what the rarity table covers.

        The coverage is said once per run rather than once per shard: a table
        matching nothing is otherwise invisible, since every weight falls back
        to the shard's own count and the run looks exactly like a healthy one.
        """
        self.accumulator.note_games(opened.tainted, frozenset(), reserved=False)
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
        self, task: ShardTask, steps, waited, *, encoder, model, tokenizer,
        sidecars, widths, optimizer, warmup, running, step, taken, of,
    ) -> tuple[int, int]:
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

        Returns the advanced ``(step, taken)`` counters.
        """
        epoch, position, budget = task.epoch, task.position, task.budget
        name = Path(task.shard).name
        # The device half's batcher: a training step's host half carries
        # none, and nothing here reads the tokenizer or the sidecars.
        device = self._batcher(tokenizer, sidecars, widths, training=True)
        started = time.perf_counter()
        opened: ShardOpened | None = None
        # Accumulated on the device like the epoch's own running loss, and read
        # back once, on the line this shard logs when it ends.
        shard_loss = torch.zeros((), device=self.device)
        shard_steps = 0
        index = 0
        parts: dict[str, float] = {}
        norms: dict[str, float] = {}
        heads: dict[str, float] = {}
        learning_rate = 0.0
        batch_waited = 0.0
        try:
            while True:
                waiting = time.perf_counter()
                try:
                    message = next(steps, None)
                except Exception:
                    logger.error(
                        "epoch %d | shard %d/%d %s | preparing it failed",
                        epoch, position, of, name,
                    )
                    raise
                elapsed = time.perf_counter() - waiting
                if message is None:
                    break
                if isinstance(message, ShardOpened):
                    waited += elapsed
                    started = time.perf_counter()
                    opened = message
                    self._note_opened(message)
                    continue
                if isinstance(message, ShardClosed):
                    self._note_closed(message)
                    continue
                batch_waited += elapsed
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate = learning_rate_at(
                        step, warmup=warmup,
                    )
                # Counted from the plan where it was prepared: no device read
                # (FR-055).
                for bucket, count in message.buckets.items():
                    self._bucket_counts[bucket] += count
                for family, count in message.families.items():
                    self._family_counts[family] += count
                # Decided before the forward pass: the decomposition has to be
                # asked for while the loss is being computed, not after.
                #
                # The last step of every shard reports, whatever the clock
                # says, so the shard's own line always carries a
                # decomposition. A wall-clock interval alone was silent here:
                # it was written when a shard took 278 steps over three
                # minutes, and a shard now takes twenty over seconds, so the
                # window never elapsed and no shard ever reported.
                due = reports_now(index=index, budget=budget)
                index += 1
                if message.prepared is None:
                    continue
                self.noise.step = step
                computed = self._loss_for(
                    None, encoder, model, tokenizer, sidecars, widths, step,
                    report_parts=due, fields=task.fields,
                    prepared=message.prepared, batcher=device,
                )
                if computed is None:
                    continue
                loss, batch_parts, *rest = computed
                # The objective validation scores; `loss` adds the
                # training-only heads on top for the backward pass.
                shipped = rest[0] if rest else loss
                (loss / self.config.grad_accum).backward()
                self._mlm_backward(encoder, model)
                if (step + 1) % self.config.grad_accum == 0:
                    # Read back only on the step that logs them: each read
                    # stalls the step until the GPU drains.
                    clipped = clip_per_group(
                        optimizer, max_norm=MAX_GRAD_NORM, read=False,
                    )
                    if due:
                        norms = {
                            group: float(value) for group, value in clipped.items()
                        }
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                running += shipped.detach()
                shard_loss += shipped.detach()
                for term, value in self._last_terms.items():
                    held = self._epoch_terms.get(term)
                    self._epoch_terms[term] = (
                        value.detach() if held is None else held + value.detach()
                    )
                self._epoch_term_steps += 1
                if due:
                    heads = {
                        term: float(value.detach())
                        for term, value in self._last_terms.items()
                    }
                shard_steps += 1
                step += 1
                taken += 1
                if batch_parts:
                    parts = batch_parts
                if self.context_cache is not None:
                    self.context_cache.note_batch()
        finally:
            # Including on the way out of an exception, where a thread may
            # still be preparing a step nobody will take.
            close = getattr(steps, "close", None)
            if close is not None:
                close()

        trainable = opened.trainable if opened is not None else 0
        held_back = opened.held_back if opened is not None else 0
        if not trainable:
            logger.info(
                "epoch %d | shard %d/%d %s | nothing trainable, all %d records "
                "held back | skipped",
                epoch, position, of, name, held_back,
            )
            return step, taken
        trained = time.perf_counter() - started
        logger.info(
            "epoch %d | shard %d/%d %s | %d records trainable, %d held back | "
            "%d steps | loss %.4f | lr %.2e | %.1f steps/s | "
            "wait %.1fs, batch wait %.1fs, train %.1fs%s%s%s",
            epoch, position, of, name, trainable, held_back, budget,
            float(shard_loss) / shard_steps if shard_steps else float("nan"),
            learning_rate, budget / trained if trained > 0 else float("nan"),
            waited, batch_waited, trained, _format_norms(norms),
            _format_parts(parts), _format_terms(heads),
        )
        return step, taken

    def _validate(
        self, records, encoder, model, tokenizer, sidecars, widths, step,
        *, parts: dict[str, float] | None = None, fields=None,
        collect: list[EntityTargetBatch] | None = None,
    ) -> float:
        """Mean loss over the stratum, batched the way training batches.

        One batch per game is what this did, and a game is not a batch. Eight
        games is eight numbers, and one of them blowing up moves the mean by a
        factor of twenty-eight. Worse, a batch's class composition decides which
        fields carry loss at all — ``active_fields`` keeps a field only while
        some class in the batch supervises it — so a batch holding one game's
        natural proportions scores a different set of fields than a training
        batch does, and the two numbers printed side by side were never
        measuring the same thing.

        Fixed-size batches over a shuffled sample fix both, and ``fields`` is
        the epoch's own set rather than each batch's, so the validation number
        and the training number beside it score the same objective even where a
        batch happens to hold no record of some class. ``parts`` collects the
        loss by field, averaged over batches, which is what says *which* term
        is not moving when the total is not moving. ``collect`` takes each
        batch's targets along the way, so the constant-predictor floor those
        terms are scaled by is scored on this sample and this masking rather
        than on a second pass's.

        Records are still grouped by game *within* a batch, for the same reason
        training groups them: one encode of an ability text then serves every
        record of that game which references it.
        """
        if not records:
            return float("nan")
        encoder.eval()
        model.eval()
        from effects.application.train_effect_model import BatchPlan

        size = self.config.batch_size
        total = torch.zeros((), device=self.device)
        batches = 0
        summed: dict[str, float] = defaultdict(float)
        counted: dict[str, int] = defaultdict(int)
        with torch.no_grad():
            for start in range(0, len(records), size):
                chunk = records[start:start + size]
                grouped: dict[str, list] = defaultdict(list)
                for record in chunk:
                    grouped[record.game_id].append(record)
                computed = self._loss_for(
                    BatchPlan(dict(grouped)), encoder, model, tokenizer,
                    sidecars, widths, step, report_parts=parts is not None,
                    fields=fields, training=False, collect=collect,
                )
                if computed is None:
                    continue
                # Accumulated on the device; read once below.
                total += computed[0]
                batches += 1
                for name, value in computed[1].items():
                    summed[name] += value
                    counted[name] += 1
        encoder.train()
        model.train()
        if parts is not None:
            # Averaged over the batches that carried each field rather than over
            # every batch: a field its batch did not supervise contributes no
            # zero, which would read as the model having got it right.
            parts.update(
                {name: summed[name] / counted[name] for name in summed}
            )
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
        split = self.accumulator.split()
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


def config_summary(config: TrainEffectModelConfig) -> dict:
    """The run's settings, for the log line and the checkpoint's extras."""
    return {k: str(v) for k, v in asdict(config).items()}
