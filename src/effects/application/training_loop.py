"""The training loop itself, separated from the pure functions it drives.

``train_effect_model`` owns the split, the mixture and the schedule — all pure
functions of their inputs, all unit-testable with no torch. This module owns the
part that needs a GPU: assembling batches into tensors, stepping the optimizer,
and validating between epochs.

The loop's shape follows four constraints from the spec:

- **An epoch is a step count, not a pass over the corpus.** The corpus reaches
  10^8 records; a pass would be a week and would make ``--patience`` meaningless.
- **The corpus is read one shard at a time.** Parsed records cost about 45 KB
  each, so holding the whole corpus would need hundreds of gigabytes. A shard
  costs about one, and it is released before the next is read. An epoch walks
  ``--shards-per-epoch`` of them and the walk advances, so a long run covers the
  corpus rather than re-reading its opening slice.
- **Validation runs on both strata every epoch, and the best checkpoint is
  chosen by the card-disjoint one** — the number that stands in for deployment
  to an unseen set, rather than in-distribution fit. Its records come from
  reserved shards the run never trains on, and they are captured once: a
  validation set redrawn each epoch would make ``--patience`` measure which
  records got drawn rather than whether the model improved.
- **The batch groups each game's records together**, so one encode of an ability
  text serves every record in that game that mentions it.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from effects.application.surface_batching import (
    IDENTITY_TABLE_SIZE,
    SurfaceBatcher,
)
from effects.application.train_effect_model import (
    LEARNING_RATE,
    MAX_GRAD_NORM,
    RANDOM_SEED,
    VALIDATION_RECORDS_PER_STRATUM,
    WEIGHT_DECAY,
    ContextCache,
    CorpusSplit,
    EarlyStopper,
    EpochResult,
    HeldOutCards,
    SplitAccumulator,
    TrainEffectModelConfig,
    ability_text_of,
    check_holdout,
    class_counts,
    epoch_shards,
    learning_rate_at,
    load_shard,
    parse_kind_mix,
    plan_batch,
    rarity_coverage,
    renormalize_mix,
    sample_weights,
    sampling_class,
    steps_per_shard,
    unique_text_resolution_records,
    variant_masks,
    warmup_steps,
)
from effects.domain.ability_encoder import (
    AbilityEncoder,
    AbilityEncoderConfig,
    surface_of,
)
from effects.domain.ability_tokenizer import AbilityTokenizer
from effects.domain.effect_head_input import (
    SlotKind,
    act_features,
    card_features,
    global_features,
    player_features,
)
from effects.domain.effect_model import (
    EffectModel,
    EffectModelConfig,
    active_fields,
    entity_target_tensors,
    per_entity_loss,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.records import EffectRecord
from effects.infrastructure.effect_model_store import (
    EffectCheckpoint,
    EffectModelStore,
    SplitProvenance,
    content_hash,
)
from effects.infrastructure.model_runner import IDENTITY_TABLE_KEY
from effects.infrastructure.shard_sweep import sweep
from effects.infrastructure.sidecar_io import SidecarCache
from price_predictor.infrastructure.tokenizer_store import load_vocabulary
from price_predictor.infrastructure.torch_training import clip_per_group

logger = logging.getLogger(__name__)

#: Records kept aside to measure feature widths from, before any shard loads.
PROBE_RECORDS = 64


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


def module_grad_norms(modules: Mapping[str, torch.nn.Module]) -> dict[str, float]:
    """Pre-clip L2 gradient norm of each module, for reporting only.

    Measured here rather than read off the optimizer because the optimizer holds
    exactly one parameter group: ``clip_per_group`` then reports a single number
    covering everything, which cannot answer the question the number is for. The
    effect head's loss reaches the ability encoder through the ``e`` path and no
    other, so an encoder norm near zero beside a healthy head norm is the
    signature of that path being cut — and a run with it cut still reports a
    falling loss.

    Deliberately not a change to how gradients are clipped. Giving the optimizer
    one group per module would clip each at ``max_norm`` separately instead of
    the whole together, which is a different optimizer rather than a different
    log line.

    Summed on the device and read once per module, on the single step per shard
    that reports.
    """
    norms: dict[str, float] = {}
    for name, module in modules.items():
        squared = None
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            term = parameter.grad.detach().pow(2).sum()
            squared = term if squared is None else squared + term
        if squared is not None:
            norms[name] = float(squared) ** 0.5
    return norms


def _format_norms(norms: Mapping[str, float]) -> str:
    """The pre-clip gradient norms, or nothing before the first backward."""
    if not norms:
        return ""
    body = ", ".join(f"{name} {value:.2f}" for name, value in norms.items())
    return f"\n  |g| {body} (clipped together at {MAX_GRAD_NORM:g})"


def _format_parts(parts: Mapping[str, float]) -> str:
    """Each field's loss term, largest first.

    Ordered by size rather than by name because the question this line answers
    is which term the loss is made of: a single field carrying nearly all of it
    is what a blow-up looks like, and alphabetical order buries that.
    """
    if not parts:
        return ""
    ranked = sorted(parts.items(), key=lambda kv: -kv[1])
    body = ", ".join(f"{name} {value:.3f}" for name, value in ranked)
    return f"\n  fields: {body}"


class TrainingLoop:
    """Owns one training run end to end.

    Holds one shard of the corpus at a time. The corpus is 14M records and would
    need hundreds of gigabytes resident; a shard needs about one, and an epoch is
    a walk over ``--shards-per-epoch`` of them rather than a pass over the whole.
    """

    def __init__(
        self,
        config: TrainEffectModelConfig,
        *,
        held_out: HeldOutCards,
        inherited: CorpusSplit | None,
        validation_shards: Sequence[Path],
        training_shards: Sequence[Path],
        rarity: Mapping[str, int] | None = None,
        corpus_digest: str = "",
    ) -> None:
        self.config = config
        self.held_out = held_out
        self.validation_shards = list(validation_shards)
        self.training_shards = list(training_shards)
        #: A curated dataset's corpus-wide ``text -> games`` table (FR-146),
        #: threaded into every ``sample_weights`` call this run makes. ``None``
        #: for an ordinary ``--records-dir`` run, which weights by the resident
        #: shard's own counts instead.
        self.rarity = rarity
        #: Whether this run has already said how much of a shard the
        #: rarity table names. Once per run rather than once per shard:
        #: a table matching nothing is otherwise invisible, since every
        #: weight falls back to the shard's own count and the run looks
        #: exactly like a healthy one.
        self._rarity_reported = False
        #: The curated dataset's manifest digest at read time (FR-147), read
        #: into the checkpoint's provenance by ``_provenance`` below. ``""``
        #: for an ordinary ``--records-dir`` run, which has no dataset to pin.
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
        self.identity_table = None
        #: Resolved once here rather than read from the config at each use, so
        #: the value logged at startup and written to the checkpoint is the one
        #: every draw actually used.
        self.seed = (
            config.seed if config.seed is not None
            else random.SystemRandom().getrandbits(32)
        )
        self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.context_cache = (
            ContextCache(config.cache_refresh) if config.context_cache else None
        )
        self.card_disjoint: list = []
        self.game_disjoint: list = []
        self.probe: list = []
        self.present: set[str] = set()

    # ── setup ───────────────────────────────────────────────────────────

    def _build_tokenizer(self) -> AbilityTokenizer:
        vocab = load_vocabulary(Path(self.config.vocab_path))
        definitions = {}
        keyword_path = Path(self.config.keyword_definitions)
        if keyword_path.exists():
            from effects.application.extract_keyword_definitions import (
                load_keyword_definitions,
            )

            definitions = load_keyword_definitions(keyword_path)
        return AbilityTokenizer(vocab, definitions)

    def _build_sidecars(self) -> SidecarCache:
        roots = {Path(f).name: Path(f) for f in self.config.cards_folders}
        if self.config.variant_scripts:
            roots["variant-scripts"] = Path(self.config.variant_scripts)
        return SidecarCache(roots)

    def _class_quota(self) -> dict[str, int]:
        """How many records of each class a stratum's sample should hold.

        The **training** mixture, because that is the distribution the loss is
        optimized against and a validation number drawn from any other one
        measures a different objective. ``build-corpus`` mixes the training
        stratum and leaves both validation strata in the proportions collection
        produced, which is five times the share of ``playability-decision`` and
        a sixth the share of ``combat``. Scored on that, a validation loss is
        mostly a reading of one class training spends a tenth of its batches on,
        and it barely moves when the classes training works hardest on improve.
        """
        wanted = parse_kind_mix(self.config.kind_mix)
        return {
            name: max(1, round(share * VALIDATION_RECORDS_PER_STRATUM))
            for name, share in wanted.items()
        }

    def _sample_is_full(self, pools: Mapping[str, Mapping[str, list]]) -> bool:
        """Whether both strata can fill every class the mixture asks for."""
        quota = self._class_quota()
        return all(
            len(pools[stratum].get(name, ())) >= size
            for stratum in ("card-disjoint", "game-disjoint")
            for name, size in quota.items()
        )

    def _capture_validation(self) -> None:
        """Read reserved shards once and keep a mixture-matched sample of each.

        Captured once rather than redrawn per epoch because ``--patience``
        compares this epoch's loss against the best so far. A validation set that
        changed every epoch would make that comparison measure which records got
        drawn rather than whether the model improved.

        The shards are **shuffled** before reading, and only as many are read as
        the sample needs. A stratum's shard list is in path order, so its opening
        shards are one collection run's: taking the sample from the front drew
        every card-disjoint validation record from the depleted run that sorts
        first — 0.9% of the stratum, eight games — and then read the remaining
        1,400 shards to keep nothing from them. Reading stops early only when the
        split came from a curated dataset's manifest; derived from the shards
        themselves it has to see every one of them, since a shard it skipped is a
        game the checkpoint would fail to enumerate.
        """
        quota = self._class_quota()
        pools: dict[str, dict[str, list]] = {
            "card-disjoint": defaultdict(list),
            "game-disjoint": defaultdict(list),
        }
        inherited = self.accumulator.inherited
        shards = list(self.validation_shards)
        if inherited:
            random.Random(f"{self.seed}:validation").shuffle(shards)
        workers = self.config.workers or (os.cpu_count() or 1)
        started = time.perf_counter()
        done = 0
        logger.info(
            "Sweeping %d reserved shards across %d workers for the validation "
            "sample.", len(shards), min(workers, max(1, len(shards))),
        )
        for digest in sweep(
            shards,
            held_out=self.held_out,
            # Given to the workers when the manifest already decided it, so a
            # worker routes rather than deriving a split it can only see part of.
            split=self.accumulator.split() if inherited else None,
            quota=quota,
            probe_cap=PROBE_RECORDS,
            workers=workers,
        ):
            done += len(digest.shards)
            self.present.update(digest.classes)
            for stratum in ("card-disjoint", "game-disjoint"):
                for name, records in digest.sample[stratum].items():
                    room = quota.get(name, 0) - len(pools[stratum][name])
                    if room > 0:
                        pools[stratum][name].extend(records[:room])
            if not inherited:
                # The derived split is the union of what the workers routed, and
                # it has to be complete, so this sweep reads every shard.
                self.accumulator.note_games(
                    card_disjoint=digest.games["card-disjoint"],
                    game_disjoint=digest.games["game-disjoint"],
                )
            for record in digest.probe:
                if len(self.probe) < PROBE_RECORDS:
                    self.probe.append(record)
            logger.info(
                "validation sweep %d/%d shards | %.0fs | sampled "
                "card-disjoint %d/%d, game-disjoint %d/%d",
                done, len(shards), time.perf_counter() - started,
                sum(len(v) for v in pools["card-disjoint"].values()),
                sum(quota.values()),
                sum(len(v) for v in pools["game-disjoint"].values()),
                sum(quota.values()),
            )
            if inherited and self._sample_is_full(pools):
                logger.info(
                    "Both validation strata match the training mixture after "
                    "%d of %d shards; the rest are not read.",
                    done, len(shards),
                )
                break

        for stratum, target in (
            ("card-disjoint", self.card_disjoint),
            ("game-disjoint", self.game_disjoint),
        ):
            for name in sorted(pools[stratum]):
                target.extend(pools[stratum][name])
            # Shuffled, because the batches cut out of this list have to mix
            # classes the way a training batch does. Collected class by class
            # and left in that order, every batch would hold one class, and
            # `active_fields` would score each against only the fields that
            # class supervises — a different and smaller objective per batch.
            random.Random(f"{self.seed}:{stratum}").shuffle(target)
            short = {
                name: size - len(pools[stratum].get(name, ()))
                for name, size in quota.items()
                if len(pools[stratum].get(name, ())) < size
            }
            logger.info(
                "%s validation: %d records over %d games, %d classes%s",
                stratum, len(target), len({r.game_id for r in target}),
                len(pools[stratum]),
                "" if not short else
                " — short of the mixture on " + ", ".join(
                    f"{name} by {count}" for name, count in sorted(short.items())
                ),
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

    def _batcher(self, tokenizer, sidecars, widths) -> SurfaceBatcher:
        """The records-to-inputs pipeline, shared with the gate-1 evaluator."""
        return SurfaceBatcher(
            tokenizer=tokenizer,
            sidecars=sidecars,
            masks=self.masks,
            surface=self.surface,
            e_dim=self.config.e_dim,
            widths=widths,
            device=self.device,
            withhold_keyword=self.config.withhold_keyword,
            keyword_expand_p=self.config.keyword_expand_p,
            context_dropout=self.config.context_dropout,
            rng=self.rng,
            identity_table=self.identity_table,
        )

    # ── stepping ────────────────────────────────────────────────────────

    def _loss_for(
        self, plan, encoder, model, tokenizer, sidecars, widths, step,
        *, report_parts: bool = False,
    ):
        """``(loss, parts)`` for one planned batch, or ``None`` when empty.

        ``report_parts`` reads each field's term back as a float, which is one
        device synchronization per active field. Passed only on the batch a
        progress line is about to report, so the cost lands a few times a
        minute rather than on all five thousand steps of an epoch.
        """
        records = plan.records
        if not records:
            return None
        batch, surfaces = self._batcher(tokenizer, sidecars, widths).build(
            records, encoder,
        )
        hidden = model(**batch)
        outputs = model.per_entity(hidden)

        present = {sampling_class(record) for record in records}
        fields = active_fields(
            present_classes=frozenset(present), step=step,
            curriculum_step=self.config.curriculum_step,
        )
        targets = [derive_targets(record) for record in records]
        gate, field_targets, mask, index = entity_target_tensors(
            surfaces, targets, fields,
        )
        # One batched gather rather than a slice per row: `index` selects each
        # surface's [CARD] and [PLAYER] columns, and the rows are independent.
        index = index.to(self.device)
        gathered = outputs.gather(
            1, index.unsqueeze(-1).expand(-1, -1, outputs.shape[-1]),
        )
        loss, parts = per_entity_loss(
            gathered, gate.to(self.device),
            {k: v.to(self.device) for k, v in field_targets.items()},
            mask.to(self.device), fields=fields,
            report_parts=report_parts,
        )
        return loss, parts

    def _pools(self, records: list, sidecars: SidecarCache) -> tuple[dict, dict]:
        """Sampling pools and rarity weights for one shard's training records.

        Keyed by the acting ability's text (:func:`ability_text_of`), not by
        ``record_id``: a record id is unique per record, so keying on it would
        give ``effective_games`` a count of exactly 1 for every key and every
        record the same weight, silently disabling rarity weighting outright.

        With a curated dataset's rarity table (``self.rarity``), weighting
        reads it text by text, falling back to this shard's own count for a
        text the table does not name — a shard collected after the table was
        built still weights sanely rather than at zero. Without one, weighting
        counts the games of the shard in hand rather than the games of the
        whole corpus, which is the one thing reading shard by shard costs: an
        ability that is rare corpus-wide but appears in several of this
        shard's games is under-weighted while this shard is resident.
        Abilities that are common are common in every shard, so they are
        unaffected either way.
        """
        def text_of(record: EffectRecord) -> str | None:
            return ability_text_of(record, sidecars, self.surface)

        if self.rarity is not None and not self._rarity_reported:
            self._rarity_reported = True
            found, distinct = rarity_coverage(
                (text_of(record) for record in records), self.rarity,
            )
            share = 100.0 * found / distinct if distinct else 0.0
            report = logger.info if found else logger.warning
            report(
                "Rarity table names %d of this shard's %d distinct ability "
                "text(s) (%.1f%%); the rest weigh by this shard's own game "
                "count. A table naming none of them means the dataset and the "
                "vocabulary disagree about the encoding surface.",
                found, distinct, share,
            )

        pools: dict[str, list] = defaultdict(list)
        for record in records:
            pools[sampling_class(record)].append(record)
        weights = {
            name: sample_weights(group, text_of, rarity=self.rarity)
            for name, group in pools.items()
        }
        return dict(pools), weights

    # ── the run ─────────────────────────────────────────────────────────

    def execute(self) -> int:
        logger.info(
            "Reading %d reserved shards for validation before training starts.",
            len(self.validation_shards),
        )
        self._capture_validation()
        if not self.probe:
            logger.error(
                "The %d reserved shards hold no records, so there is nothing to "
                "measure feature widths from.", len(self.validation_shards),
            )
            return 1
        if not self.card_disjoint and not self.game_disjoint:
            logger.warning(
                "No validation records in the reserved shards: every game there "
                "is a training game. Early stopping has nothing to read, so the "
                "run will train for all %d epochs.", self.config.epochs,
            )

        logger.info(
            "Seed %d (%s), %d of %d training shards per epoch, drawn across "
            "the corpus.",
            self.seed,
            "given" if self.config.seed is not None else "drawn; pass "
            f"--seed {self.seed} to repeat this run",
            min(self.config.shards_per_epoch, len(self.training_shards)),
            len(self.training_shards),
        )
        mix = renormalize_mix(parse_kind_mix(self.config.kind_mix), self.present)
        logger.info("Classes present: %s", ", ".join(sorted(self.present)))
        logger.info(
            "Sampling mixture over the classes present: %s",
            ", ".join(f"{name} {share:.0%}" for name, share in sorted(mix.items())),
        )

        tokenizer = self._build_tokenizer()
        sidecars = self._build_sidecars()
        widths = self._feature_widths(self.probe)

        # Sized after the sidecars exist, because an acting line's text is what
        # says whether a record is in gate 1's slice. An empty stratum stops the
        # run: left alone it trains for its full patience, validates `nan` every
        # epoch, and saves no checkpoint at all.
        text_of = self._batcher(tokenizer, sidecars, widths).text_of
        warning = check_holdout(
            card_disjoint_records=len(self.card_disjoint),
            unique_text_records=unique_text_resolution_records(
                self.card_disjoint, self.held_out.texts, text_of=text_of,
            ),
            minimum=self.config.min_holdout_records,
        )
        if warning:
            logger.warning("%s", warning)

        encoder_config = AbilityEncoderConfig(
            vocab_size=tokenizer.vocab_size,
            e_dim=self.config.e_dim,
            e_noise=self.config.e_noise,
        )
        model_config = EffectModelConfig(
            global_features=widths[SlotKind.GLOBAL],
            act_features=widths[SlotKind.ACT],
            player_features=widths[SlotKind.PLAYER],
            card_features=widths[SlotKind.CARD],
            e_dim=self.config.e_dim,
            vocab_size=tokenizer.vocab_size,
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

        trainable = [*encoder.parameters(), *model.parameters()]
        if self.identity_table is not None:
            trainable += list(self.identity_table.parameters())
        optimizer = torch.optim.AdamW(
            trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        warmup = warmup_steps(
            epochs=self.config.epochs, steps_per_epoch=self.config.steps_per_epoch,
        )

        stopper = EarlyStopper(self.config.patience)
        store = EffectModelStore(self.config.resolved_model_output())
        step = 0

        for epoch in range(1, self.config.epochs + 1):
            encoder.train()
            model.train()
            # Accumulated on the device and read once at the end of the epoch.
            # Reading it per step would synchronize once per batch for a number
            # nothing looks at until the epoch closes.
            running = torch.zeros((), device=self.device)
            taken = 0
            shards = epoch_shards(
                self.training_shards, epoch=epoch,
                per_epoch=self.config.shards_per_epoch,
                seed=self.seed,
            )
            allocation = steps_per_shard(self.config.steps_per_epoch, len(shards))
            for position, (shard, budget) in enumerate(
                zip(shards, allocation), start=1,
            ):
                if budget <= 0:
                    continue
                step, taken = self._train_on_shard(
                    shard, budget, mix=mix, encoder=encoder, model=model,
                    tokenizer=tokenizer, sidecars=sidecars, widths=widths,
                    optimizer=optimizer, warmup=warmup, running=running,
                    step=step, taken=taken, epoch=epoch, position=position,
                    of=len(shards),
                )

            card_parts: dict[str, float] = {}
            result = EpochResult(
                epoch=epoch,
                train_loss=float(running) / max(taken, 1),
                card_disjoint_loss=self._validate(
                    self.card_disjoint, encoder, model, tokenizer, sidecars,
                    widths, step, parts=card_parts,
                ),
                game_disjoint_loss=self._validate(
                    self.game_disjoint, encoder, model, tokenizer, sidecars,
                    widths, step,
                ),
            )
            logger.info(
                "epoch %d | train %.4f | card-disjoint %.4f | "
                "game-disjoint %.4f%s",
                result.epoch, result.train_loss, result.card_disjoint_loss,
                result.game_disjoint_loss, _format_parts(card_parts),
            )
            if sidecars.unresolved:
                worst = sorted(
                    sidecars.unresolved.items(), key=lambda kv: -kv[1],
                )[:3]
                logger.info(
                    "%d ability scripts the converted corpus does not hold, "
                    "%d lookups so far; those abilities reach the model as no "
                    "text. Most asked: %s",
                    len(sidecars.unresolved),
                    sum(sidecars.unresolved.values()),
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
                ))
            if stopper.should_stop:
                logger.info(
                    "Early stop: %d epochs without a new card-disjoint best",
                    self.config.patience,
                )
                break
        return 0

    def _train_on_shard(
        self, shard, budget, *, mix, encoder, model, tokenizer, sidecars,
        widths, optimizer, warmup, running, step, taken, epoch, position, of,
    ) -> tuple[int, int]:
        """Load one shard, take ``budget`` steps on it, and let it go.

        The shard is released before the next one is read, so resident memory
        stays at one shard however long the run and however large the corpus.

        Returns the advanced ``(step, taken)`` counters.
        """
        started = time.perf_counter()
        records = load_shard(shard)
        self.accumulator.note_shard(records, self.held_out, reserved=False)
        split = self.accumulator.split()
        training = [r for r in records if split.is_training_game(r.game_id)]
        held_back = len(records) - len(training)
        del records
        loaded = time.perf_counter() - started

        if not training:
            logger.info(
                "epoch %d | shard %d/%d %s | nothing trainable, all %d records "
                "held back | skipped",
                epoch, position, of, shard.name, held_back,
            )
            return step, taken

        pools, weights = self._pools(training, sidecars)
        # Accumulated on the device like the epoch's own running loss, and read
        # back once per progress line rather than once per step.
        # Accumulated on the device like the epoch's own running loss, and read
        # back once, on the line this shard logs when it ends.
        shard_loss = torch.zeros((), device=self.device)
        shard_steps = 0
        parts: dict[str, float] = {}
        norms: dict[str, float] = {}
        learning_rate = 0.0
        for index in range(budget):
            for group in optimizer.param_groups:
                group["lr"] = learning_rate = learning_rate_at(
                    step, warmup=warmup,
                )
            plan = plan_batch(
                pools, weights, mix,
                batch_size=self.config.batch_size, rng=self.rng,
            )
            # Decided before the forward pass: the decomposition has to be
            # asked for while the loss is being computed, not after.
            #
            # The last step of every shard reports, whatever the clock says, so
            # the shard's own line always carries a decomposition. A wall-clock
            # interval alone was silent here: it was written when a shard took
            # 278 steps over three minutes, and a shard now takes twenty over
            # seconds, so the window never elapsed and no shard ever reported.
            due = reports_now(index=index, budget=budget)
            computed = self._loss_for(
                plan, encoder, model, tokenizer, sidecars, widths, step,
                report_parts=due,
            )
            if computed is None:
                continue
            loss, batch_parts = computed
            (loss / self.config.grad_accum).backward()
            if due:
                # Read before the clip, and per module rather than per optimizer
                # group, because the optimizer holds exactly one group.
                norms = module_grad_norms(
                    {"encoder": encoder, "head": model}
                    | ({"identity": self.identity_table}
                       if self.identity_table is not None else {})
                )
            if (step + 1) % self.config.grad_accum == 0:
                clip_per_group(optimizer, max_norm=MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += loss.detach()
            shard_loss += loss.detach()
            shard_steps += 1
            step += 1
            taken += 1
            if batch_parts:
                parts = batch_parts
            if self.context_cache is not None:
                self.context_cache.note_batch()

        trained = time.perf_counter() - started - loaded
        logger.info(
            "epoch %d | shard %d/%d %s | %d records trainable, %d held back | "
            "%d steps | loss %.4f | lr %.2e | %.1f steps/s | "
            "load %.1fs, train %.1fs%s%s",
            epoch, position, of, shard.name, len(training), held_back, budget,
            float(shard_loss) / shard_steps if shard_steps else float("nan"),
            learning_rate, budget / trained if trained > 0 else float("nan"),
            loaded, trained, _format_norms(norms), _format_parts(parts),
        )
        del training, pools, weights
        return step, taken

    def _validate(
        self, records, encoder, model, tokenizer, sidecars, widths, step,
        *, parts: dict[str, float] | None = None,
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

        Fixed-size batches over a shuffled sample fix both. ``parts`` collects
        the loss by field, averaged over batches, which is what says *which*
        term is not moving when the total is not moving.

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

        Both strata are enumerated from the games this run actually read, so a
        run that stops early names fewer games than a full one. That is what the
        evaluator needs: it scores only games the checkpoint recorded, and a
        game the model never saw is neither training nor validation.
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
            holdout_permille=self.config.holdout_permille,
            holdout_max_carriers=self.config.holdout_max_carriers,
            corpus_path=self.config.corpus or "",
            corpus_digest=self.corpus_digest,
        )


def config_summary(config: TrainEffectModelConfig) -> dict:
    """The run's settings, for the log line and the checkpoint's extras."""
    return {k: str(v) for k, v in asdict(config).items()}
