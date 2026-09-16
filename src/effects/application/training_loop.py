"""The training loop itself, separated from the pure functions it drives.

``train_effect_model`` owns reading the corpus and the schedule — all pure
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
    WEIGHT_DECAY,
    ContextCache,
    CorpusSplit,
    EarlyStopper,
    EpochResult,
    HeldOutCards,
    SplitAccumulator,
    TrainEffectModelConfig,
    ability_text_of,
    batches_without_replacement,
    check_holdout,
    epoch_shards,
    learning_rate_at,
    load_shard,
    rarity_coverage,
    sample_weights,
    sampling_class,
    steps_per_shard,
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
        training_shards: Sequence[Path],
        validation_samples: Mapping[str, Path],
        holdout_permille: int,
        holdout_max_carriers: int,
        gate_one_records: int = 0,
        rarity: Mapping[str, int] | None = None,
        corpus_digest: str = "",
    ) -> None:
        self.config = config
        self.held_out = held_out
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
        #: Whether this run has already said how much of a shard the
        #: rarity table names. Once per run rather than once per shard:
        #: a table matching nothing is otherwise invisible, since every
        #: weight falls back to the shard's own count and the run looks
        #: exactly like a healthy one.
        self._rarity_reported = False
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

    def _weighted(self, records: list, sidecars: SidecarCache) -> list[float]:
        """Rarity weights for a shard's records, from the manifest's table (FR-086)."""
        def text_of(record: EffectRecord) -> str | None:
            return ability_text_of(record, sidecars, self.surface)

        if self.rarity is not None and not self._rarity_reported:
            self._rarity_reported = True
            found, distinct = rarity_coverage((text_of(r) for r in records), self.rarity)
            share = 100.0 * found / distinct if distinct else 0.0
            report = logger.info if found else logger.warning
            report(
                "Rarity table names %d of this shard's %d distinct ability "
                "text(s) (%.1f%%); the rest weigh by this shard's own game count.",
                found, distinct, share,
            )
        return sample_weights(records, text_of, rarity=self.rarity)

    # ── the run ─────────────────────────────────────────────────────────

    def execute(self) -> int:
        logger.info(
            "Reading the corpus's validation samples: %s.",
            ", ".join(str(path) for path in self.validation_samples.values()),
        )
        self._load_validation()
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
                    shard, budget, encoder=encoder, model=model,
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
        self, shard, budget, *, encoder, model, tokenizer, sidecars,
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

        weights = self._weighted(training, sidecars)
        batches = batches_without_replacement(
            training, weights, batch_size=self.config.batch_size, rng=self.rng,
        )
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
            plan = next(batches)
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
            if (step + 1) % self.config.grad_accum == 0:
                clipped = clip_per_group(optimizer, max_norm=MAX_GRAD_NORM)
                if due:
                    norms = clipped
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
        del training, weights, batches
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
        )


def config_summary(config: TrainEffectModelConfig) -> dict:
    """The run's settings, for the log line and the checkpoint's extras."""
    return {k: str(v) for k, v in asdict(config).items()}
