"""The host half of a training step, wherever it runs.

A step splits into a host half — the plan, tokenizing, the surfaces, every
target, the MLM mask, all CPU tensors — and a device half: the encoder, the
noise on ``e``, the model, the losses, backward. This module owns the host half
and nothing that reads the weights, so the same code runs on a thread of the
training process (``--prefetch-workers 0``) or in a worker process of its own,
and the training process does only the device half.

Moving it into processes is what made its randomness explicit. Every draw the
host half takes — the shard's weighted shuffle, keyword expansion, context
dropout, the MLM mask — used to come from one generator shared with the
training process, in step order, which only one thread can honour. Each draw
now comes from a generator seeded by ``(seed, epoch, shard position, step)``
alone, so a step's inputs are a function of where it sits in the run and not of
which process prepared it, or how many there were.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch

from effects.application.surface_batching import (
    MaskedTokens,
    PreparedBatch,
    SurfaceBatcher,
    pinned,
)
from effects.application.train_effect_model import (
    CorpusSplit,
    HeldOutCards,
    TrainEffectModelConfig,
    VariantMasks,
    ability_text_of,
    batches_without_replacement,
    effective_games,
    rarity_coverage,
    sample_weights,
    sampling_class,
    shard_games,
)
from effects.domain.ability_encoder import TruncationLog, surface_of
from effects.domain.ability_tokenizer import (
    INFERENCE_KEYWORD_EXPAND_P,
    AbilityTokenizer,
)
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_model import (
    CREATED_OBJECTS_WIDTH,
    VERDICT_WIDTH,
    FieldSpec,
    active_fields,
    entity_target_tensors,
)
from effects.domain.effect_targets import (
    created_objects_targets,
    derive_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.rarity import rarity_bucket
from effects.domain.records import EffectRecord
from effects.domain.rule_families import rule_family
from effects.domain.value_targets import params, segments, value_targets
from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots
from price_predictor.infrastructure.tokenizer_store import load_vocabulary

#: Rarity-bucket label for a record with no acting text (combat, legality).
NO_TEXT_BUCKET = "no-text"


def plan_rng(seed: int, epoch: int, position: int) -> random.Random:
    """The generator one shard's weighted shuffles draw from.

    Keyed by the shard's place in the run rather than handed down from the
    previous shard, so a shard's plans do not depend on how many steps the
    shards before it took, or on which process planned them.
    """
    # A string rather than a tuple: 3.14 seeds only from None, int, float,
    # str, bytes and bytearray, and hashes a string the same in every process.
    return random.Random(f"{seed}:{epoch}:{position}:plan")


def step_rng(seed: int, epoch: int, position: int, index: int) -> random.Random:
    """The generator one step's host half draws from.

    Keyword expansion, context dropout and the MLM mask, in that order, all
    from this one; nothing else draws from it.
    """
    return random.Random(f"{seed}:{epoch}:{position}:{index}")


@dataclass
class HeadTargets:
    """The targets of every loss term beside the per-entity one, on the host.

    Each is None where its head is not computed for this batch. Built with the
    step's other targets, so the device half reads them rather than deriving
    them between two pieces of GPU work.
    """

    #: ``(target, mask)`` for the verdict head (FR-060a).
    verdict: tuple[torch.Tensor, torch.Tensor] | None = None
    #: ``(target, mask)`` for the created-objects head (FR-060a).
    created: tuple[torch.Tensor, torch.Tensor] | None = None
    #: ``(target, mask)`` for the value head, one row per text (FR-058).
    value: tuple[torch.Tensor, torch.Tensor] | None = None
    #: ``(types, keys)`` for the script-API head, one row per text (FR-060b).
    api: tuple[torch.Tensor, torch.Tensor] | None = None
    #: The MLM pass's masked tokens (FR-060b).
    mlm: MaskedTokens | None = None

    def pin(self) -> None:
        """Every target in page-locked memory; see ``pinned``."""
        self.verdict = pinned(self.verdict)
        self.created = pinned(self.created)
        self.value = pinned(self.value)
        self.api = pinned(self.api)
        if self.mlm is not None:
            self.mlm.pin()


@dataclass
class PreparedStep:
    """One step's host half: what ``_loss_for`` needs besides the model.

    Everything here is a function of the planned records, the sidecars, the
    tokenizer and the step's generator, and nothing of the weights, which is
    what lets it be built ahead of time, on another thread or in another
    process. ``e`` is not among it: the encoder runs in the device half, so
    the effect loss still reaches it.
    """

    #: The batcher that prepared it, which validation's device half reuses.
    #: None on a training step: what it carried besides tensors is the
    #: tokenizer and the sidecars, which the device half never reads, so the
    #: training loop hands every step one device-side batcher of its own.
    batcher: SurfaceBatcher | None
    batch: PreparedBatch
    fields: tuple[FieldSpec, ...]
    #: ``entity_target_tensors``' four outputs.
    gate: torch.Tensor
    field_targets: dict[str, torch.Tensor]
    mask: torch.Tensor
    index: torch.Tensor
    heads: HeadTargets

    def slim(self) -> PreparedStep:
        """Only what the device half reads: tensors and the rows' texts.

        Drops the batcher, each text's sidecar line, the tokenized lines and
        the surfaces — host-side intermediates the targets were built from,
        and the bulk of what a worker would otherwise pickle every step.
        """
        self.batcher = None
        self.batch.texts = {}
        self.batch.prepared = []
        self.batch.surfaces = []
        return self

    def pin(self) -> None:
        """Every tensor the device half copies, in page-locked memory."""
        self.batch.pin()
        self.field_targets = pinned(self.field_targets)
        self.index = self.index.pin_memory()
        self.heads.pin()


@dataclass(frozen=True)
class ShardTask:
    """One shard's share of an epoch: everything a preparer needs to plan it.

    Small and picklable, because it is what the training process sends a
    worker; the records it names are read where they are prepared.
    """

    shard: Path
    epoch: int
    #: The shard's 1-based place in its epoch's draw; seeds its generators.
    position: int
    budget: int
    #: The epoch's field set (FR-082).
    fields: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class ShardOpened:
    """The first message of a shard: its split, before any step."""

    trainable: int
    held_back: int
    #: Games naming a held-out card, for the split accumulator.
    tainted: frozenset[str]
    #: ``(found, distinct)``: how many of the trainable records' distinct
    #: ability texts the rarity table names; None without a table.
    rarity_coverage: tuple[int, int] | None


@dataclass
class StepReady:
    """One prepared step, with the epoch-line shares of the records it planned."""

    prepared: PreparedStep | None
    #: Trained records by rarity bucket and by rule family (FR-055).
    buckets: dict[str, int]
    families: dict[str, int]


@dataclass(frozen=True)
class ShardClosed:
    """A worker's last message of a shard: what its own caches tallied.

    The training process logs both, but in a worker they were counted against
    the worker's sidecar cache and truncation log rather than its own.
    """

    #: ``script_file -> lookups`` naming an unconverted script, this shard only.
    unresolved: dict[str, int] = field(default_factory=dict)
    #: ``(text, keys)`` for each text first truncated during this shard.
    truncated: tuple[tuple[str, tuple], ...] = ()


def one_ahead(iterator: Iterator, pool: ThreadPoolExecutor) -> Iterator:
    """``iterator``, each item computed on ``pool`` while the last is used.

    The in-process path's prefetch: the next step's host half runs on the
    pool's thread while the caller runs this one's device half, whose torch
    calls release the interpreter lock. An exception the iterator raises is
    re-raised here, at the item it belongs to.
    """
    end = object()
    pending = pool.submit(next, iterator, end)
    while True:
        item = pending.result()
        if item is end:
            return
        pending = pool.submit(next, iterator, end)
        yield item


class StepPreparation:
    """The host half's methods, shared by the training loop and its workers.

    A mixin over the attributes both set: ``config``, ``seed``, ``surface``,
    ``masks``, ``held_out``, ``inherited``, ``rarity``, ``api_types``,
    ``param_keys``, ``truncations``, ``noise``, ``identity_table`` and
    ``device``. The worker's ``noise``, ``identity_table`` and ``device`` are
    inert — they are read only by the device half, which never runs there.
    """

    config: TrainEffectModelConfig
    seed: int
    surface: str
    masks: VariantMasks
    held_out: HeldOutCards
    inherited: CorpusSplit | None
    rarity: Mapping[str, int] | None
    api_types: list[str]
    param_keys: list[str]
    truncations: TruncationLog | None

    def _init_preparation(self) -> None:
        """The host half's own caches; every subclass calls this."""
        self._api_targets: dict[str, tuple[int, tuple[int, ...]]] = {}
        self._resolver: KeywordResolver | None = None
        self._resolver_for = None

    # ── resources ───────────────────────────────────────────────────────

    def _build_tokenizer(self) -> AbilityTokenizer:
        vocab = load_vocabulary(Path(self.config.vocab_path))
        definitions = {}
        keyword_path = Path(self.config.keyword_definitions)
        if keyword_path.exists():
            from effects.application.extract_keyword_definitions import (
                load_keyword_definitions,
            )

            definitions = load_keyword_definitions(keyword_path)
        return AbilityTokenizer(vocab, definitions, surface=self.surface)

    def _build_sidecars(self) -> SidecarCache:
        return SidecarCache(sidecar_roots(
            self.config.cards_folders, self.config.variant_scripts,
        ))

    def _batcher(
        self, tokenizer, sidecars, widths, *, training: bool = True,
        rng: random.Random | None = None,
    ) -> SurfaceBatcher:
        """The records-to-inputs pipeline, built the same way for other callers.

        ``training=False`` turns the two augmentations off: keyword expansion
        and context dropout are there to vary what the model sees from one
        step to the next, and a validation number they varied would move with
        the draw rather than with the model. The withheld keyword stays
        withheld — it is a split, not an augmentation, and a validation batch
        that handed it back would score the one thing training never saw.

        ``rng`` is the training step's own generator (``step_rng``). Scoring
        takes a dedicated stream instead: sharing the training generator made
        how many validation batches ran decide which records the next epoch
        drew, so a run's training path moved with the size of its samples.
        """
        return SurfaceBatcher(
            tokenizer=tokenizer,
            sidecars=sidecars,
            masks=self.masks,
            surface=self.surface,
            e_dim=self.config.e_dim,
            widths=widths,
            device=self.device,
            withhold_keyword=self.config.withhold_keyword,
            keyword_expand_p=(
                self.config.keyword_expand_p if training
                else INFERENCE_KEYWORD_EXPAND_P
            ),
            context_dropout=self.config.context_dropout if training else 0.0,
            noise=self.noise if training else None,
            rng=(
                (rng or random.Random(f"{self.seed}:unplanned")) if training
                else random.Random(f"{self.seed}:validation")
            ),
            identity_table=self.identity_table,
            truncations=self.truncations,
        )

    # ── one step ────────────────────────────────────────────────────────

    def _prepare_step(
        self, plan, tokenizer, sidecars, widths, step, *, fields=None,
        training: bool = True, heads: frozenset[str] = frozenset(),
        pin: bool = False, rng: random.Random | None = None,
        slim: bool = False,
    ) -> PreparedStep | None:
        """One step's host half, or ``None`` when the plan holds no records.

        Pure CPU: no model, no device, nothing that reads the weights. Every
        draw it takes comes from ``rng`` — the keyword expansion, the context
        dropout and the MLM mask, in that order.

        ``heads`` names the training-only heads to build targets for, read off
        the model by the training loop. ``pin`` moves the tensors the device
        half copies into page-locked memory, which is what lets those copies
        run without holding the host; only the training process can, since
        page-locked memory does not survive a trip between processes. ``slim``
        keeps only what the device half reads (``PreparedStep.slim``).
        """
        records = plan.records
        if not records:
            return None
        if fields is None:
            present = {sampling_class(record) for record in records}
            fields = active_fields(
                present_classes=frozenset(present), step=step,
                curriculum_step=self.config.curriculum_step,
            )
        batcher = self._batcher(
            tokenizer, sidecars, widths, training=training, rng=rng,
        )
        batch = batcher.prepare(records)
        targets = [derive_targets(record) for record in records]
        gate, field_targets, mask, index = entity_target_tensors(
            batch.surfaces, targets, fields,
        )
        head_targets = self._head_targets(
            records, batcher, batch, training=training, heads=heads,
        )
        prepared = PreparedStep(
            batcher=batcher, batch=batch, fields=fields, gate=gate,
            field_targets=field_targets, mask=mask, index=index,
            heads=head_targets,
        )
        if slim:
            prepared.slim()
        if pin:
            prepared.pin()
        return prepared

    def _head_targets(
        self, records, batcher, batch: PreparedBatch, *, training: bool,
        heads: frozenset[str],
    ) -> HeadTargets:
        """The targets ``_head_terms`` scores, built on the host.

        The verdict head reads ``[ACT]`` — a decision's verdict bits, a cost
        half's mana paid, a trigger's fired bit — and the created-objects head
        reads ``[GLOBAL]`` on effect halves and in-place rewrites (FR-060a).
        In training only, the value and script-API heads read the batch's
        encoded ``e`` rows, one per unique text, and the MLM head a masked
        second pass over the same texts (FR-058, FR-060b); ``heads`` names the
        ones whose weight is not zero, and no other is prepared. The MLM mask
        is the step's last draw from the batcher's generator.
        """
        targets = HeadTargets()
        verdicts = [verdict_targets(record) for record in records]
        if any(v is not None for v in verdicts):
            targets.verdict = (
                torch.tensor(
                    [v[0] if v else [0.0] * VERDICT_WIDTH for v in verdicts],
                ),
                torch.tensor(
                    [v[1] if v else [False] * VERDICT_WIDTH for v in verdicts],
                ),
            )
        created = [supervises_created_objects(record) for record in records]
        if any(created):
            targets.created = (
                torch.tensor([
                    created_objects_targets(record) if wanted
                    else [0.0] * CREATED_OBJECTS_WIDTH
                    for record, wanted in zip(records, created)
                ]),
                torch.tensor(created),
            )
        # No text, no matrix: the device half then has nothing to score
        # these heads on.
        if not training or not batch.rows:
            return targets
        texts = sorted(batch.rows, key=batch.rows.__getitem__)
        if "value" in heads:
            rows = [value_targets(text) for text in texts]
            targets.value = (
                torch.tensor([row.values for row in rows]),
                torch.tensor([row.mask for row in rows]),
            )
        if "api" in heads:
            targets.api = self._api_batch(texts, batch.texts)
        if "mlm" in heads:
            targets.mlm = batcher.draw_mlm_mask(
                batch.prepared, self.config.mlm_mask_prob,
            )
        return targets

    def _api_batch(
        self, texts: Sequence[str], lines: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The script-API head's targets for the batch's texts, in row order.

        The line's ``script_api_type`` (``-1`` where it has none or one outside
        the vocabulary) and the multi-hot of its parameter keys over every
        segment of its chained text, cached per text. The cache is touched
        only by whichever thread prepares training steps.
        """
        type_index = {name: i for i, name in enumerate(self.api_types)}
        key_index = {name: i for i, name in enumerate(self.param_keys)}
        types: list[int] = []
        keys = torch.zeros(len(texts), max(len(self.param_keys), 1))
        for row, text in enumerate(texts):
            cached = self._api_targets.get(text)
            if cached is None:
                line = lines.get(text)
                api = getattr(line, "script_api_type", None)
                found = set(getattr(line, "script_param_keys", ()) or ())
                for segment in segments(getattr(line, "script_text", None) or ""):
                    found.update(params(segment))
                cached = (
                    type_index.get(api, -1),
                    tuple(sorted(key_index[k] for k in found if k in key_index)),
                )
                self._api_targets[text] = cached
            types.append(cached[0])
            for column in cached[1]:
                keys[row, column] = 1.0
        return torch.tensor(types, dtype=torch.long), keys

    # ── one shard ───────────────────────────────────────────────────────

    def _texts(
        self, records: Sequence[EffectRecord], sidecars: SidecarCache,
    ) -> dict[str, str | None]:
        """``record_id -> acting text`` on this run's surface, once per shard.

        The weights, the rarity coverage and the epoch-line labels all key on
        it, and resolving it is a sidecar join per record.
        """
        return {
            record.record_id: ability_text_of(record, sidecars, self.surface)
            for record in records
        }

    def _weighted(
        self, records: list, sidecars: SidecarCache,
        texts: Mapping[str, str | None] | None = None,
    ) -> list[float]:
        """Rarity weights for a shard's records, from the manifest's table (FR-086)."""
        if texts is None:
            texts = self._texts(records, sidecars)
        return sample_weights(
            records, lambda record: texts[record.record_id],
            rarity=self.rarity, class_of=sampling_class,
        )

    def _shard_labels(
        self, records: list, sidecars: SidecarCache,
        texts: Mapping[str, str | None] | None = None,
    ) -> dict[str, tuple[str, str]]:
        """``record_id -> (rarity bucket, rule family)`` for one shard's records.

        Computed once per shard, so each step's count is a dict lookup per
        planned record. The bucket reads the corpus-wide rarity table where it
        names the text and the shard's own game count where it does not, the
        same fallback the weights use.
        """
        if texts is None:
            texts = self._texts(records, sidecars)
        own_games = effective_games(records, lambda r: texts[r.record_id])
        resolver = self._keyword_resolver(sidecars)
        labels: dict[str, tuple[str, str]] = {}
        for record in records:
            text = texts[record.record_id]
            if text is None:
                bucket = NO_TEXT_BUCKET
            else:
                games = (self.rarity or {}).get(text, own_games.get(text, 1))
                bucket = rarity_bucket(games)
            labels[record.record_id] = (
                bucket, rule_family(record, sidecars, resolver=resolver),
            )
        return labels

    def _keyword_resolver(self, sidecars) -> KeywordResolver:
        """One resolver per preparer, so its per-key memo outlives a shard."""
        if self._resolver is None or self._resolver_for is not sidecars:
            self._resolver = KeywordResolver(sidecars)
            self._resolver_for = sidecars
        return self._resolver

    def shard_steps(
        self, task: ShardTask, records: list, tokenizer, sidecars, widths,
        *, heads: frozenset[str], pin: bool = False,
    ) -> Iterator[ShardOpened | StepReady]:
        """One shard's messages: a ``ShardOpened``, then ``task.budget`` steps.

        The split first. A game naming a held-out card is never trained on;
        under an inherited split (every ``--corpus`` run) the validation games
        are the manifest's, and otherwise the shard's own tainted games, which
        is the whole of the accumulated set as far as this shard is concerned
        because a game never spans two shards.

        Then the weighted shuffle without replacement, from ``plan_rng``, and
        each step's host half from its own ``step_rng``: exactly the budget,
        and the shard ends with no trainable record in it.
        """
        tainted, _clean = shard_games(records, self.held_out)
        excluded = (
            self.inherited.validation_games if self.inherited is not None
            else tainted
        )
        training = [r for r in records if r.game_id not in excluded]
        texts = self._texts(training, sidecars)
        yield ShardOpened(
            trainable=len(training), held_back=len(records) - len(training),
            tainted=tainted,
            rarity_coverage=(
                rarity_coverage(texts.values(), self.rarity)
                if self.rarity is not None else None
            ),
        )
        if not training:
            return
        weights = self._weighted(training, sidecars, texts)
        labels = self._shard_labels(training, sidecars, texts)
        del texts
        batches = batches_without_replacement(
            training, weights, batch_size=self.config.batch_size,
            rng=plan_rng(self.seed, task.epoch, task.position),
        )
        for index in range(task.budget):
            plan = next(batches)
            buckets: Counter[str] = Counter()
            families: Counter[str] = Counter()
            for record in plan.records:
                bucket, family = labels[record.record_id]
                buckets[bucket] += 1
                families[family] += 1
            # `fields` is the epoch's, so no step number is needed to derive
            # them.
            prepared = self._prepare_step(
                plan, tokenizer, sidecars, widths, None, fields=task.fields,
                training=True, heads=heads, pin=pin, slim=True,
                rng=step_rng(self.seed, task.epoch, task.position, index),
            )
            yield StepReady(prepared, dict(buckets), dict(families))


@dataclass(frozen=True)
class PreparerSettings:
    """What a worker process needs to rebuild the host half: picklable, once.

    Sent once per worker at start rather than with every shard, because the
    rarity table and the inherited split are corpus-wide and each is several
    megabytes for a real corpus.
    """

    config: TrainEffectModelConfig
    seed: int
    held_out: HeldOutCards
    inherited: CorpusSplit | None
    rarity: Mapping[str, int] | None
    api_types: tuple[str, ...]
    param_keys: tuple[str, ...]
    widths: dict[SlotKind, int]
    heads: frozenset[str]


class StepPreparer(StepPreparation):
    """The host half alone, as a worker process runs it.

    Builds its own tokenizer, sidecar cache and keyword resolver from the
    settings, so nothing heavier than the settings crosses into the process.
    Truncations are collected rather than logged, because a worker's log
    reaches no handler; the worker hands them back with each shard.
    """

    def __init__(self, settings: PreparerSettings) -> None:
        self.config = settings.config
        self.seed = settings.seed
        self.surface = surface_of(settings.config.vocab_path)
        from effects.application.train_effect_model import variant_masks

        self.masks = variant_masks(settings.config.variant)
        self.held_out = settings.held_out
        self.inherited = settings.inherited
        self.rarity = settings.rarity
        self.api_types = list(settings.api_types)
        self.param_keys = list(settings.param_keys)
        self.widths = settings.widths
        self.heads = settings.heads
        self.device = torch.device("cpu")
        self.noise = None
        self.identity_table = None
        self.truncated: list[tuple[str, tuple]] = []
        self.truncations = TruncationLog(
            lambda text, keys: self.truncated.append((text, keys)),
        )
        self._init_preparation()
        self.tokenizer = self._build_tokenizer()
        self.sidecars = self._build_sidecars()

    def steps(self, task: ShardTask, records: list) -> Iterator:
        """``shard_steps`` with this worker's resources, then a ``ShardClosed``."""
        before = dict(self.sidecars.unresolved)
        yield from self.shard_steps(
            task, records, self.tokenizer, self.sidecars, self.widths,
            heads=self.heads,
        )
        unresolved = {
            name: count - before.get(name, 0)
            for name, count in self.sidecars.unresolved.items()
            if count != before.get(name, 0)
        }
        truncated, self.truncated = tuple(self.truncated), []
        yield ShardClosed(unresolved=unresolved, truncated=truncated)
