"""The host half of a training step, in the training process or a worker.

Nothing here reads the weights. Every draw comes from a generator seeded by
``(seed, epoch, shard position, step)`` alone, so a step's inputs do not depend
on which process prepared it, or how many there were.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from effects.application.ability_text import KeyResolver
from effects.application.surface_batching import (
    Materializer,
    PreparedBatch,
    SurfaceBatcher,
)
from effects.application.text_staging import MaskedTokens, TextStager, pinned
from effects.application.train_effect_model import (
    TrainEffectModelConfig,
    rarity_coverage,
)
from effects.domain.ability_encoder import TruncationLog, surface_of
from effects.domain.ability_tokenizer import INFERENCE_KEYWORD_EXPAND_P
from effects.domain.aux_heads import TRAINING_HEADS, Head
from effects.domain.baselines import baseline_masks
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_head_layout import CREATED_OBJECTS_WIDTH, VERDICT_WIDTH, FieldSpec
from effects.domain.effect_targets import (
    created_objects_targets,
    derive_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.entity_field_targets import EntityFieldTargets
from effects.domain.lethality_counterfactuals import counterfactual, qualify
from effects.domain.rarity import rarity_bucket
from effects.domain.records import EffectRecord
from effects.domain.rule_families import rule_family
from effects.domain.sampling import (
    batches_without_replacement,
    effective_games,
    sample_weights,
    sampling_class,
)
from effects.domain.split import (
    CorpusSplit,
)
from effects.domain.surface_collation import entity_target_tensors
from effects.domain.value_targets import params, segments, value_targets
from effects.infrastructure.sidecar_io import SidecarCache, sidecar_roots
from effects.infrastructure.tokenizer_loading import load_tokenizer

#: Rarity-bucket label for a record with no acting text (combat, legality).
NO_TEXT_BUCKET = "no-text"
#: The epoch line's family label for the step's counterfactual copies (FR-063e).
COUNTERFACTUAL_FAMILY = "(lethality copy)"


def plan_rng(seed: int, epoch: int, position: int) -> random.Random:
    """The generator one shard's weighted shuffles draw from."""
    # A string, because 3.14's Random rejects a tuple seed and a string seeds
    # the same in every process.
    return random.Random(f"{seed}:{epoch}:{position}:plan")


def step_rng(seed: int, epoch: int, position: int, index: int) -> random.Random:
    """The generator one step's host half draws from: counterfactual copies,
    keyword expansion, context dropout and the MLM mask, in that order."""
    return random.Random(f"{seed}:{epoch}:{position}:{index}")


class HeadTargets(dict[Head, object]):
    """The targets of every loss term beside the per-entity one, on the host,
    for only the heads computed for this batch."""

    def pin(self) -> None:
        for head, targets in self.items():
            if isinstance(targets, MaskedTokens):
                targets.pin()
            else:
                self[head] = pinned(targets)


@dataclass
class DeviceStep:
    """One step as its device half reads it: tensors, and the rows ``e`` fills.

    ``batch`` keeps only what ``Materializer.materialize`` reads; the host
    intermediates are dropped, as the bulk of what a worker would pickle.
    """

    batch: PreparedBatch
    fields: tuple[FieldSpec, ...]
    #: ``entity_target_tensors``' four outputs.
    gate: torch.Tensor
    field_targets: EntityFieldTargets
    mask: torch.Tensor
    index: torch.Tensor
    heads: HeadTargets

    def pin(self) -> None:
        self.batch.pin()
        self.field_targets = replace(
            self.field_targets, values=pinned(dict(self.field_targets.values)),
        )
        self.index = self.index.pin_memory()
        self.heads.pin()


@dataclass(frozen=True)
class ShardTask:
    """One shard's share of an epoch, small and picklable for a worker."""

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
    #: ``(found, distinct)``: how many of the trainable records' distinct
    #: ability texts the rarity table names; None without a table.
    rarity_coverage: tuple[int, int] | None


@dataclass
class StepReady:
    """One prepared step, with the epoch-line shares of the records it planned."""

    prepared: DeviceStep
    #: Trained records by rarity bucket and by rule family (FR-055).
    buckets: dict[str, int]
    families: dict[str, int]


@dataclass(frozen=True)
class ShardClosed:
    """A worker's last message of a shard: what its own caches tallied, for
    the training process to log."""

    #: ``script_file -> lookups`` naming an unconverted script, this shard only.
    unresolved: dict[str, int] = field(default_factory=dict)
    #: ``(text, keys)`` for each text first truncated during this shard.
    truncated: tuple[tuple[str, tuple], ...] = ()


def value_batch(
    texts: Sequence[str], lines: Mapping[str, object],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(values, mask)`` of the value head's targets, one row per text (FR-058).

    Read from the sidecar line's script whatever the surface, since prose
    holds no ``$`` parameter; a text with no line reads itself.
    """
    rows = [
        value_targets(getattr(lines.get(text), "script_text", None) or text)
        for text in texts
    ]
    return (
        torch.tensor([row.values for row in rows]),
        torch.tensor([row.mask for row in rows]),
    )


@dataclass(frozen=True)
class PreparerSettings:
    """What a preparer is built from: picklable, and sent once per worker
    rather than per shard because the rarity table and split are megabytes."""

    config: TrainEffectModelConfig
    seed: int
    split: CorpusSplit
    rarity: Mapping[str, int] | None
    api_types: tuple[str, ...]
    param_keys: tuple[str, ...]
    widths: dict[SlotKind, int]
    heads: frozenset[Head]
    legality_mode_counts: Mapping[str, int] | None = None


def training_heads(config: TrainEffectModelConfig, *, api_types) -> frozenset[Head]:
    """The training-only heads a step computes: those with a nonzero weight,
    and the script-API head only with an API vocabulary."""
    return frozenset(
        spec.head for spec in TRAINING_HEADS
        if spec.weight(config) > 0.0
        and (api_types or not spec.needs_api_vocabulary)
    )


class StepPreparer:
    """The host half of every training and scoring step, built from its settings.

    A worker passes no ``truncations``, since its log reaches no handler; they
    are collected and handed back with each shard (:class:`ShardClosed`).
    """

    def __init__(
        self, settings: PreparerSettings, *, truncations: TruncationLog | None = None,
    ) -> None:
        self.config = settings.config
        self.seed = settings.seed
        self.surface = surface_of(settings.config.vocab_path)
        self.masks = baseline_masks(settings.config.variant)
        self.split = settings.split
        self.rarity = settings.rarity
        self.legality_mode_counts = settings.legality_mode_counts
        self.api_types = list(settings.api_types)
        self.param_keys = list(settings.param_keys)
        self._api_type_index = {name: i for i, name in enumerate(self.api_types)}
        self._api_key_index = {name: i for i, name in enumerate(self.param_keys)}
        self.widths = settings.widths
        self.heads = settings.heads
        self.truncated: list[tuple[str, tuple]] = []
        self.truncations = truncations or TruncationLog(
            lambda text, keys: self.truncated.append((text, keys)),
        )
        self._api_targets: dict[str, tuple[int, tuple[int, ...]]] = {}
        self.tokenizer = load_tokenizer(
            self.config.vocab_path, self.config.keyword_definitions,
        )
        self.sidecars = SidecarCache(sidecar_roots(
            self.config.cards_folders, self.config.variant_scripts,
        ))
        #: One memoised join for the run, not per step.
        self.resolver = KeyResolver(self.sidecars, self.surface)
        self._keywords = KeywordResolver(self.sidecars)

    def batcher(
        self, *, training: bool = True, rng: random.Random | None = None,
        materializer: Materializer | None = None,
    ) -> SurfaceBatcher:
        """The records-to-inputs pipeline, built the same way for every caller.

        ``training=False`` turns keyword expansion and context dropout off;
        the withheld keyword stays withheld, being a split rather than an
        augmentation. Scoring draws from a stream of its own, never the
        training step's ``rng``. ``materializer`` is for callers that build
        batches whole.
        """
        return SurfaceBatcher(
            stager=TextStager(
                tokenizer=self.tokenizer,
                surface=self.surface,
                withhold_keyword=self.config.withhold_keyword,
                keyword_expand_p=(
                    self.config.keyword_expand_p if training
                    else INFERENCE_KEYWORD_EXPAND_P
                ),
                truncations=self.truncations,
            ),
            resolver=self.resolver,
            masks=self.masks,
            e_dim=self.config.e_dim,
            widths=self.widths,
            materializer=materializer,
            context_dropout=self.config.context_dropout if training else 0.0,
            rng=(
                (rng or random.Random(f"{self.seed}:unplanned")) if training
                else random.Random(f"{self.seed}:validation")
            ),
        )

    def prepare_training_step(
        self, plan, *, fields: tuple[FieldSpec, ...], rng: random.Random,
        pin: bool = False,
    ) -> tuple[DeviceStep, int]:
        """One training step's host half, and how many counterfactual copies
        it appended to the planned records (FR-063e).

        ``pin`` is for the training process only: page-locked memory does not
        survive a trip between processes.
        """
        batcher = self.batcher(training=True, rng=rng)
        records = plan.records
        targets = [derive_targets(record) for record in records]
        copies = []
        if self.config.lethality_copies > 0:
            copies = self._counterfactuals(records, targets, batcher)
            records = [*records, *(copy.record for copy in copies)]
            targets = [*targets, *(copy.targets for copy in copies)]
        step = self._build(records, targets, fields, batcher, heads=self.heads)
        if pin:
            step.pin()
        return step, len(copies)

    def prepare_scoring_step(self, plan, fields: tuple[FieldSpec, ...]) -> DeviceStep:
        """One validation batch's host half: no augmentation, no copies, and
        only the shipped heads' targets."""
        records = plan.records
        return self._build(
            records, [derive_targets(record) for record in records], fields,
            self.batcher(training=False), heads=frozenset(),
        )

    def _build(
        self, records, targets, fields: tuple[FieldSpec, ...],
        batcher: SurfaceBatcher, *, heads: frozenset[Head],
    ) -> DeviceStep:
        """The surfaces, the per-entity targets and ``heads``' targets."""
        batch = batcher.prepare(records)
        gate, field_targets, mask, index = entity_target_tensors(
            batch.surfaces, targets, fields,
        )
        return DeviceStep(
            batch=replace(batch, texts={}, prepared=[], surfaces=[]),
            fields=fields, gate=gate, field_targets=field_targets, mask=mask,
            index=index,
            heads=self._head_targets(records, batcher, batch, heads=heads),
        )

    def _counterfactuals(self, records, targets, batcher) -> list:
        """``--lethality-copies`` copies of each step record that qualifies
        (FR-063e), drawn before the build's own draws so a run without copies
        draws exactly as before."""
        copies = []
        for record, record_targets in zip(records, targets):
            if not record.ability:
                continue
            key = record.ability[0]
            line = batcher.line_of(key)
            qualified = qualify(record, line, record_targets, self._keywords.keywords_of)
            if qualified is None:
                continue
            for number in range(self.config.lethality_copies):
                copy = counterfactual(
                    record, qualified, record_targets, line, batcher.text_of(key),
                    batcher.rng, number,
                )
                if copy.key is not None:
                    batcher.stand_in(
                        copy.key, copy.text, key,
                        script=copy.script, role_spans=copy.role_spans,
                    )
                copies.append(copy)
        return copies

    def _head_targets(
        self, records, batcher, batch: PreparedBatch, *, heads: frozenset[Head],
    ) -> HeadTargets:
        """The shipped heads' targets (FR-060a), then those of the
        training-only ``heads`` (FR-058, FR-060b). The MLM mask is the step's
        last draw from the batcher's generator."""
        targets = HeadTargets()
        verdicts = [verdict_targets(record) for record in records]
        if any(v is not None for v in verdicts):
            targets[Head.VERDICT] = (
                torch.tensor(
                    [v[0] if v else [0.0] * VERDICT_WIDTH for v in verdicts],
                ),
                torch.tensor(
                    [v[1] if v else [False] * VERDICT_WIDTH for v in verdicts],
                ),
            )
        created = [supervises_created_objects(record) for record in records]
        if any(created):
            targets[Head.CREATED_OBJECTS] = (
                torch.tensor([
                    created_objects_targets(record) if wanted
                    else [0.0] * CREATED_OBJECTS_WIDTH
                    for record, wanted in zip(records, created)
                ]),
                torch.tensor(created),
            )
        # No text, no matrix for these heads to read.
        if not heads or not batch.rows:
            return targets
        texts = sorted(batch.rows, key=batch.rows.__getitem__)
        if Head.VALUE in heads:
            targets[Head.VALUE] = value_batch(texts, batch.texts)
        if Head.API in heads:
            targets[Head.API] = self._api_batch(texts, batch.texts)
        if Head.MLM in heads:
            targets[Head.MLM] = batcher.draw_mlm_mask(
                batch.prepared, self.config.mlm_mask_prob,
            )
        return targets

    def _api_batch(
        self, texts: Sequence[str], lines: Mapping[str, object],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The script-API head's ``(types, keys)`` for the batch's texts, in
        row order: the API type (``-1`` when none or unknown) and the
        multi-hot of parameter keys over every chained segment."""
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
                    self._api_type_index.get(api, -1),
                    tuple(sorted(
                        self._api_key_index[k] for k in found if k in self._api_key_index
                    )),
                )
                self._api_targets[text] = cached
            types.append(cached[0])
            for column in cached[1]:
                keys[row, column] = 1.0
        return torch.tensor(types, dtype=torch.long), keys

    def _texts(self, records: Sequence[EffectRecord]) -> dict[str, str | None]:
        """``record_id -> acting text`` on this run's surface, once per shard."""
        return {
            record.record_id: self.resolver.acting_text(record) for record in records
        }

    def _weighted(
        self, records: list, texts: Mapping[str, str | None],
    ) -> list[float]:
        """Rarity weights for a shard's records, from the manifest's table (FR-086)."""
        return sample_weights(
            records, lambda record: texts[record.record_id],
            rarity=self.rarity, class_of=sampling_class,
        )

    def _shard_labels(
        self, records: list, texts: Mapping[str, str | None],
    ) -> dict[str, tuple[str, str]]:
        """``record_id -> (rarity bucket, rule family)`` for one shard's
        records; the bucket falls back to the shard's own game count as the
        weights do."""
        own_games = effective_games(records, lambda r: texts[r.record_id])
        labels: dict[str, tuple[str, str]] = {}
        for record in records:
            text = texts[record.record_id]
            if text is None:
                bucket = NO_TEXT_BUCKET
            else:
                games = (self.rarity or {}).get(text, own_games.get(text, 1))
                bucket = rarity_bucket(games)
            labels[record.record_id] = (
                bucket, rule_family(
                    record, self.sidecars, resolver=self._keywords,
                    legality_mode_counts=self.legality_mode_counts,
                ),
            )
        return labels

    def shard_steps(
        self, task: ShardTask, records: list, *, pin: bool = False,
    ) -> Iterator[ShardOpened | StepReady]:
        """One shard's messages: a ``ShardOpened``, then ``task.budget`` steps,
        none from a game the manifest places in a validation stratum."""
        excluded = self.split.validation_games
        training = [r for r in records if r.game_id not in excluded]
        texts = self._texts(training)
        yield ShardOpened(
            trainable=len(training), held_back=len(records) - len(training),
            rarity_coverage=(
                rarity_coverage(texts.values(), self.rarity)
                if self.rarity is not None else None
            ),
        )
        if not training:
            return
        weights = self._weighted(training, texts)
        labels = self._shard_labels(training, texts)
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
            step, copies = self.prepare_training_step(
                plan, fields=task.fields, pin=pin,
                rng=step_rng(self.seed, task.epoch, task.position, index),
            )
            if copies:
                families[COUNTERFACTUAL_FAMILY] += copies
            yield StepReady(step, dict(buckets), dict(families))

    def steps(self, task: ShardTask, records: list) -> Iterator:
        """``shard_steps``, then a ``ShardClosed``: a worker's whole shard."""
        before = dict(self.sidecars.unresolved)
        yield from self.shard_steps(task, records)
        unresolved = {
            name: count - before.get(name, 0)
            for name, count in self.sidecars.unresolved.items()
            if count != before.get(name, 0)
        }
        truncated, self.truncated = tuple(self.truncated), []
        yield ShardClosed(unresolved=unresolved, truncated=truncated)
