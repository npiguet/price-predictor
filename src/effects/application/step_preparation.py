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

from effects.application.ability_text import KeyResolver
from effects.application.head_target_builders import ApiTargetBuilder, HeadTargetBuilder
from effects.application.lethality_augmenter import LethalityAugmenter
from effects.application.line_preparation import TruncationLog
from effects.application.record_head_targets import HeadTargets, shipped_head_targets
from effects.application.shard_plan import ShardPlan
from effects.application.surface_batching import (
    PreparedBatch,
    StepDraws,
    SurfaceBatcher,
    SurfaceSpec,
)
from effects.application.text_resolution import TextResolution
from effects.application.text_staging import TextStager
from effects.application.train_effect_model import (
    TrainEffectModelConfig,
    ValidationGameInTrainingError,
)
from effects.domain.aux_heads import TRAINING_HEADS, Head
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_families import effect_family
from effects.domain.effect_head_layout import (
    FieldSpec,
)
from effects.domain.effect_targets import derive_targets
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.lethality_counterfactuals import Counterfactual
from effects.domain.rarity import RarityCoverage, TextGames
from effects.domain.records import EffectRecord
from effects.domain.rule_mask import (
    FULL_WEIGHT,
    CopyWeights,
    copy_weights,
    weighed_by_class,
)
from effects.domain.split import (
    CorpusSplit,
)
from effects.domain.surface_collation import entity_target_tensors
from effects.infrastructure.tokenizer_loading import load_tokenizer

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


@dataclass(frozen=True)
class StepRows:
    """The records of one step, with the per-entity targets and rule-mask
    weights derived for each."""

    records: list
    targets: list
    weights: list[CopyWeights]

    def with_copies(self, copies: Sequence[Counterfactual]) -> StepRows:
        """The rows with the lethality ``copies`` appended after them."""
        if not copies:
            return self
        return StepRows(
            [*self.records, *(copy.record for copy in copies)],
            [*self.targets, *(copy.targets for copy in copies)],
            # A lethality copy trains every output at full weight (FR-061),
            # and is made for a record's first natural row alone.
            [*self.weights, *([FULL_WEIGHT] * len(copies))],
        )


@dataclass
class DeviceStep:
    """One step as its device half reads it: tensors, and the rows ``e`` fills.

    ``batch`` keeps only what ``Materializer.materialize`` reads; the host
    intermediates are dropped, as the bulk of what a worker would pickle.
    """

    batch: PreparedBatch
    fields: tuple[FieldSpec, ...]
    targets: EntityTargetBatch
    heads: HeadTargets

    def pin(self) -> None:
        self.batch.pin()
        self.targets = self.targets.pinned()
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
    #: How many of the trainable records' distinct ability texts the rarity
    #: table names; None without a table.
    rarity_coverage: RarityCoverage | None


@dataclass
class StepReady:
    """One prepared step, with the epoch-line shares of the records it planned."""

    prepared: DeviceStep
    #: Trained records by rarity bucket and by effect family (FR-055).
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
    heads: frozenset[Head]
    legality_mode_counts: Mapping[str, int] = field(default_factory=dict)
    #: The manifest's ``rule -> outputs`` (FR-060a), which the rule mask
    #: reads (FR-061).
    rule_outputs: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


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
        self.surface_spec = SurfaceSpec.for_training(settings.config)
        self.split = settings.split
        self.rarity = settings.rarity
        self.legality_mode_counts = settings.legality_mode_counts
        self.api_types = list(settings.api_types)
        self.param_keys = list(settings.param_keys)
        self.heads = settings.heads
        self.rule_outputs = settings.rule_outputs
        self.truncated: list[tuple[str, tuple]] = []
        self.truncations = truncations or TruncationLog(
            lambda text, keys: self.truncated.append((text, keys)),
        )
        self.tokenizer = load_tokenizer(self.config.tokenizer_files)
        self.surface = self.tokenizer.surface
        self.sidecars = self.config.source_trees.sidecar_cache()
        #: One memoised join for the run, not per step.
        self.resolver = KeyResolver(self.sidecars, self.surface)
        self._keywords = KeywordResolver(self.sidecars)
        self.head_builder = HeadTargetBuilder(
            self.heads, self.config.mlm_mask_prob,
            ApiTargetBuilder(self.api_types, self.param_keys),
        )
        self.lethality = LethalityAugmenter(
            self.config.lethality_copies, self._keywords.keywords_of, self.sidecars,
        )

    def training_batcher(self, rng: random.Random) -> SurfaceBatcher:
        """One training step's records-to-inputs pipeline, drawing keyword
        expansion and context dropout from the step's ``rng``."""
        stager = TextStager(
            tokenizer=self.tokenizer,
            surface=self.surface,
            withhold_keyword=self.config.withhold_keyword,
            keyword_expand_p=self.config.keyword_expand_p,
            truncations=self.truncations,
        )
        return SurfaceBatcher(
            text=TextResolution(self.resolver, stager),
            spec=self.surface_spec,
            draws=StepDraws(context_dropout=self.config.context_dropout, rng=rng),
        )

    def scoring_batcher(self) -> SurfaceBatcher:
        """Validation's pipeline: no expansion beyond inference's and no
        context dropout, drawing from a stream of its own rather than any
        training step's, so how many batches it scores moves no step."""
        stager = TextStager.for_inference(
            self.tokenizer, self.surface,
            withhold_keyword=self.config.withhold_keyword,
            truncations=self.truncations,
        )
        return SurfaceBatcher(
            text=TextResolution(self.resolver, stager),
            spec=self.surface_spec,
            draws=StepDraws(rng=random.Random(f"{self.seed}:validation")),
        )

    def prepare_training_step(
        self, records: Sequence[EffectRecord], *, fields: tuple[FieldSpec, ...],
        rng: random.Random, pin: bool = False,
    ) -> tuple[DeviceStep, int]:
        """One training step's host half over its planned ``records``, and how
        many counterfactual copies it appended to them (FR-063e).

        ``pin`` is for the training process only: page-locked memory does not
        survive a trip between processes.
        """
        batcher = self.training_batcher(rng)
        records = list(records)
        rows = StepRows(records, [derive_targets(record) for record in records],
                        self.rule_mask(records))
        copies = []
        if self.config.lethality_copies > 0:
            copies = self.lethality.copies_of(rows.records, rows.targets, batcher)
        step = self._build(rows.with_copies(copies), fields, batcher)
        if pin:
            step.pin()
        return step, len(copies)

    def prepare_scoring_step(
        self, records: Sequence[EffectRecord], fields: tuple[FieldSpec, ...], *,
        class_weighted: bool = False,
    ) -> DeviceStep:
        """One validation batch's host half: no augmentation, no copies, and
        only the shipped heads' targets, under the run's rule mask as training
        reads its own records (FR-068a), so the loss denominators and the
        explained deviance weigh each output as training does.
        ``class_weighted`` scales each row by its ``class_weight`` (FR-059a),
        for the explained-deviance sample alone."""
        records = list(records)
        weights = self.rule_mask(records)
        if class_weighted:
            weights = [weighed_by_class(copy, record) for copy, record in zip(weights, records)]
        return scoring_step(records, weights, fields, self.scoring_batcher())

    def rule_mask(self, records: Sequence[EffectRecord]) -> list[CopyWeights]:
        """Each record's weights under ``--rule-mask`` (FR-061)."""
        return [
            copy_weights(record, self.config.rule_mask, self.rule_outputs)
            for record in records
        ]

    def _build(
        self, rows: StepRows, fields: tuple[FieldSpec, ...], batcher: SurfaceBatcher,
    ) -> DeviceStep:
        """The surfaces, the per-entity targets and the heads' targets, each
        record's outputs weighed by ``rows.weights``."""
        batch = batcher.prepare(rows.records)
        return DeviceStep(
            batch=stripped(batch), fields=fields,
            targets=entity_target_tensors(batch.surfaces, rows.targets, fields, rows.weights),
            heads=self.head_builder.build(rows.records, rows.weights, batcher, batch),
        )

    def _texts(self, records: Sequence[EffectRecord]) -> dict[str, str | None]:
        """``record_id -> acting text`` on this run's surface, once per shard."""
        return {
            record.record_id: self.resolver.acting_text(record) for record in records
        }

    def _games(self, records: list, texts: Mapping[str, str | None]) -> TextGames:
        """Each text's games for a shard's records: the manifest's corpus-wide
        table (FR-086), else the shard's own count."""
        return TextGames.of(records, lambda record: texts[record.record_id], self.rarity)

    def _family_of(self, record: EffectRecord) -> str:
        """``record``'s effect family on the epoch line."""
        return effect_family(
            record, self.sidecars, resolver=self._keywords,
            legality_mode_counts=self.legality_mode_counts,
        )

    def _shard_plan(
        self, records: list, texts: Mapping[str, str | None], games: TextGames,
    ) -> ShardPlan:
        """One shard's rarity weights and epoch-line labels."""
        return ShardPlan.of(records, texts, games, self._family_of)

    def shard_steps(
        self, task: ShardTask, records: list, *, pin: bool = False,
    ) -> Iterator[ShardOpened | StepReady]:
        """One shard's messages: a ``ShardOpened``, then ``task.budget`` steps.

        Raises :class:`ValidationGameInTrainingError` for a shard holding a
        game the manifest places in a validation stratum, rather than training
        around it: the corpus and its manifest disagree.
        """
        self._refuse_validation_games(task, records)
        texts = self._texts(records)
        games = self._games(records, texts)
        yield ShardOpened(trainable=len(records), rarity_coverage=games.coverage())
        if not records:
            return
        plan = self._shard_plan(records, texts, games)
        del texts
        batches = plan.batches(
            self.config.batch_size, plan_rng(self.seed, task.epoch, task.position),
        )
        for index in range(task.budget):
            planned = next(batches)
            buckets, families = plan.shares(planned.records)
            step, copies = self.prepare_training_step(
                planned.records, fields=task.fields, pin=pin,
                rng=step_rng(self.seed, task.epoch, task.position, index),
            )
            if copies:
                families[COUNTERFACTUAL_FAMILY] += copies
            yield StepReady(step, dict(buckets), dict(families))

    def _refuse_validation_games(self, task: ShardTask, records: list) -> None:
        excluded = self.split.validation_games
        leaked = Counter(r.game_id for r in records if r.game_id in excluded)
        if not leaked:
            return
        game, count = leaked.most_common(1)[0]
        raise ValidationGameInTrainingError(
            f"training shard {task.shard} holds {sum(leaked.values())} record(s) "
            f"of {len(leaked)} game(s) its corpus's manifest places in a "
            f"validation stratum, e.g. {game} ({count} record(s)). build-corpus "
            "writes no validation game into training/, so this shard and the "
            "manifest come from different builds: rebuild the corpus."
        )

    def steps(self, task: ShardTask, records: list) -> Iterator:
        """``shard_steps``, then a ``ShardClosed``: a worker's whole shard."""
        before = self.sidecars.join_gaps().unresolved
        yield from self.shard_steps(task, records)
        unresolved = dict(self.sidecars.join_gaps().unresolved - before)
        truncated, self.truncated = tuple(self.truncated), []
        yield ShardClosed(unresolved=unresolved, truncated=truncated)


def stripped(batch: PreparedBatch) -> PreparedBatch:
    """A prepared batch without the host intermediates no device half reads."""
    return replace(
        batch, texts={}, staged=replace(batch.staged, prepared=[]), surfaces=[],
        value_lines=(),
    )


def scoring_step(
    records: Sequence[EffectRecord], weights: list[CopyWeights],
    fields: tuple[FieldSpec, ...], batcher: SurfaceBatcher,
) -> DeviceStep:
    """One scoring step's host half over ``records`` through ``batcher``: no
    augmentation, no copies, the shipped heads' targets alone, each record's
    outputs weighed by ``weights``. Validation's steps and a saved
    checkpoint's (``control_score.CheckpointPreparer``) are both built here."""
    records = list(records)
    batch = batcher.prepare(records)
    return DeviceStep(
        batch=stripped(batch), fields=fields,
        targets=entity_target_tensors(
            batch.surfaces, [derive_targets(record) for record in records], fields, weights),
        heads=shipped_head_targets(records, weights),
    )
