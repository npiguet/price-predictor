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
from typing import NamedTuple

import torch

from effects.application.ability_text import KeyResolver
from effects.application.mlm_masking import MaskedTokens
from effects.application.surface_batching import (
    PreparedBatch,
    SurfaceBatcher,
    SurfaceSpec,
)
from effects.application.text_staging import TextStager
from effects.application.train_effect_model import (
    TrainEffectModelConfig,
    ValidationGameInTrainingError,
)
from effects.domain.ability_encoder import TruncationLog
from effects.domain.aux_head_targets import (
    created_objects_targets,
    supervises_created_objects,
    verdict_targets,
)
from effects.domain.aux_heads import TRAINING_HEADS, Head
from effects.domain.damage_step_keywords import KeywordResolver
from effects.domain.effect_head_input import SlotKind
from effects.domain.effect_head_layout import CREATED_OBJECTS_WIDTH, VERDICT_WIDTH, FieldSpec
from effects.domain.effect_targets import derive_targets
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.lethality_counterfactuals import counterfactual, qualify
from effects.domain.provenance import SidecarLine
from effects.domain.rarity import RarityCoverage, TextGames
from effects.domain.records import EffectRecord
from effects.domain.rule_families import rule_family
from effects.domain.sampling import (
    batches_without_replacement,
    sample_weights,
    sampling_class,
)
from effects.domain.script_text import params, segments
from effects.domain.split import (
    CorpusSplit,
)
from effects.domain.surface_collation import entity_target_tensors
from effects.domain.value_targets import value_targets
from effects.infrastructure.tokenizer_loading import load_tokenizer
from effects.infrastructure.torch_device import pinned

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
    targets: EntityTargetBatch
    heads: HeadTargets

    def pin(self) -> None:
        self.batch.pin()
        self.targets = self.targets.pinned()
        self.heads.pin()


class RecordLabels(NamedTuple):
    """Where one trained record is counted on the epoch line (FR-055)."""

    bucket: str
    family: str


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


def _script_of(line: SidecarLine | None) -> str | None:
    return line.script_text if line is not None else None


def value_batch(
    texts: Sequence[str], lines: Mapping[str, SidecarLine],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(values, mask)`` of the value head's targets, one row per text (FR-058).

    Read from the sidecar line's script whatever the surface, since prose
    holds no ``$`` parameter; a text with no line reads itself.
    """
    rows = [value_targets(_script_of(lines.get(text)) or text) for text in texts]
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
    legality_mode_counts: Mapping[str, int] = field(default_factory=dict)


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
        self.surface = settings.config.tokenizer_files.surface
        self.surface_spec = SurfaceSpec.for_training(settings.config, settings.widths)
        self.split = settings.split
        self.rarity = settings.rarity
        self.legality_mode_counts = settings.legality_mode_counts
        self.api_types = list(settings.api_types)
        self.param_keys = list(settings.param_keys)
        self._api_type_index = {name: i for i, name in enumerate(self.api_types)}
        self._api_key_index = {name: i for i, name in enumerate(self.param_keys)}
        self.heads = settings.heads
        self.truncated: list[tuple[str, tuple]] = []
        self.truncations = truncations or TruncationLog(
            lambda text, keys: self.truncated.append((text, keys)),
        )
        self._api_targets: dict[str, tuple[int, tuple[int, ...]]] = {}
        self.tokenizer = load_tokenizer(self.config.tokenizer_files)
        self.sidecars = self.config.source_trees.sidecar_cache()
        #: One memoised join for the run, not per step.
        self.resolver = KeyResolver(self.sidecars, self.surface)
        self._keywords = KeywordResolver(self.sidecars)

    def training_batcher(self, rng: random.Random) -> SurfaceBatcher:
        """One training step's records-to-inputs pipeline, drawing keyword
        expansion and context dropout from the step's ``rng``."""
        return SurfaceBatcher(
            stager=TextStager(
                tokenizer=self.tokenizer,
                surface=self.surface,
                withhold_keyword=self.config.withhold_keyword,
                keyword_expand_p=self.config.keyword_expand_p,
                truncations=self.truncations,
            ),
            resolver=self.resolver,
            spec=self.surface_spec,
            context_dropout=self.config.context_dropout,
            rng=rng,
        )

    def scoring_batcher(self) -> SurfaceBatcher:
        """Validation's pipeline: no expansion beyond inference's and no
        context dropout, drawing from a stream of its own rather than any
        training step's, so how many batches it scores moves no step."""
        return SurfaceBatcher(
            stager=TextStager.for_inference(
                self.tokenizer, self.surface,
                withhold_keyword=self.config.withhold_keyword,
                truncations=self.truncations,
            ),
            resolver=self.resolver,
            spec=self.surface_spec,
            rng=random.Random(f"{self.seed}:validation"),
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

    def prepare_scoring_step(
        self, records: Sequence[EffectRecord], fields: tuple[FieldSpec, ...],
    ) -> DeviceStep:
        """One validation batch's host half: no augmentation, no copies, and
        only the shipped heads' targets."""
        return self._build(
            records, [derive_targets(record) for record in records], fields,
            self.scoring_batcher(), heads=frozenset(),
        )

    def _build(
        self, records, targets, fields: tuple[FieldSpec, ...],
        batcher: SurfaceBatcher, *, heads: frozenset[Head],
    ) -> DeviceStep:
        """The surfaces, the per-entity targets and ``heads``' targets."""
        batch = batcher.prepare(records)
        return DeviceStep(
            batch=replace(
                batch, texts={}, staged=replace(batch.staged, prepared=[]), surfaces=[],
            ),
            fields=fields,
            targets=entity_target_tensors(batch.surfaces, targets, fields),
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
                edit = copy.edit
                if edit.key is not None:
                    batcher.override_text(
                        edit.key, edit.text, key,
                        script=edit.script, role_spans=edit.role_spans,
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
        if not heads or not batch.staged.rows:
            return targets
        texts = batch.staged.ordered
        if Head.VALUE in heads:
            targets[Head.VALUE] = value_batch(texts, batch.texts)
        if Head.API in heads:
            targets[Head.API] = self._api_batch(texts, batch.texts)
        if Head.MLM in heads:
            targets[Head.MLM] = batcher.draw_mlm_mask(
                batch.staged.encoder_inputs, self.config.mlm_mask_prob,
            )
        return targets

    def _api_batch(
        self, texts: Sequence[str], lines: Mapping[str, SidecarLine],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The script-API head's ``(types, keys)`` for the batch's texts, in
        row order: the API type (``-1`` when none or unknown) and the
        multi-hot of parameter keys over every chained segment."""
        types: list[int] = []
        rows: list[int] = []
        columns: list[int] = []
        for row, text in enumerate(texts):
            cached = self._api_targets.get(text)
            if cached is None:
                line = lines.get(text)
                api = line.script_api_type if line is not None else None
                found = set(line.script_param_keys if line is not None else ())
                for segment in segments(_script_of(line) or ""):
                    found.update(params(segment))
                cached = (
                    self._api_type_index.get(api, -1),
                    tuple(sorted(
                        self._api_key_index[k] for k in found if k in self._api_key_index
                    )),
                )
                self._api_targets[text] = cached
            types.append(cached[0])
            rows += [row] * len(cached[1])
            columns += cached[1]
        # One indexed write: torch dispatches every element assignment on the
        # host half of the step.
        keys = torch.zeros(len(texts), max(len(self.param_keys), 1))
        keys[torch.tensor(rows, dtype=torch.long), torch.tensor(columns, dtype=torch.long)] = 1.0
        return torch.tensor(types, dtype=torch.long), keys

    def _texts(self, records: Sequence[EffectRecord]) -> dict[str, str | None]:
        """``record_id -> acting text`` on this run's surface, once per shard."""
        return {
            record.record_id: self.resolver.acting_text(record) for record in records
        }

    def _games(self, records: list, texts: Mapping[str, str | None]) -> TextGames:
        """Each text's games for a shard's records: the manifest's corpus-wide
        table (FR-086), else the shard's own count."""
        return TextGames.of(records, lambda record: texts[record.record_id], self.rarity)

    def _weighted(
        self, records: list, texts: Mapping[str, str | None], games: TextGames,
    ) -> list[float]:
        """Rarity weights for a shard's records."""
        return sample_weights(
            records, lambda record: texts[record.record_id],
            games=games, class_of=sampling_class,
        )

    def _shard_labels(
        self, records: list, texts: Mapping[str, str | None], games: TextGames,
    ) -> dict[str, RecordLabels]:
        """``record_id -> labels`` for one shard's records, bucketed by the
        games the weights read."""
        labels: dict[str, RecordLabels] = {}
        for record in records:
            text = texts[record.record_id]
            bucket = NO_TEXT_BUCKET if text is None else games.bucket(text)
            labels[record.record_id] = RecordLabels(
                bucket, rule_family(
                    record, self.sidecars, resolver=self._keywords,
                    legality_mode_counts=self.legality_mode_counts,
                ),
            )
        return labels

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
        weights = self._weighted(records, texts, games)
        labels = self._shard_labels(records, texts, games)
        del texts
        batches = batches_without_replacement(
            records, weights, batch_size=self.config.batch_size,
            rng=plan_rng(self.seed, task.epoch, task.position),
        )
        for index in range(task.budget):
            plan = next(batches)
            buckets: Counter[str] = Counter()
            families: Counter[str] = Counter()
            for record in plan.records:
                counted = labels[record.record_id]
                buckets[counted.bucket] += 1
                families[counted.family] += 1
            step, copies = self.prepare_training_step(
                plan.records, fields=task.fields, pin=pin,
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
        before = dict(self.sidecars.unresolved)
        yield from self.shard_steps(task, records)
        unresolved = {
            name: count - before.get(name, 0)
            for name, count in self.sidecars.unresolved.items()
            if count != before.get(name, 0)
        }
        truncated, self.truncated = tuple(self.truncated), []
        yield ShardClosed(unresolved=unresolved, truncated=truncated)
