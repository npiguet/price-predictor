"""The control score: each epoch's selection number, from the frozen selection
set's selection halves (FR-068, FR-068c, FR-069; research E2, E6;
contracts/selection-set.md), and the same score of a saved checkpoint on
either half (FR-070).

Built once, at startup (:meth:`ControlScorer.prepare`): one half of each
validation stratum — the selection half in training — is read once, and the
host half of every item it holds is prepared and cached —

- the probe records, each read at its entity's slot (``[ACT]`` for a verdict
  probe) through :data:`~effects.domain.probe_readouts.READOUTS`;
- the grid edits, each grid record edited into every cell by a
  :class:`GridEditor` and read at the cell's entity, beside the rule's
  answer for that record there;
- the rule-test records, read for each of the rule's effective outputs (the
  outputs the manifest's map names that its tagged records carry, E6): at
  the tag's outcome entities (their mean; research R1), or at the record for
  a verdict, global or created-objects output;
- the deviance sample's copies, as scoring steps under the run's
  ``--rule-mask``, the way validation scores its own records, each copy
  weighed by its ``class_weight`` (FR-059a).

A record several items read is prepared once. The cache stays on the host,
packed (``host_cache``: each tensor in its smallest exact form), and is
unpacked a chunk at a time as the chunk is run. The cells of one grid record
fill chunks of their own, so a chunk's rows are near copies of each other and
are kept as their differences.

Each epoch (:meth:`ControlScorer.epoch`), on whatever weights the caller has
swapped in (the averaged ones), only the device forward runs, through
``record_forward``, and each stratum's four components are scored:

============  ===============================================  ===============
component     item score                                       rescaled
============  ===============================================  ===============
probes        rung-3 AUC (binary) or R² (amount), per type     2·AUC − 1; R²
grids         MAE of each record's reading against its answer  1 − MAE / B
rule tests    per rule, mean over its outputs of the AUC        2·AUC − 1
              ranking ``fires`` above ``does_not_fire``
deviance      per output term, 1 − loss / constant loss        as is
============  ===============================================  ===============

Every item is clipped below at −1, a component is the mean of its items
(the grids': the mean over the damage, combat and evasion grids of each
one's mean over its variants, ``grid_edits.grid_of``), a stratum score the
mean of its four components, and the control score
``W · card_disjoint + (1 − W) · game_disjoint`` with ``W = --stratum-weight``.
A component with nothing to score — no item kept, none readable, or a
deviance component every field of which falls at or below
``--deviance-floor`` — raises :class:`ControlScoreError` naming stratum,
half and component, which ``train-effect-model`` turns into exit 1 (FR-068c).

**A rule test's readout** (E6). For a binary or categorical output the
reading is the probability of the scenario's category, for a count or delta
its expected value under the negative binomial; per-entity readings are the
unconditional ones gate 2 and the probes read (``HeadReading``: the gate's
probability times the field's, a legality field ungated), except on a record
that trains no gate (a cost half, a playability record, a rewrite whose outcome
is partly unknown), where a field trains at the entities its targets mark
affected and is read conditionally (``control_reads._ConditionalRead``). The selection set
names a rule's records and their polarity, not the scenario's category, so
the category is read off the frozen records' own targets once, at startup:
the category (or, for a multi-member output, the member) whose observed share
differs most between the ``fires`` and the ``does_not_fire`` records, and the
direction of that difference, which turns the reading into its complement
where the scenario's outcome is an absence (a creature that does not die). A
count or delta takes the sign of the difference of the two polarities' mean
targets. An output no ``fires`` record observes, an output name no head
carries (``new:…``, ``planned:…``), an output at an entity or a record whose
targets do not define it, and an output at an entity the record gives no
slot are not read, the last two counted and logged once per stratum and
half; a rule with no output left is left out of its component.

**The deviance component** reads the items the frozen set kept under the
run's mask (each one's standard error at most the freeze's item target):
every shipped head's terms, the ones the loss denominators divide by — the
per-entity head's fields and gate (by their names), the verdict head's fields
at ``[ACT]``, the global head's at ``[GLOBAL]`` and the created-objects
head's slot columns and overflow flag (``<head>.<term>``,
``selection_set.deviance_item``). The model's loss on each is the deviance
steps' own, weighed by the run's ``--rule-mask`` as training weighs it and
by each copy's ``class_weight``, and its constant-prediction loss the
floor's over the same steps, the constant fitted at those weights
(``loss_floor.constant_predictor_losses``, ``weighted``); a term whose
constant loss is at or below ``--deviance-floor`` is left out. A copy of
class weight k reads as k repeated copies would.

Only the prepared half's games and the records its items name are taken
from a stratum, each distinct record once (a validation stratum holds a
record's row once plus once for every added copy, FR-068a), copies only for
the deviance sample.

In a training run the validator (``validation_pass.Validator.prepare``)
builds one :class:`ControlScorer` at startup from the run's
:class:`FrozenSelection`, over the selection half — training never reads the
reporting half — handing it every field set the run's epochs use and
``grid_edits.GridEdits`` as its editor, and scores it at each epoch's close
inside the averaged weights' swap; the training loop keeps the epoch of the
best :attr:`ControlScore.control` (FR-069), and the CLI turns a
:class:`ControlScoreError` into exit 1.

A saved checkpoint is scored by :func:`score_half`: ``evaluate-effect-model``
scores the reporting half, every component, and the knowledge probes score
each half's rule tests alone (``components``). Both read it lenient
(``strict=False``): a component with nothing to score is NaN and named,
never an error, since a report reports.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from effects.application.control_items import (
    SELECTION,
    ControlScoreError,
    component_score,
    explained_deviance,
    mixed,
    probe_item,
    rule_item,
)
from effects.application.control_reads import Heads
from effects.application.control_stratum import (
    GridEditor,
    GridItem,
    ScoringPreparer,
    StratumCache,
    prepare_stratum,
)
from effects.application.host_cache import host_bytes, unpack
from effects.application.record_forward import RecordForward
from effects.application.step_preparation import DeviceStep, scoring_step
from effects.application.stratum_records import StratumReader
from effects.application.surface_batching import SurfaceBatcher
from effects.application.validation_pass import FieldParts, KeptTargets, score_steps
from effects.domain.effect_head_layout import FieldSpec, active_fields
from effects.domain.loss_floor import constant_predictor_losses
from effects.domain.rule_mask import RuleMask, copy_weights, weighed_by_class
from effects.domain.sampling import SamplingClass
from effects.domain.score_scales import clipped, grid_score, non_finite
from effects.domain.selection_set import (
    COMPONENTS,
    DEVIANCE,
    GRIDS,
    PROBES,
    RULE_TESTS,
    SelectionSet,
    deviance_items,
)
from effects.domain.strata import Stratum

if TYPE_CHECKING:
    from effects.application.step_loss import StepLossComputer
    from effects.domain.records import EffectRecord

logger = logging.getLogger(__name__)

#: Records per prepared chunk of the probes', grids' and rule tests' forward,
#: and per deviance step: forward passes only, so a chunk this large fits the
#: card beside a training run's state, and runs several times faster than
#: chunks of a training batch's size (T160).
CHUNK_RECORDS = 128


class CheckpointPreparer:
    """A saved checkpoint's scoring steps: each batcher a fresh one over the
    checkpoint's own stager, join and surface layout (``batcher``, a loaded
    model's), drawing from a seeded stream of its own; each step's records
    weighed under the checkpoint's ``--rule-mask``, as its validation read its
    own (FR-068a)."""

    def __init__(self, batcher: SurfaceBatcher, *, mask: str,
                 rule_outputs: Mapping[str, Sequence[str]], seed: int = 42) -> None:
        self.template = batcher
        self.mask = RuleMask(mask)
        self.rule_outputs = {rule: tuple(outputs) for rule, outputs in rule_outputs.items()}
        self.seed = seed

    def scoring_batcher(self) -> SurfaceBatcher:
        template = self.template
        return SurfaceBatcher(stager=template.stager, resolver=template.resolver,
                              spec=template.spec, materializer=template.materializer,
                              rng=random.Random(f"{self.seed}:control"))

    def prepare_scoring_step(self, records: Sequence[EffectRecord],
                             fields: tuple[FieldSpec, ...], *,
                             class_weighted: bool = False) -> DeviceStep:
        records = list(records)
        weights = [copy_weights(record, self.mask, self.rule_outputs) for record in records]
        if class_weighted:
            weights = [weighed_by_class(copy, record) for copy, record in zip(weights, records)]
        return scoring_step(records, weights, fields, self.scoring_batcher())


# ── scores ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StratumScore:
    """One stratum's four components, each the mean of its clipped items."""

    components: Mapping[str, float]
    #: ``component -> item -> clipped rescaled score``, what each mean read.
    items: Mapping[str, Mapping[str, float]]
    #: ``component -> items read but left out this epoch, with why``.
    left_out: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: The model readings (and deviance terms) this epoch that were not
    #: finite; each was scored at the floor (:data:`SCORE_FLOOR`) with its item.
    non_finite: int = 0

    @property
    def score(self) -> float:
        """The mean of the four components; NaN when one was not scored (a
        lenient score, or one of chosen components)."""
        if any(name not in self.components for name in COMPONENTS):
            return math.nan
        return float(np.mean([self.components[name] for name in COMPONENTS]))


@dataclass(frozen=True)
class ControlScore:
    """Both strata's scores and the control score they mix to (FR-069)."""

    strata: Mapping[str, StratumScore]
    stratum_weight: float

    @property
    def control(self) -> float:
        """``W · card_disjoint + (1 − W) · game_disjoint``."""
        return mixed(self.strata[str(Stratum.CARD_DISJOINT)].score,
                     self.strata[str(Stratum.GAME_DISJOINT)].score, self.stratum_weight)


@dataclass(frozen=True)
class FrozenSelection:
    """What a training run's control score is built from: the corpus's
    frozen selection set, and a reader of each validation stratum it halves."""

    selection: SelectionSet
    strata: Mapping[str, StratumReader]

    @property
    def digest(self) -> str:
        return self.selection.digest


class ControlScorer:
    """The control score's cache and its per-epoch scoring."""

    def __init__(self, strata: dict[str, StratumCache], *, stratum_weight: float,
                 deviance_floor: float, strict: bool = True) -> None:
        self.strata = strata
        self.stratum_weight = stratum_weight
        self.deviance_floor = deviance_floor
        #: Whether a component with nothing to score raises (training) or is
        #: NaN and named (a saved checkpoint's report).
        self.strict = strict

    # ── startup ──

    @classmethod
    def prepare(
        cls, selection: SelectionSet, readers: Mapping[str, StratumReader], *,
        preparer: ScoringPreparer, fields: Sequence[tuple[FieldSpec, ...]],
        rule_outputs: Mapping[str, Sequence[str]], mask: str | None,
        stratum_weight: float, deviance_floor: float, editor: GridEditor | None = None,
        chunk_records: int = CHUNK_RECORDS, batch_size: int | None = None,
        half: str = SELECTION, components: Sequence[str] = COMPONENTS, strict: bool = True,
    ) -> ControlScorer:
        """Read each stratum's ``half`` once and prepare every item's host
        half (research E2).

        ``fields`` are every field set the run's epochs score (the curriculum's
        before and after its switch): the deviance steps are prepared over
        their union. ``mask`` is the run's ``--rule-mask`` (None reads it as
        ``hard``, the default). ``half`` is the selection half for a training
        run, which never reads the reporting half; ``components`` narrows what
        is prepared and scored. Raises :class:`ControlScoreError` on a
        component with no item to read, unless not ``strict``.
        """
        union = _field_union(fields)
        wanted = tuple(name for name in COMPONENTS if name in set(components))
        strata = {}
        for stratum in sorted(selection.halves):
            strata[stratum] = prepare_stratum(
                selection, stratum, readers[stratum], preparer=preparer, union=union,
                rule_outputs=rule_outputs, mask=mask or "hard", editor=editor,
                chunk_records=chunk_records, batch_size=batch_size or chunk_records,
                half=str(half), components=wanted, strict=strict,
            )
        scorer = cls(strata, stratum_weight=stratum_weight, deviance_floor=deviance_floor,
                     strict=strict)
        logger.info("Control score (%s half): %s, %.1f MB on the host.", half,
                    scorer.describe(), scorer.host_bytes() / 2**20)
        return scorer

    def describe(self) -> str:
        parts = []
        for name, cache in sorted(self.strata.items()):
            parts.append(
                f"{name} {len(cache.probes)} probe type(s), {len(cache.grids)} grid(s), "
                f"{len(cache.rules)} rule(s), {len(cache.kept_items)} deviance item(s) over "
                f"{sum(c.reads for c in cache.chunks)} read(s) and "
                f"{sum(int(s.targets.gate.shape[0]) for s in cache.steps)} deviance copies")
        return "; ".join(parts)

    def host_bytes(self) -> int:
        """The cache's bytes on the host."""
        return sum(host_bytes(chunk.batch) + chunk.index.nbytes
                   for cache in self.strata.values() for chunk in cache.chunks) + sum(
            host_bytes(step) for cache in self.strata.values() for step in cache.steps)

    # ── each epoch ──

    def epoch(self, forward: RecordForward, losses: StepLossComputer,
              fields: tuple[FieldSpec, ...]) -> ControlScore:
        """Both strata scored on the weights ``forward`` runs; ``fields`` is
        the epoch's field set. Raises :class:`ControlScoreError` on a
        component with nothing to score."""
        return ControlScore(
            {name: self._stratum(cache, forward, losses, fields)
             for name, cache in sorted(self.strata.items())},
            self.stratum_weight,
        )

    def _stratum(self, cache: StratumCache, forward: RecordForward,
                 losses: StepLossComputer, fields: tuple[FieldSpec, ...]) -> StratumScore:
        values = self._values(cache, forward)
        items = {
            PROBES: lambda: {name: probe_item(item.kind, item.labels, values[item.positions])
                             for name, item in sorted(cache.probes.items())},
            GRIDS: lambda: {name: _grid_item(item, values)
                            for name, item in sorted(cache.grids.items())},
            RULE_TESTS: lambda: {rule: rule_item(
                (_record_values(reading.fires, values), _record_values(reading.does_not, values))
                for reading in outputs.values()) for rule, outputs in sorted(cache.rules.items())},
            DEVIANCE: lambda: self._deviance(cache, forward, losses, fields),
        }
        items = {component: items[component]() for component in cache.components}
        left_out = {component: tuple(sorted(
            {name for name, value in by_item.items() if math.isnan(value)}
            | set(cache.left_out.get(component, ()))))
                    for component, by_item in items.items()}
        components = {component: component_score(items[component], stratum=cache.stratum,
                                                  component=component, half=cache.half,
                                                  strict=self.strict)
                      for component in cache.components}
        return StratumScore(
            components,
            {component: {name: clipped(value) for name, value in by_item.items()
                         if not math.isnan(value)} for component, by_item in items.items()},
            {component: names for component, names in left_out.items() if names},
            non_finite(values[_scored_positions(cache)]) + cache.unreadable_terms,
        )

    def _values(self, cache: StratumCache, forward: RecordForward) -> np.ndarray:
        values = np.full(cache.values, np.nan)
        if not cache.chunks:
            return values
        batches = (unpack(chunk.batch) for chunk in cache.chunks)
        positions, reads = [], []
        with forward.running():  # evaluation mode until every read is taken, however it ends
            for chunk, outputs in zip(cache.chunks, forward.prepared(batches)):
                if not chunk.groups:
                    continue
                heads = Heads(outputs, chunk.index)
                reads.append(torch.cat([group.read(heads, group.rows, group.columns).float()
                                        .reshape(-1) for group in chunk.groups]))
                positions.append(chunk.positions)
        if reads:  # one copy to the host per stratum, so no chunk waits on the device
            values[np.concatenate(positions)] = torch.cat(reads).cpu().numpy()
        return values

    def _deviance(self, cache: StratumCache, forward: RecordForward,
                  losses: StepLossComputer, fields: tuple[FieldSpec, ...]) -> dict[str, float]:
        """The deviance items; ``cache.unreadable_terms`` takes the number of
        kept terms whose loss is not finite."""
        if not cache.steps:
            return {}
        names = {spec.name for spec in fields}
        prepared = {spec.name for spec in cache.steps[0].fields}
        if not names <= prepared:
            raise ValueError(f"the deviance sample was prepared without {sorted(names - prepared)}")
        key = tuple(spec.name for spec in fields)
        # Every shipped head's terms, the per-entity fields' and the record
        # heads' alike, in nats per record over the sample's copies.
        parts = FieldParts(heads=True)
        kept = KeptTargets() if key not in cache.floors else None
        collectors = (parts,) if kept is None else (parts, kept)
        score_steps((unpack(replace(step, fields=fields)) for step in cache.steps),
                    forward, losses, collectors, device=_device_of(forward))
        if kept is not None:
            cache.floors[key] = deviance_items(
                constant_predictor_losses(kept.sample, fields=fields, weighted=True))
        means = parts.means()
        explained = explained_deviance(means, cache.floors[key],
                                       kept=cache.kept_items, deviance_floor=self.deviance_floor)
        cache.unreadable_terms = non_finite([means[name] for name in explained])
        if not explained and self.strict:
            raise ControlScoreError(
                cache.stratum, cache.half, DEVIANCE,
                f"every term's constant-prediction loss is at or below the deviance "
                f"floor {self.deviance_floor}, or no kept term is scored this epoch")
        return explained


def _device_of(forward: RecordForward) -> torch.device:
    materializer = forward.materializer
    return materializer.device if materializer is not None else torch.device("cpu")


def _scored_positions(cache: StratumCache) -> np.ndarray:
    """The value positions an item of the cache reads: a left-out output's
    readings, taken but scored by nothing, are not among them."""
    held = [item.positions for item in cache.probes.values()]
    held += [item.positions() for item in cache.grids.values()]
    held += [reading.positions() for outputs in cache.rules.values()
             for reading in outputs.values()]
    return np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in held]
                                    or [np.empty(0, dtype=np.int64)]))


def _record_values(records: list[list[int]], values: np.ndarray) -> list[float]:
    """Each record's reading, the mean over its entities' readings (not
    finite when any is: :func:`rule_item` scores that at the floor)."""
    return [float(values[positions].mean()) for positions in records if positions]


def _grid_item(item: GridItem, values: np.ndarray) -> float:
    """A grid's score: each record's reading in each cell against the rule's
    answer for that record there (``score_scales.grid_score``)."""
    return grid_score([values[positions] for positions in item.cells], item.answers)


def _field_union(sets: Sequence[tuple[FieldSpec, ...]]) -> tuple[FieldSpec, ...]:
    seen: dict[str, FieldSpec] = {}
    for specs in sets:
        for spec in specs:
            seen.setdefault(spec.name, spec)
    return tuple(seen.values())


# ── a saved checkpoint ──────────────────────────────────────────────────


#: The field set a saved checkpoint is scored over: every per-entity field,
#: the sparse group included, as the gates score a finished model.
CHECKPOINT_FIELDS: tuple[FieldSpec, ...] = active_fields(
    frozenset(str(klass) for klass in SamplingClass), sparse=True)


def score_half(
    selection: SelectionSet, readers: Mapping[str, StratumReader], *, half: str,
    encoder, model, batcher: SurfaceBatcher, rule_outputs: Mapping[str, Sequence[str]],
    mask: str | None, stratum_weight: float, deviance_floor: float,
    components: Sequence[str] = COMPONENTS, config=None,
) -> ControlScore:
    """A saved checkpoint's control score on ``half`` of every stratum
    ``selection`` halves, read once through the scorer training selects with
    (FR-070): ``encoder`` and ``model`` its modules, ``batcher`` a batcher of
    its own (its stager, join, layout and materializer), ``mask``,
    ``stratum_weight`` and ``deviance_floor`` the values it trained under.
    Lenient: a component with nothing to score is NaN and named in the
    stratum's ``left_out``, never an error. ``config`` builds the device
    half's loss (the training defaults when None; only the shipped loss and
    the parts in nats are read)."""
    from effects.application.step_loss import StepLossComputer
    from effects.application.train_effect_model import TrainEffectModelConfig
    from effects.domain.grid_edits import GridEdits

    preparer = CheckpointPreparer(batcher, mask=mask or "hard", rule_outputs=rule_outputs)
    scorer = ControlScorer.prepare(
        selection, readers, preparer=preparer, fields=[CHECKPOINT_FIELDS],
        rule_outputs=rule_outputs, mask=mask, stratum_weight=stratum_weight,
        deviance_floor=deviance_floor, editor=GridEdits(preparer.scoring_batcher()),
        half=half, components=components, strict=False,
    )
    device = batcher.materializer.device
    losses = StepLossComputer(config or TrainEffectModelConfig.unpathed(), device)
    return scorer.epoch(RecordForward(encoder, model, batcher.materializer), losses,
                        CHECKPOINT_FIELDS)


__all__ = [
    "CHECKPOINT_FIELDS", "CHUNK_RECORDS", "CheckpointPreparer", "ControlScore",
    "ControlScorer", "FrozenSelection", "StratumScore", "score_half",
]
