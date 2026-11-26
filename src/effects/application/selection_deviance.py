"""The constant prediction's loss on each record of a deviance sample, under
each rule-mask mode: the reader ``freeze_selection_set`` takes for an output
field's worst-case standard error (research E4).

E4's loss-ratio bound clusters ``w_i·ℓ_i`` by game, ``ℓ_i`` the constant
prediction's loss on record i and ``w_i`` the record's weight in the
component. Here ``ℓ_i`` is a head's own loss term at the sample's best
constant, fitted over the whole sample as ``loss_floor`` fits it (a rate,
class frequencies or a mean, a count's dispersion by moments and Newton
steps): for a per-entity field summed over the entities the record scores it
at, for a verdict or global field at the record, for a created-objects term
summed over the record's slots. ``w_i`` is the copy's ``curation.class_weight``
(FR-059a) times its weight under the mode (``rule_mask.copy_weights``): a
verdict or global field's own, the created-objects head's whole (the map
names that head whole); the affected gate, which the hard mask trains on a
drawn rule's entities alone, weighs the share of the record's gate entities
the rule names. Every constant is fitted with each copy at its class weight,
so a copy of class weight k reads as k repeated copies; the mode's weight
stays out of the fit, which serves both modes.

Every shipped head's terms are items, named by ``selection_set.deviance_item``
(a per-entity field by its name, any other head's term ``<head>.<term>``):
the terms the loss denominators divide by.

Targets come unmasked from the scoring step, and each mode's weight is
applied here, so one preparation serves both modes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path

import torch

from effects.domain.aux_heads import Head
from effects.domain.effect_head_layout import (
    CREATED_OBJECTS_WIDTH,
    FIELD_SLICES,
    GATE_TERM,
    GLOBAL_LAYOUT,
    PER_ENTITY_FIELDS,
    VERDICT_LAYOUT,
    FieldSpec,
    FieldType,
    RecordHeadLayout,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import FIELD_KINDS, FieldScoring, fit_log_dispersion
from effects.domain.input_layout import MAGNITUDE_CAP
from effects.domain.loss_floor import (
    created_objects_record_losses,
    record_head_record_losses,
    weighted_mean,
)
from effects.domain.records import EffectRecord
from effects.domain.rule_mask import CREATED_OBJECTS_OUTPUT
from effects.domain.selection_set import deviance_item

#: Records prepared per scoring step.
BATCH_SIZE = 64

#: The per-record heads whose fields are deviance items, with their layouts.
RECORD_HEADS: tuple[tuple[Head, RecordHeadLayout], ...] = (
    (Head.VERDICT, VERDICT_LAYOUT), (Head.GLOBAL, GLOBAL_LAYOUT),
)

#: One chunk's prepared targets: the per-entity head's, and each shipped
#: record head's ``(target, mask[, known])`` keyed by :class:`Head` (a
#: scoring step's ``targets`` and ``heads``).
PreparedTargets = tuple[EntityTargetBatch, Mapping]


@dataclass(frozen=True)
class DevianceSettings:
    """How :class:`TargetDevianceLosses` reads a sample: the per-entity
    ``fields`` it covers, the ``batch_size`` of a scoring step, the rule-mask
    ``modes`` it weighs under, and ``weights_of(record, mode)``, a copy's
    weights under a mode (``rule_mask.copy_weights`` where None)."""

    fields: tuple[FieldSpec, ...] = PER_ENTITY_FIELDS
    batch_size: int = BATCH_SIZE
    modes: Sequence[str] = ("hard", "soft")
    weights_of: Callable | None = None


class TargetDevianceLosses:
    """``DevianceLosses`` over every shipped head's targets: the per-entity
    head's gate and fields, the verdict and global heads' fields and the
    created-objects head's terms, each keyed by its item name
    (``selection_set.deviance_item``).

    ``prepare`` turns records into their unmasked targets (a scoring step's
    ``targets`` and ``heads``); ``rule_outputs`` is the manifest's
    ``rule -> outputs``, which the mask reads; ``settings`` is
    :class:`DevianceSettings`.
    """

    def __init__(
        self, prepare: Callable[[list[EffectRecord]], PreparedTargets],
        rule_outputs: Mapping[str, tuple[str, ...]], *,
        settings: DevianceSettings = DevianceSettings(),
    ) -> None:
        self.prepare = prepare
        self.rule_outputs = dict(rule_outputs)
        self.fields = settings.fields
        self.batch_size = settings.batch_size
        self.modes = tuple(settings.modes)
        self.weights_of = settings.weights_of or self._copy_weights

    def _copy_weights(self, record: EffectRecord, mode: str):
        from effects.domain.rule_mask import RuleMask, copy_weights

        return copy_weights(record, RuleMask(mode), self.rule_outputs)

    def losses(self, records: Iterable[EffectRecord]) -> dict[str, dict[str, list[float]]]:
        gathered = _Gathered(self.modes)
        chunk: list[EffectRecord] = []
        for record in records:
            chunk.append(record)
            if len(chunk) >= self.batch_size:
                self._fold(gathered, chunk)
                chunk = []
        self._fold(gathered, chunk)
        return gathered.losses({spec.name: spec for spec in self.fields})

    def _fold(self, gathered: _Gathered, chunk: list[EffectRecord]) -> None:
        """One chunk's targets, each copy's weights under every mode and its
        class weight."""
        if not chunk:
            return
        batch, head_targets = self.prepare(list(chunk))
        gathered.keep_targets(batch, head_targets, self.fields)
        for record in chunk:
            gathered.classes.append(_class_weight(record))
            for mode in self.modes:
                gathered.weights[mode].append(self.weights_of(record, mode))
        gathered.count += len(chunk)


class _Gathered:
    """What :meth:`TargetDevianceLosses.losses` reads off its chunks: each
    field's scored targets beside their record, each head's targets, each
    copy's weights by mode, its class weight and its gate entities."""

    def __init__(self, modes: Sequence[str]) -> None:
        self.modes = tuple(modes)
        self.rows: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        self.heads: dict[str, list[tuple[int, tuple]]] = {}
        self.weights: dict[str, list] = {mode: [] for mode in self.modes}
        self.classes: list[float] = []
        self.gate_entities: list[int] = []
        self.count = 0

    def keep_targets(self, batch: EntityTargetBatch, head_targets: Mapping | None,
                     fields: tuple[FieldSpec, ...]) -> None:
        scored = batch.gate_scored
        self.gate_entities.extend(int(n) for n in scored.sum(-1))
        _keep(self.rows, GATE_TERM, (scored, batch.gate.detach().cpu()), self.count)
        for spec in fields:
            if spec.name in batch.fields:
                _keep(self.rows, spec.name, (batch.scored(spec.name),
                      batch.fields.values[spec.name].detach().cpu()), self.count)
        for head, targets in (head_targets or {}).items():
            self.heads.setdefault(str(head), []).append((self.count, tuple(
                None if part is None else part.detach().cpu() for part in targets)))

    def losses(self, specs: Mapping[str, FieldSpec]) -> dict[str, dict[str, list[float]]]:
        """Each item's per-copy constant losses by mode, at the copy's class
        weight times its mode weight, the constant fitted at the class weights."""
        sample = _Sample(self.count, torch.tensor(self.classes, dtype=torch.float64))
        out = self._field_losses(specs, sample)
        out.update(self._head_item_losses(sample))
        return out

    def _field_losses(self, specs: Mapping[str, FieldSpec],
                      sample: _Sample) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        for name, parts in self.rows.items():
            index = torch.cat([part[0] for part in parts])
            target = torch.cat([part[1] for part in parts])
            per_record = _per_record_losses(specs.get(name), target, index, sample
                                            ) * sample.classes
            out[name] = {mode: (per_record * torch.tensor(
                self._mode_weights(name, mode), dtype=torch.float64)).tolist()
                for mode in self.modes}
        return out

    def _head_item_losses(self, sample: _Sample) -> dict[str, dict[str, list[float]]]:
        out: dict[str, dict[str, list[float]]] = {}
        for head, by_term in _head_losses(self.heads, sample).items():
            # The map names the created-objects head whole.
            whole = head == str(Head.CREATED_OBJECTS)
            for term, per_record in by_term.items():
                output = CREATED_OBJECTS_OUTPUT if whole else term
                out[deviance_item(head, term)] = {mode: (
                    per_record * sample.classes * torch.tensor(
                        [cw.weight(output) for cw in self.weights[mode]], dtype=torch.float64,
                    )).tolist() for mode in self.modes}
        return out

    def _mode_weights(self, name: str, mode: str) -> list[float]:
        if name == GATE_TERM:
            return [_gate_weight(cw, n) for cw, n in zip(self.weights[mode], self.gate_entities)]
        return [cw.weight(name) for cw in self.weights[mode]]


def _class_weight(record: EffectRecord) -> float:
    """The copy's ``curation.class_weight`` (FR-059a), one with no curation."""
    curation = getattr(record, "curation", None)
    return 1.0 if curation is None else float(curation.class_weight)


@dataclass(frozen=True)
class _Sample:
    """The deviance sample's size and each record's class weight (one a
    record), the weights the constants are fitted at."""

    records: int
    classes: torch.Tensor


def _head_losses(heads: Mapping[str, list[tuple[int, tuple]]],
                 sample: _Sample) -> dict[str, dict[str, torch.Tensor]]:
    """Each shipped record head's per-record constant losses by term, over
    every record (zero where a record does not supervise the term), the
    constant fitted at each record's class weight."""
    out: dict[str, dict[str, torch.Tensor]] = {}
    for head, layout in RECORD_HEADS:
        held = heads.get(str(head))
        found = _record_head_found(held, layout, sample) if held else None
        if found:
            out[str(head)] = found
    held = heads.get(str(Head.CREATED_OBJECTS))
    found = _created_objects_found(held, sample) if held else None
    if found:
        out[str(Head.CREATED_OBJECTS)] = found
    return out


def _record_head_found(held: list[tuple[int, tuple]], layout: RecordHeadLayout,
                       sample: _Sample) -> dict[str, torch.Tensor]:
    target = torch.zeros(sample.records, layout.target_width)
    mask = torch.zeros(sample.records, len(layout.fields), dtype=torch.bool)
    for offset, (values, flags, *_) in held:
        flags = flags != 0
        if flags.dim() == 1:
            flags = flags.unsqueeze(-1).expand(-1, len(layout.fields))
        target[offset:offset + values.shape[0]] = values.float()
        mask[offset:offset + flags.shape[0]] = flags
    return record_head_record_losses(target, mask, layout, sample.classes)


def _created_objects_found(held: list[tuple[int, tuple]],
                           sample: _Sample) -> dict[str, torch.Tensor]:
    target = torch.zeros(sample.records, CREATED_OBJECTS_WIDTH)
    known = torch.ones(sample.records, CREATED_OBJECTS_WIDTH)
    mask = torch.zeros(sample.records, dtype=torch.bool)
    for offset, (values, flags, *rest) in held:
        size = values.shape[0]
        target[offset:offset + size] = values.float()
        mask[offset:offset + size] = flags != 0
        if rest and rest[0] is not None:
            known[offset:offset + size] = (rest[0] != 0).float()
    return created_objects_record_losses(target, mask, known, sample.classes)


def _keep(rows: dict, name: str, targets: tuple[torch.Tensor, torch.Tensor],
          offset: int) -> None:
    """The scored targets of one field (``(scored, values)``), beside the
    record each belongs to."""
    scored, values = targets
    if not bool(scored.any()):
        return
    rows.setdefault(name, []).append(
        (scored.nonzero()[:, 0] + offset, values[scored]))


def _per_record_losses(spec: FieldSpec | None, target: torch.Tensor, index: torch.Tensor,
                       sample: _Sample) -> torch.Tensor:
    """The field's constant-prediction loss summed per record, the constant
    fitted with each target at its record's class weight (``sample.classes``,
    one a record), so a copy of weight k fits as k repeated copies; zero for a field
    whose targets never vary, which has no deviance to explain (the floor
    reports it as zero)."""
    totals = torch.zeros(sample.records, dtype=torch.float64)
    if target.numel() == 0 or bool((target == target[0]).all()):
        return totals
    kind_type = FieldType.BINARY if spec is None else spec.type
    arity = 1 if spec is None else spec.arity
    kind = FIELD_KINDS[kind_type]
    weights = sample.classes[index]
    constant = kind.best_constant(target, arity=arity, cap=MAGNITUDE_CAP, weights=weights)
    constant = torch.as_tensor(constant, dtype=torch.float32).reshape(-1)
    if spec is not None:
        start, end = FIELD_SLICES[spec.name]
        width = end - start
    else:
        width = 1
    dispersion = None
    if spec is not None and spec.type.has_magnitude:
        magnitude = kind.magnitude(target, cap=MAGNITUDE_CAP)
        dispersion = fit_log_dispersion(magnitude, weighted_mean(magnitude, weights), weights)
    terms = kind.terms(constant.expand(target.shape[0], width), target,
                       FieldScoring(arity, MAGNITUDE_CAP, dispersion))
    return totals.index_add_(0, index, terms.detach().double().reshape(-1))


def _gate_weight(weights, scored_entities: int) -> float:
    """The gate's weight on one copy: its output weight, or under the hard
    mask the share of the record's gate entities the drawn rule names."""
    if weights.gate_entities is None:
        return weights.weight(GATE_TERM)
    if not scored_entities:
        return 0.0
    return min(1.0, len(weights.gate_entities) / scored_entities)


def scoring_targets(corpus: Path, manifest, *,
                    cards_folders: Sequence[Path] | None = None) -> Callable:
    """``records -> (per-entity targets, head targets)``, a scoring step over ``corpus``,
    built as a training run over it would be, its script-API head off and no
    rule mask applied.

    It reads no keyword table: the step's targets and heads come from the
    records alone, and the table feeds only the encoder's input text, which
    this step never builds (``tests/unit/effects/application/test_scoring_targets.py``
    holds the targets equal with and without it)."""
    from effects.application.run_context import preparer_settings
    from effects.application.step_preparation import StepPreparer
    from effects.application.train_effect_model import (
        CorpusFacts,
        RunSpec,
        TrainEffectModelConfig,
        with_variant_scripts,
    )

    options = {"corpus": str(corpus), "vocab_path": Path(manifest.vocab_path),
               "api_weight": 0.0,
               "keyword_definitions": None}
    if cards_folders:
        options["cards_folders"] = tuple(Path(folder) for folder in cards_folders)
    config = with_variant_scripts(TrainEffectModelConfig.unpathed(**options), manifest)
    if "rule_mask" in {f.name for f in dataclass_fields(config)}:
        config = replace(config, rule_mask=None)
    spec = RunSpec(config, CorpusFacts.of(manifest), manifest.seed)
    preparer = StepPreparer(preparer_settings(spec))

    def prepare(records: list[EffectRecord]) -> PreparedTargets:
        step = preparer.prepare_scoring_step(records, PER_ENTITY_FIELDS)
        return step.targets, step.heads

    return prepare
