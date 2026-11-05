"""The constant prediction's loss on each record of a deviance sample, under
each rule-mask mode: the reader ``freeze_selection_set`` takes for an output
field's worst-case standard error (research E4).

E4's loss-ratio bound clusters ``w_i·ℓ_i`` by game, ``ℓ_i`` the constant
prediction's loss on record i and ``w_i`` the record's weight in the
component. Here ``ℓ_i`` is the per-entity head's own loss term at the
sample's best constant, fitted over the whole sample as
``loss_floor.constant_predictor_outputs`` fits it (a rate, class frequencies
or a mean, a count's dispersion by moments and Newton steps), summed over the
entities the record scores the field at. ``w_i`` is the copy's weight under
the mode (``rule_mask.copy_weights``); the affected gate, which the hard mask
trains on a drawn rule's entities alone, weighs the share of the record's
gate entities the rule names.

Targets come unmasked from the scoring step, and each mode's weight is
applied here, so one preparation serves both modes.

The per-entity head covers the gate and every per-entity field; the verdict,
global and created-objects heads are not items here yet.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path

import torch

from effects.domain.effect_head_input import MAGNITUDE_CAP
from effects.domain.effect_head_layout import (
    FIELD_SLICES,
    GATE_TERM,
    PER_ENTITY_FIELDS,
    FieldSpec,
    FieldType,
)
from effects.domain.entity_field_targets import EntityTargetBatch
from effects.domain.field_kinds import FIELD_KINDS, fit_log_dispersion
from effects.domain.records import EffectRecord

#: Records prepared per scoring step.
BATCH_SIZE = 64


class TargetDevianceLosses:
    """``DevianceLosses`` over the per-entity head's targets.

    ``prepare`` turns records into their unmasked per-entity targets (a
    scoring step's ``targets``); ``rule_outputs`` is the manifest's
    ``rule -> outputs``, which the mask reads. ``weights_of(record, mode)``
    is a copy's weights under a mode, ``rule_mask.copy_weights`` by default.
    """

    def __init__(
        self, prepare: Callable[[list[EffectRecord]], EntityTargetBatch],
        rule_outputs: Mapping[str, tuple[str, ...]], *,
        fields: tuple[FieldSpec, ...] = PER_ENTITY_FIELDS, batch_size: int = BATCH_SIZE,
        modes: Sequence[str] = ("hard", "soft"),
        weights_of: Callable | None = None,
    ) -> None:
        self.prepare = prepare
        self.rule_outputs = dict(rule_outputs)
        self.fields = fields
        self.batch_size = batch_size
        self.modes = tuple(modes)
        self.weights_of = weights_of or self._copy_weights

    def _copy_weights(self, record: EffectRecord, mode: str):
        from effects.domain.rule_mask import RuleMask, copy_weights

        return copy_weights(record, RuleMask(mode), self.rule_outputs)

    def losses(self, records: Iterable[EffectRecord]) -> dict[str, dict[str, list[float]]]:
        rows: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        weights: dict[str, list] = {mode: [] for mode in self.modes}
        gate_entities: list[int] = []
        count = 0
        chunk: list[EffectRecord] = []

        def flush() -> None:
            nonlocal count
            if not chunk:
                return
            batch = self.prepare(list(chunk))
            scored = batch.gate_scored
            gate_entities.extend(int(n) for n in scored.sum(-1))
            _keep(rows, GATE_TERM, scored, batch.gate.detach().cpu(), count)
            for spec in self.fields:
                if spec.name in batch.fields:
                    _keep(rows, spec.name, batch.scored(spec.name),
                          batch.fields.values[spec.name].detach().cpu(), count)
            for record in chunk:
                for mode in self.modes:
                    weights[mode].append(self.weights_of(record, mode))
            count += len(chunk)
            chunk.clear()

        for record in records:
            chunk.append(record)
            if len(chunk) >= self.batch_size:
                flush()
        flush()

        out: dict[str, dict[str, list[float]]] = {}
        specs = {spec.name: spec for spec in self.fields}
        for name, parts in rows.items():
            index = torch.cat([part[0] for part in parts])
            target = torch.cat([part[1] for part in parts])
            spec = specs.get(name)
            per_record = _per_record_losses(spec, target, index, count)
            out[name] = {}
            for mode in self.modes:
                if name == GATE_TERM:
                    w = [_gate_weight(cw, n) for cw, n in zip(weights[mode], gate_entities)]
                else:
                    w = [cw.weight(name) for cw in weights[mode]]
                out[name][mode] = (per_record * torch.tensor(w, dtype=torch.float64)).tolist()
        return out


def _keep(rows: dict, name: str, scored: torch.Tensor, values: torch.Tensor,
          offset: int) -> None:
    """The scored targets of one field, beside the record each belongs to."""
    if not bool(scored.any()):
        return
    rows.setdefault(name, []).append(
        (scored.nonzero()[:, 0] + offset, values[scored]))


def _per_record_losses(spec: FieldSpec | None, target: torch.Tensor, index: torch.Tensor,
                       records: int) -> torch.Tensor:
    """The field's constant-prediction loss summed per record; zero for a field
    whose targets never vary, which has no deviance to explain (the floor
    reports it as zero)."""
    totals = torch.zeros(records, dtype=torch.float64)
    if target.numel() == 0 or bool((target == target[0]).all()):
        return totals
    kind_type = FieldType.BINARY if spec is None else spec.type
    arity = 1 if spec is None else spec.arity
    kind = FIELD_KINDS[kind_type]
    constant = kind.best_constant(target, arity=arity, cap=MAGNITUDE_CAP)
    constant = torch.as_tensor(constant, dtype=torch.float32).reshape(-1)
    if spec is not None:
        start, end = FIELD_SLICES[spec.name]
        width = end - start
    else:
        width = 1
    dispersion = None
    if spec is not None and spec.type.has_magnitude:
        magnitude = kind.magnitude(target, cap=MAGNITUDE_CAP)
        dispersion = fit_log_dispersion(magnitude, magnitude.mean())
    terms = kind.terms(constant.expand(target.shape[0], width), target, arity=arity,
                       cap=MAGNITUDE_CAP, log_dispersion=dispersion)
    return totals.index_add_(0, index, terms.detach().double().reshape(-1))


def _gate_weight(weights, scored_entities: int) -> float:
    """The gate's weight on one copy: its output weight, or under the hard
    mask the share of the record's gate entities the drawn rule names."""
    if weights.gate_entities is None:
        return weights.weight(GATE_TERM)
    if not scored_entities:
        return 0.0
    return min(1.0, len(weights.gate_entities) / scored_entities)


def scoring_targets(corpus: Path, manifest, *, cards_folders: Sequence[Path] | None = None,
                    sample: Sequence[EffectRecord] = ()) -> Callable:
    """``records -> per-entity targets`` from a scoring step over ``corpus``,
    built as a training run over it would be, its script-API head off and no
    rule mask applied; ``sample`` sizes the feature widths."""
    from effects.application.run_context import preparer_settings
    from effects.application.step_preparation import StepPreparer
    from effects.application.train_effect_model import (
        CorpusFacts,
        RunSpec,
        TrainEffectModelConfig,
        with_variant_scripts,
    )
    from effects.domain.effect_head_input import feature_widths

    options = {"corpus": str(corpus), "vocab_path": Path(manifest.vocab_path),
               "api_weight": 0.0}
    if cards_folders:
        options["cards_folders"] = tuple(Path(folder) for folder in cards_folders)
    config = with_variant_scripts(TrainEffectModelConfig(**options), manifest)
    if "rule_mask" in {f.name for f in dataclass_fields(config)}:
        config = replace(config, rule_mask=None)
    spec = RunSpec(config, CorpusFacts.of(manifest), manifest.seed)
    preparer = StepPreparer(preparer_settings(spec, widths=feature_widths(list(sample))))

    def prepare(records: list[EffectRecord]) -> EntityTargetBatch:
        return preparer.prepare_scoring_step(records, PER_ENTITY_FIELDS).targets

    return prepare
